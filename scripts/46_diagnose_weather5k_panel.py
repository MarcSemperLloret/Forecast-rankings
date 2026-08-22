#!/usr/bin/env python3
"""Diagnose the global engineering panel without claiming confirmation.

This applies the frozen site deduplication and equal-area cell weighting, then
reports source-variant errors.  It deliberately does not fit the primary
roughness regression: the seven source variants are not seven independent
model families and the dynamic global roughness table is not yet materialised.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from tessellation import EqualAreaGrid

CONFIG = ROOT / "config" / "regional_t2m_2020.yaml"
DEFAULT_PANEL = ROOT / "data" / "interim" / "weather5k_weatherbench2_dry_run_2020" / "panel.parquet"
DEFAULT_OUTPUT = ROOT / "results" / "weather5k_weatherbench2_dry_run_2020"


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--analysis-role",
        default="engineering_dry_run_only_not_confirmatory_evidence",
    )
    parser.add_argument(
        "--observation-limit",
        default="WEATHER-5K/ISD may overlap observations assimilated by ERA5.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    panel = args.panel.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    known = [output / "sites.csv", output / "scores_by_source_variant.csv",
             output / "cell_day_errors.parquet", output / "diagnostics.json"]
    if any(path.exists() for path in known) and not args.force:
        raise FileExistsError(f"Diagnostic output exists in {output}; use --force")
    if args.force:
        for path in known:
            if path.exists():
                if not path.is_file():
                    raise RuntimeError(f"Refusing to replace non-file: {path}")
                path.unlink()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    connection = duckdb.connect()
    stations = connection.execute(
        f"""SELECT DISTINCT station_id, latitude, longitude, altitude_m, network
            FROM read_parquet('{panel.as_posix()}') ORDER BY station_id"""
    ).fetchdf()
    stations["station_id"] = stations.station_id.astype("string")
    build_sites = importlib.import_module("41_deduplicate_stations").build_sites
    sites, kept_apart = build_sites(stations, cfg["station_deduplication"])
    grid = EqualAreaGrid(float(cfg["spatial_weighting"]["target_cell_km"]))
    sites["cell_id"] = grid.assign(sites.mean_latitude.to_numpy(), sites.mean_longitude.to_numpy())
    membership = sites[["site_id", "cell_id", "members"]].copy()
    membership["station_id"] = membership.members.str.split("|")
    membership = membership.explode("station_id")
    membership["member_priority"] = membership.groupby("site_id").cumcount()
    membership = membership.drop(columns="members")
    connection.register("site_membership", membership)
    connection.execute("CREATE TABLE membership AS SELECT * FROM site_membership")
    sites.to_csv(output / "sites.csv", index=False)

    n_models = connection.execute(
        f"SELECT count(DISTINCT model) FROM read_parquet('{panel.as_posix()}')"
    ).fetchone()[0]
    connection.execute(
        f"""
        CREATE TEMP TABLE selected AS
        WITH ranked AS (
            SELECT p.*, m.site_id, m.cell_id,
                   row_number() OVER (
                       PARTITION BY m.site_id, p.model, p.valid_time
                       ORDER BY m.member_priority, p.station_id
                   ) AS member_rank
            FROM read_parquet('{panel.as_posix()}') p
            JOIN membership m USING (station_id)
        ), deduplicated AS (
            SELECT * EXCLUDE (member_rank) FROM ranked WHERE member_rank = 1
        ), common_cases AS (
            SELECT site_id, valid_time
            FROM deduplicated
            GROUP BY ALL
            HAVING count(DISTINCT model) = {int(n_models)}
        )
        SELECT d.*
        FROM deduplicated d
        JOIN common_cases c USING (site_id, valid_time)
        """
    )
    connection.execute(
        """
        CREATE TEMP TABLE site_day AS
        SELECT model, model_family, cohort, cell_id, site_id,
               cast(valid_time AS DATE) AS valid_day,
               avg(abs(forecast_t2m_c - era5_t2m_c)) AS era5_mae_c,
               avg(abs(forecast_t2m_c - observed_t2m_c)) AS observed_mae_c,
               avg(pow(forecast_t2m_c - era5_t2m_c, 2)) AS era5_mse_c2,
               avg(pow(forecast_t2m_c - observed_t2m_c, 2)) AS observed_mse_c2
        FROM selected
        GROUP BY ALL
        """
    )
    cell_day_path = output / "cell_day_errors.parquet"
    connection.execute(
        f"""COPY (
            SELECT model, model_family, cohort, cell_id, valid_day,
                   avg(era5_mae_c) AS era5_mae_c,
                   avg(observed_mae_c) AS observed_mae_c,
                   avg(era5_mse_c2) AS era5_mse_c2,
                   avg(observed_mse_c2) AS observed_mse_c2,
                   count(*) AS sites
            FROM site_day GROUP BY ALL ORDER BY model, cell_id, valid_day
        ) TO '{cell_day_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)"""
    )
    scores = connection.execute(
        f"""
        WITH by_cell AS (
            SELECT model, model_family, cohort, cell_id,
                   avg(era5_mae_c) AS era5_mae_c,
                   avg(observed_mae_c) AS observed_mae_c,
                   avg(era5_mse_c2) AS era5_mse_c2,
                   avg(observed_mse_c2) AS observed_mse_c2
            FROM read_parquet('{cell_day_path.as_posix()}') GROUP BY ALL
        ), weighted AS (
            SELECT model, model_family, cohort,
                   avg(era5_mae_c) AS era5_mae_equal_cell_c,
                   avg(observed_mae_c) AS observed_mae_equal_cell_c,
                   sqrt(avg(era5_mse_c2)) AS era5_rmse_equal_cell_c,
                   sqrt(avg(observed_mse_c2)) AS observed_rmse_equal_cell_c,
                   count(*) AS occupied_cells
            FROM by_cell GROUP BY ALL
        ), unweighted AS (
            SELECT model,
                   avg(era5_mae_c) AS era5_mae_equal_site_day_c,
                   avg(observed_mae_c) AS observed_mae_equal_site_day_c,
                   sqrt(avg(era5_mse_c2)) AS era5_rmse_equal_site_day_c,
                   sqrt(avg(observed_mse_c2)) AS observed_rmse_equal_site_day_c,
                   count(*) AS site_days
            FROM site_day GROUP BY model
        )
        SELECT w.*, u.era5_mae_equal_site_day_c, u.observed_mae_equal_site_day_c,
               u.site_days,
               w.era5_mae_equal_cell_c - w.observed_mae_equal_cell_c
                   AS gridded_reference_advantage_c,
               w.era5_rmse_equal_cell_c - w.observed_rmse_equal_cell_c
                   AS gridded_reference_rmse_advantage_c
        FROM weighted w JOIN unweighted u USING (model)
        ORDER BY model
        """
    ).fetchdf()
    scores_path = output / "scores_by_source_variant.csv"
    scores.to_csv(scores_path, index=False)

    row_counts = connection.execute(
        """SELECT
             (SELECT count(*) FROM selected),
             (SELECT count(DISTINCT (site_id, valid_time)) FROM selected),
             (SELECT count(DISTINCT site_id) FROM selected),
             (SELECT count(DISTINCT cell_id) FROM selected),
             (SELECT min(valid_time) FROM selected),
             (SELECT max(valid_time) FROM selected)
        """
    ).fetchone()
    connection.close()
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": args.analysis_role,
        "panel": str(panel),
        "panel_sha256": sha256(panel),
        "config_sha256": sha256(CONFIG),
        "source_variants": int(n_models),
        "station_records": int(len(stations)),
        "deduplicated_sites": int(len(sites)),
        "same_site_records_removed": int(len(stations) - len(sites)),
        "near_pairs_kept_apart_for_elevation": int(len(kept_apart)),
        "common_selected_rows": int(row_counts[0]),
        "common_site_time_cases": int(row_counts[1]),
        "common_sites": int(row_counts[2]),
        "occupied_equal_area_cells": int(row_counts[3]),
        "first_valid_time": row_counts[4].isoformat(),
        "last_valid_time": row_counts[5].isoformat(),
        "grid": grid.summary(),
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (output / "sites.csv", scores_path, cell_day_path)
        },
        "confirmatory_regression_run": False,
        "reason_not_confirmatory": [
            args.observation_limit,
            f"The {int(n_models)} source variants must be interpreted through their named families and cohorts.",
            "Dynamic model-cell-day roughness has not yet been materialised with a globally isotropic filter.",
        ],
    }
    (output / "diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(scores.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(json.dumps({key: report[key] for key in (
        "deduplicated_sites", "common_site_time_cases", "occupied_equal_area_cells",
        "confirmatory_regression_run")}, indent=2))


if __name__ == "__main__":
    main()
