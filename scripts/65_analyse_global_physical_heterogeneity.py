#!/usr/bin/env python3
"""Test pre-specified physical heterogeneity predictions on global cells."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ERRORS = ROOT / "results" / "weather5k_weatherbench2_metrics_v2_2020" / "cell_day_errors.parquet"
DEFAULT_COVARIATES = ROOT / "data" / "interim" / "weather5k_physical_covariates_2020" / "cell_covariates.csv"
DEFAULT_OUTPUT = ROOT / "results" / "weather5k_physical_heterogeneity_2020"
DEFAULT_MODELS = "ifs_hres,graphcast_hres_init,pangu_hres_init"
PREDICTED = [
    "absolute_elevation_mismatch_m",
    "era5_orography_sd_m",
    "era5_landsea_heterogeneity_100km",
]
DESCRIPTIVE = [
    "era5_filtered_subgrid_orography_sd_m",
    "era5_subgrid_orography_slope",
    "absolute_latitude_deg",
    "sites",
]


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def interval(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0 or not np.isfinite(values).any():
        return {"bootstrap_median": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    return {
        "bootstrap_median": float(np.nanmedian(values)),
        "ci_low": float(np.nanquantile(values, 0.025)),
        "ci_high": float(np.nanquantile(values, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--errors", type=Path, default=DEFAULT_ERRORS)
    parser.add_argument("--covariates", type=Path, default=DEFAULT_COVARIATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    errors, covariates, output = args.errors.resolve(), args.covariates.resolve(), args.output.resolve()
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    output.mkdir(parents=True, exist_ok=True)
    names = ["cell_effects.csv", "associations.csv", "quartile_contrasts.csv", "summary.json"]
    if any((output / name).exists() for name in names) and not args.force:
        raise FileExistsError(f"outputs exist in {output}; use --force")
    if args.force:
        for name in [*names, "bootstrap_associations.npz", "outputs_manifest.json"]:
            path = output / name
            if path.exists():
                if not path.is_file():
                    raise RuntimeError(f"refusing to replace non-file {path}")
                path.unlink()

    connection = duckdb.connect()
    model_sql = ",".join("?" for _ in models)
    effects = connection.execute(
        f"""
        WITH model_cell AS (
            SELECT model, cell_id,
                   avg(era5_mae_c) - avg(observed_mae_c) AS mae_advantage_c,
                   sqrt(avg(era5_mse_c2)) - sqrt(avg(observed_mse_c2)) AS rmse_advantage_c,
                   count(DISTINCT valid_day) AS days
            FROM read_parquet('{errors.as_posix()}')
            WHERE model IN ({model_sql})
            GROUP BY ALL
        ), complete AS (
            SELECT cell_id FROM model_cell GROUP BY cell_id
            HAVING count(DISTINCT model) = {len(models)}
        )
        SELECT m.cell_id,
               avg(m.mae_advantage_c) AS mae_advantage_c,
               avg(m.rmse_advantage_c) AS rmse_advantage_c,
               min(m.days) AS minimum_days,
               max(m.days) AS maximum_days
        FROM model_cell m JOIN complete USING (cell_id)
        GROUP BY m.cell_id ORDER BY m.cell_id
        """,
        models,
    ).fetchdf()
    connection.close()
    cov = pd.read_csv(covariates, dtype={"cell_id": "string"})
    frame = effects.merge(cov, on="cell_id", how="inner", validate="one_to_one")
    if len(frame) != len(effects):
        raise RuntimeError("one or more error cells lack pre-specified covariates")
    frame["spatial_block"] = (
        np.floor((frame.mean_latitude + 90.0) / 10.0).astype(int).astype(str)
        + "_"
        + np.floor((np.mod(frame.mean_longitude + 180.0, 360.0)) / 10.0).astype(int).astype(str)
    )
    frame.to_csv(output / "cell_effects.csv", index=False)

    rng = np.random.default_rng(args.seed)
    block_groups = {name: index.to_numpy() for name, index in frame.groupby("spatial_block").groups.items()}
    block_names = sorted(block_groups)
    bootstrap_indices = []
    for _ in range(args.bootstrap_resamples):
        selected = rng.choice(block_names, len(block_names), replace=True)
        bootstrap_indices.append(np.concatenate([block_groups[name] for name in selected]))

    association_rows, quartile_rows = [], []
    bootstrap_store: dict[str, np.ndarray] = {}
    for metric in ("mae", "rmse"):
        y = frame[f"{metric}_advantage_c"].to_numpy(float)
        for covariate in [*PREDICTED, *DESCRIPTIVE]:
            x = frame[covariate].to_numpy(float)
            values = np.array([
                spearmanr(x[index], y[index]).statistic for index in bootstrap_indices
            ])
            key = f"{metric}_{covariate}_spearman"
            bootstrap_store[key] = values
            row = {
                "metric": metric,
                "covariate": covariate,
                "prediction": "negative" if covariate in PREDICTED else "descriptive",
                "spearman": float(spearmanr(x, y).statistic),
            }
            row.update(interval(values))
            row["predicted_sign_passed"] = (
                bool(row["spearman"] < 0) if covariate in PREDICTED else None
            )
            association_rows.append(row)

            quartile = np.asarray(pd.qcut(x, 4, labels=False, duplicates="drop"), dtype=float)
            bins = np.unique(quartile[np.isfinite(quartile)])
            if len(bins) >= 2:
                low, high = quartile == bins.min(), quartile == bins.max()
                contrast = float(y[high].mean() - y[low].mean())
                contrast_draws = np.array([
                    y[index][high[index]].mean() - y[index][low[index]].mean()
                    for index in bootstrap_indices
                    if high[index].any() and low[index].any()
                ])
            else:
                low = high = np.zeros(len(x), dtype=bool)
                contrast = float("nan")
                contrast_draws = np.array([], dtype=float)
            bootstrap_store[f"{metric}_{covariate}_q4_minus_q1"] = contrast_draws
            qrow = {
                "metric": metric,
                "covariate": covariate,
                "prediction": "negative" if covariate in PREDICTED else "descriptive",
                "q4_minus_q1_advantage_c": contrast,
                "q1_cells": int(low.sum()),
                "q4_cells": int(high.sum()),
            }
            qrow.update(interval(contrast_draws))
            qrow["predicted_sign_passed"] = (
                bool(contrast < 0) if covariate in PREDICTED else None
            )
            quartile_rows.append(qrow)

    associations = pd.DataFrame(association_rows)
    quartiles = pd.DataFrame(quartile_rows)
    associations.to_csv(output / "associations.csv", index=False)
    quartiles.to_csv(output / "quartile_contrasts.csv", index=False)
    np.savez_compressed(output / "bootstrap_associations.npz", **bootstrap_store)
    predicted_associations = associations[associations.covariate.isin(PREDICTED)]
    predicted_quartiles = quartiles[quartiles.covariate.isin(PREDICTED)]
    gates = {
        "negative_spearman_both_metrics_all_three_predicted_covariates": bool(
            predicted_associations.predicted_sign_passed.all()
        ),
        "negative_q4_minus_q1_both_metrics_all_three_predicted_covariates": bool(
            predicted_quartiles.predicted_sign_passed.all()
        ),
        "all_predicted_spearman_95ci_below_zero": bool(
            (predicted_associations.ci_high < 0).all()
        ),
        "all_predicted_q4_minus_q1_95ci_below_zero": bool(
            (predicted_quartiles.ci_high < 0).all()
        ),
    }
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "pre-specified global physical heterogeneity test",
        "models": models,
        "model_reduction": "cell effect calculated per model, then averaged equally across selected models",
        "support": {
            "cells": int(len(frame)),
            "spatial_blocks_10deg": int(frame.spatial_block.nunique()),
            "minimum_days_per_model_cell": int(frame.minimum_days.min()),
            "maximum_days_per_model_cell": int(frame.maximum_days.max()),
        },
        "predicted_covariates": PREDICTED,
        "descriptive_covariates": DESCRIPTIVE,
        "bootstrap": {
            "unit": "10-degree latitude-longitude block",
            "resamples": args.bootstrap_resamples,
            "seed": args.seed,
        },
        "gates": gates,
        "interpretation_limit": (
            "Physical heterogeneity associations explain where the reference effect is strongest; "
            "they are not additional causal model-family replications and WEATHER-5K/ISD may overlap ERA5 assimilation."
        ),
        "inputs": {
            "errors": {"path": str(errors), "sha256": sha256(errors)},
            "covariates": {"path": str(covariates), "sha256": sha256(covariates)},
        },
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    output_files = [*names, "bootstrap_associations.npz"]
    (output / "outputs_manifest.json").write_text(json.dumps({
        "script_sha256": sha256(Path(__file__).resolve()),
        "outputs": {
            name: {"bytes": (output / name).stat().st_size, "sha256": sha256(output / name)}
            for name in output_files
        },
    }, indent=2), encoding="utf-8")
    print(associations.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(quartiles.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    main()
