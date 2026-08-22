#!/usr/bin/env python3
"""The single confirmatory run. One command, no decisions left inside.

Everything this script needs was fixed before the global archive existed: the
scales, the radii, the minima, the bootstrap widths, the tessellation, the
deduplication policy. It verifies that seal before computing anything and stops
if a parameter moved, so changing a scale after seeing a result breaks the run
instead of passing unnoticed.

    python scripts/43_confirmatory_analysis.py --panel PANEL.parquet \
        --roughness ROUGHNESS.csv --output results/global

Input contract for the panel, one row per site, valid time and model:

    station_id, latitude, longitude, altitude_m, network (optional),
    valid_time, model, forecast_t2m_c, era5_t2m_c, observed_t2m_c

and optionally forecast_sigma_{s}km_c columns, one per pre-registered blur
width. When they are present the two secondary hypotheses that need them run;
when they are absent the run says so rather than skipping quietly.

Input contract for a held-out confirmatory roughness table, one row per model,
equal-area cell and valid day:

    model, cell_id, valid_day,
    roughness_25km_k, roughness_50km_k, roughness_100km_k

The cell-day form is mandatory outside ``--self-test`` so spatial and temporal
bootstrap draws recalculate both sides of the regression. The development-set
self-test may use one already aggregated row per model because it diagnoses
code execution rather than estimating confirmatory uncertainty.

The primary result is the slope of the advantage the gridded reference concedes,
B_m = E_grid,m − E_station,m, on the roughness of the model's field. Confirmation
is a positive slope: the more small-scale structure a field keeps, the less the
gridded reference flatters it.

``--self-test`` runs the whole thing on the pilot's own panel, which is how the
script is known to work before any global data arrives.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tessellation import EqualAreaGrid
from validation import check_coordinates, check_elevation, check_temperature_units, summarise

CONFIG = ROOT / "config" / "regional_t2m_2020.yaml"
REQUIRED = ["station_id", "latitude", "longitude", "valid_time", "model",
            "forecast_t2m_c", "era5_t2m_c", "observed_t2m_c"]


def verify_seal() -> str:
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "40_freeze_analysis_parameters.py"), "--verify"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"the sealed parameters do not match the lock; refusing to run\n{result.stderr}")
    return result.stdout.strip()


def load_panel(path: Path) -> pd.DataFrame:
    frame = (pd.read_csv(path) if path.suffix == ".csv"
             else duckdb.sql(f"SELECT * FROM read_parquet('{path.as_posix()}')").fetchdf())
    missing = [column for column in REQUIRED if column not in frame]
    if missing:
        raise SystemExit(f"the panel breaks the input contract; missing columns: {missing}")
    frame = frame.dropna(subset=["forecast_t2m_c", "era5_t2m_c", "observed_t2m_c"])
    frame["valid_day"] = pd.to_datetime(frame.valid_time).dt.date
    return frame


def to_cells(panel: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Deduplicate records to one physical-site series, then assign cells.

    A site can be published by several networks. Merely attaching the same
    ``site_id`` to every member would still let that site enter a cell average
    several times. Members are therefore ordered by the frozen network
    priority and exactly one available member is retained per site, model and
    valid time. Lower-priority members act only as deterministic fallbacks when
    the preferred record is absent.
    """
    policy, weighting = cfg["station_deduplication"], cfg["spatial_weighting"]
    sites = panel.drop_duplicates("station_id")[["station_id", "latitude", "longitude"]].copy()
    if "altitude_m" in panel:
        sites = sites.merge(panel.drop_duplicates("station_id")[["station_id", "altitude_m"]], on="station_id")
    if "network" in panel:
        sites = sites.merge(panel.drop_duplicates("station_id")[["station_id", "network"]], on="station_id")
    sys.path.insert(0, str(ROOT / "scripts"))
    from importlib import import_module
    build_sites = import_module("41_deduplicate_stations").build_sites
    collapsed, kept_apart = build_sites(sites, policy)
    membership = collapsed[["site_id", "members"]].copy()
    membership["station_id"] = membership.members.str.split("|")
    membership = membership.explode("station_id")
    membership["member_priority"] = membership.groupby("site_id").cumcount()
    membership = membership.drop(columns="members")
    panel = panel.merge(membership, on="station_id", validate="many_to_one")
    rows_before_collapse = len(panel)
    panel = (panel.sort_values(["site_id", "model", "valid_time", "member_priority", "station_id"])
             .drop_duplicates(["site_id", "model", "valid_time"], keep="first")
             .drop(columns="member_priority"))
    rows_after_site_collapse = len(panel)

    # Every model must be scored on the exact same site-time cases. Otherwise a
    # model can improve simply because it is missing a difficult station or day.
    n_models = panel.model.nunique()
    common = (panel.groupby(["site_id", "valid_time"], as_index=False).model.nunique()
              .query("model == @n_models")[["site_id", "valid_time"]])
    panel = panel.merge(common, on=["site_id", "valid_time"], validate="many_to_one")
    if panel.empty:
        raise SystemExit("no exact site-time intersection remains across all models")

    grid = EqualAreaGrid(float(weighting["target_cell_km"]))
    centres = collapsed[["site_id", "mean_latitude", "mean_longitude"]].rename(
        columns={"mean_latitude": "latitude", "mean_longitude": "longitude"})
    centres = centres[centres.site_id.isin(panel.site_id.unique())].copy()
    centres["cell_id"] = grid.assign(centres.latitude.to_numpy(), centres.longitude.to_numpy())
    panel = panel.merge(centres[["site_id", "cell_id"]], on="site_id", validate="many_to_one")
    provenance = {"records": int(len(sites)), "sites": int(len(collapsed)),
                  "rows_before_site_collapse": int(rows_before_collapse),
                  "rows_after_site_collapse": int(rows_after_site_collapse),
                  "rows_after_site_collapse_and_common_case_filter": int(len(panel)),
                  "duplicate_member_rows_removed": int(rows_before_collapse - rows_after_site_collapse),
                  "noncommon_model_rows_removed": int(rows_after_site_collapse - len(panel)),
                  "common_site_time_cases": int(len(common)),
                  "pairs_kept_apart_on_elevation": int(len(kept_apart)),
                  "occupied_cells": int(panel.cell_id.nunique()), "grid": grid.summary()}
    return panel, provenance


def cell_day_errors(panel: pd.DataFrame, forecast: str) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Mean absolute error per model, cell and day, against both references.

    Site-day means are formed first so a site with more valid cycles cannot
    receive more weight than another site in the same cell.
    """
    frame = panel.assign(era5_error=(panel[forecast] - panel.era5_t2m_c).abs(),
                         observed_error=(panel[forecast] - panel.observed_t2m_c).abs())
    site_day = frame.groupby(["model", "cell_id", "site_id", "valid_day"], as_index=False)[
        ["era5_error", "observed_error"]].mean()
    grouped = site_day.groupby(["model", "cell_id", "valid_day"], as_index=False)[
        ["era5_error", "observed_error"]].mean()
    models = np.sort(grouped.model.unique())
    cells = np.sort(grouped.cell_id.unique())
    days = np.sort(grouped.valid_day.unique())
    cubes = {}
    for reference in ("era5_error", "observed_error"):
        stack = []
        for model in models:
            wide = grouped[grouped.model == model].pivot(index="cell_id", columns="valid_day", values=reference)
            stack.append(wide.reindex(index=cells, columns=days).to_numpy())
        cubes[reference] = np.stack(stack)  # (model, cell, day)
    return cubes, cells, days


def advantage(cubes: dict[str, np.ndarray], cell_index: np.ndarray, day_index: np.ndarray) -> np.ndarray:
    """B_m = E_grid,m − E_station,m, averaged over days inside a cell then over cells."""
    values = []
    for reference in ("era5_error", "observed_error"):
        selected = cubes[reference][:, cell_index][:, :, day_index]
        by_cell = np.nanmean(selected, axis=2)
        values.append(np.nanmean(by_cell, axis=1))
    return values[0] - values[1]


def roughness_cube(frame: pd.DataFrame, models: np.ndarray, cells: np.ndarray,
                   days: np.ndarray, column: str) -> np.ndarray:
    """Model × cell × day roughness aligned to the error cubes."""
    required = ["model", "cell_id", "valid_day", column]
    missing = [name for name in required if name not in frame]
    if missing:
        raise SystemExit(f"the confirmatory roughness table lacks columns: {missing}")
    frame = frame.copy()
    frame["valid_day"] = pd.to_datetime(frame.valid_day).dt.date
    duplicates = frame.duplicated(["model", "cell_id", "valid_day"]).sum()
    if duplicates:
        raise SystemExit(f"the roughness table has {duplicates} duplicate model-cell-day rows")
    stack = []
    for model in models:
        wide = frame[frame.model == model].pivot(index="cell_id", columns="valid_day", values=column)
        stack.append(wide.reindex(index=cells, columns=days).to_numpy(float))
    cube = np.stack(stack)
    if any(model not in set(frame.model) for model in models):
        missing_models = [model for model in models if model not in set(frame.model)]
        raise SystemExit(f"the roughness table lacks models: {missing_models}")
    return cube


def aggregate_roughness(cube: np.ndarray, cell_index: np.ndarray, day_index: np.ndarray) -> np.ndarray:
    """Aggregate roughness with the same equal-cell weighting as the errors."""
    selected = cube[:, cell_index][:, :, day_index]
    return np.nanmean(np.nanmean(selected, axis=2), axis=1)


def slope(predictor: np.ndarray, response: np.ndarray) -> float:
    """Least-squares slope. Two points determine one, fewer do not."""
    usable = np.isfinite(predictor) & np.isfinite(response)
    if usable.sum() < 2 or np.ptp(predictor[usable]) == 0:
        return float("nan")
    design = np.column_stack([np.ones(usable.sum()), predictor[usable]])
    return float(np.linalg.lstsq(design, response[usable], rcond=None)[0][1])


def block_days(rng: np.random.Generator, days: int, width: int) -> np.ndarray:
    starts = rng.integers(0, days, size=int(np.ceil(days / width)))
    return ((starts[:, None] + np.arange(width)) % days).ravel()[:days]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--roughness", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    seal = verify_seal()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    scale = float(cfg["field_smoothness"]["primary_scale_km"])
    args.output.mkdir(parents=True, exist_ok=True)

    if args.self_test:
        args.panel, args.roughness = build_pilot_panel(cfg, args.output)
    if args.panel is None or args.roughness is None:
        raise SystemExit("--panel and --roughness are required unless --self-test is given")

    panel = load_panel(args.panel)
    checks = [*check_coordinates(panel.drop_duplicates("station_id")),
              *check_elevation(panel.drop_duplicates("station_id")),
              *check_temperature_units(panel.observed_t2m_c)]
    validation = summarise(checks)
    if not validation["passed"]:
        for item in checks:
            print(item)
        raise SystemExit("the panel fails archive validation; refusing to run")

    panel, provenance = to_cells(panel, cfg)
    cubes, cells, days = cell_day_errors(panel, "forecast_t2m_c")
    models = np.sort(panel.model.unique())
    roughness_frame = (pd.read_csv(args.roughness) if args.roughness.suffix == ".csv"
                       else duckdb.sql(
                           f"SELECT * FROM read_parquet('{args.roughness.as_posix()}')"
                       ).fetchdf())
    roughness_column = f"roughness_{scale:g}km_k"
    dynamic_roughness = {"cell_id", "valid_day"}.issubset(roughness_frame.columns)
    if args.self_test and not dynamic_roughness:
        roughness = roughness_frame.set_index("model")[roughness_column]
        predictor = roughness.reindex(models).to_numpy(float)
        roughness_values = None
        if not np.isfinite(predictor).all():
            raise SystemExit(f"the roughness table lacks models: {list(models[~np.isfinite(predictor)])}")
    else:
        if not dynamic_roughness:
            raise SystemExit("a held-out run requires model-cell-day roughness so every bootstrap draw "
                             "recalculates both sides of the regression")
        roughness_values = roughness_cube(roughness_frame, models, cells, days, roughness_column)
        # All models and both references use one shared support. This is stricter
        # than taking separate nanmeans and prevents model-specific missingness
        # from changing either axis of the regression.
        common_support = (np.all(np.isfinite(roughness_values), axis=0)
                          & np.all(np.isfinite(cubes["era5_error"]), axis=0)
                          & np.all(np.isfinite(cubes["observed_error"]), axis=0))
        if not common_support.any():
            raise SystemExit("no common model-cell-day support remains after aligning roughness and errors")
        for values in (*cubes.values(), roughness_values):
            values[:, ~common_support] = np.nan
        predictor = aggregate_roughness(roughness_values, np.arange(len(cells)), np.arange(len(days)))

    point = advantage(cubes, np.arange(len(cells)), np.arange(len(days)))
    beta = slope(predictor, point)
    rank = float(pd.Series(predictor).corr(pd.Series(point), method="spearman"))

    bootstrap = cfg["bootstrap"]
    draws = {}
    for width in cfg["controlled_smoothing"]["bootstrap_block_days"]:
        rng = np.random.default_rng(bootstrap["seed"] + width)
        estimates = []
        for _ in range(bootstrap["n_resamples"]):
            cell_index = rng.integers(0, len(cells), size=len(cells))
            day_index = block_days(rng, len(days), width)
            sampled_predictor = (predictor if roughness_values is None else
                                 aggregate_roughness(roughness_values, cell_index, day_index))
            estimates.append(slope(sampled_predictor, advantage(cubes, cell_index, day_index)))
        draws[width] = np.array(estimates, dtype=float)

    tail = (1 - bootstrap["ci"]) / 2
    intervals = {int(width): {"mean": float(np.nanmean(values)),
                              "ci_low": float(np.nanquantile(values, tail)),
                              "ci_high": float(np.nanquantile(values, 1 - tail)),
                              "p_slope_gt_zero": float(np.nanmean(values > 0))}
                 for width, values in draws.items()}

    # Dropping one model has to leave a fit worth reading. With three models it
    # leaves two, which determine a line exactly and a rank correlation of ±1
    # whatever the data say, so the check is reported as degenerate rather than
    # as a failure. A silent NaN here would read as instability that was never
    # measured.
    informative = len(models) >= 4
    dropped = []
    for position, model in enumerate(models):
        keep = np.arange(len(models)) != position
        dropped.append({"excluded_model": model, "n_models_remaining": int(keep.sum()),
                        "slope": slope(predictor[keep], point[keep]),
                        "spearman": (float(pd.Series(predictor[keep]).corr(pd.Series(point[keep]), method="spearman"))
                                     if informative else float("nan"))})
    leave_one_out = pd.DataFrame(dropped)
    signs = np.sign(leave_one_out.slope.to_numpy())
    stability = (bool(np.isfinite(signs).all() and signs.min() == signs.max()) if informative else None)

    per_model = pd.DataFrame({"model": models, "roughness_k": predictor, "advantage_c": point})
    per_model.to_csv(args.output / "model_level_advantage.csv", index=False)
    leave_one_out.to_csv(args.output / "leave_one_model_out.csv", index=False)
    plot(per_model, beta, args.output / "confirmatory_primary")

    widest = max(intervals)
    slope_criterion_met = bool(beta > 0 and intervals[widest]["ci_low"] > 0)
    leave_one_out_criterion_met = stability is True
    primary_criterion_met = bool(slope_criterion_met and leave_one_out_criterion_met)
    verdict = {"analysis": "confirmatory primary regression", "seal": seal,
               # A run on the pilot exercises the code. It cannot confirm
               # anything, because the pilot is the dataset the hypothesis was
               # developed on; confirmation only means something on the held-out
               # global set. The stamp is here so that a "confirmed" printed
               # during a dry run cannot be quoted as a result.
               "role": "pipeline dry run on the hypothesis-development dataset; not a confirmation"
                       if args.self_test else "confirmatory run on the held-out dataset",
               "hypothesis": "B_m = E_grid - E_station rises with roughness; confirmation is a positive slope",
               "primary_scale_km": scale, "models": list(models), "n_cells": int(len(cells)), "n_days": int(len(days)),
               "provenance": {**provenance,
                              "roughness_input": ("fixed model summary for development-set dry run"
                                                   if roughness_values is None else
                                                   "model-cell-day values jointly resampled with errors")},
               "validation": validation["counts"],
               "slope": beta, "spearman": rank, "bootstrap_by_block_width": intervals,
               "leave_one_model_out": leave_one_out.to_dict(orient="records"),
               "leave_one_model_out_informative": informative,
               "leave_one_model_out_note": ("each refit keeps at least three models" if informative else
                                            "fewer than four models: each refit keeps two, which determine a line "
                                            "exactly, so the check cannot distinguish stability from arithmetic"),
               "slope_sign_stable_under_leave_one_out": stability,
               "primary_slope_criterion_met": slope_criterion_met,
               "primary_leave_one_model_out_criterion_met": leave_one_out_criterion_met,
               "primary_criterion_met": primary_criterion_met,
               # This script evaluates only the primary regression. The full
               # preregistered confirmation also requires the three scales,
               # MAE and RMSE, and S1/S2 outside the development region.
               "full_confirmation": None,
               "confirmed": None,
               "note": ("dry run only: criterion flags diagnose the pipeline and are not scientific evidence"
                        if args.self_test else
                        "primary_criterion_met covers the slope and informative leave-one-model-out checks only; "
                        "full confirmation must be assembled from every preregistered criterion")}
    (args.output / "confirmatory_summary.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    print(per_model.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(leave_one_out.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(json.dumps({key: verdict[key] for key in ("slope", "spearman", "bootstrap_by_block_width",
                                                    "leave_one_model_out_informative",
                                                    "slope_sign_stable_under_leave_one_out",
                                                    "primary_slope_criterion_met", "primary_criterion_met",
                                                    "full_confirmation")},
                     indent=2, default=str))


def plot(per_model: pd.DataFrame, beta: float, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 5.0), layout="constrained")
    axis.scatter(per_model.roughness_k, per_model.advantage_c, s=60, color="#0072B2",
                 edgecolors="black", linewidths=0.3)
    for _, row in per_model.iterrows():
        axis.annotate(row.model.replace("_hres_init", "").replace("_", " "), (row.roughness_k, row.advantage_c),
                      fontsize=8, xytext=(5, 4), textcoords="offset points")
    if np.isfinite(beta):
        span = np.linspace(per_model.roughness_k.min(), per_model.roughness_k.max(), 10)
        centre = per_model.advantage_c.mean() + beta * (span - per_model.roughness_k.mean())
        axis.plot(span, centre, color="#D55E00", linewidth=1.2, label=f"pendiente = {beta:.3f}")
        axis.legend(fontsize=9, frameon=False)
    axis.set(xlabel="Rugosidad del campo (K)",
             ylabel="Ventaja que concede la referencia de rejilla (°C)\nE_rejilla − E_estación",
             title="Análisis confirmatorio primario")
    axis.grid(color="#dddddd", linewidth=0.5)
    figure.savefig(output.with_suffix(".png"), dpi=300, facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)


def build_pilot_panel(cfg: dict, output: Path) -> tuple[Path, Path]:
    """Assemble the pilot's own data in the input contract, to exercise the run."""
    paths = cfg["paths"]
    lead = cfg["controlled_smoothing"]["lead_hours"]
    source = ROOT / "data" / "interim" / "controlled_smoothing" / f"blurred_stations_lead{lead:03d}.parquet"
    if not source.exists():
        raise SystemExit("the self test needs scripts/34_analyse_controlled_smoothing.py to have run")
    stations = (ROOT / paths["spatial_stations_csv"]).as_posix()
    panel = duckdb.sql(
        f"""SELECT b.station_id, s.latitude, s.longitude, s.altitude_m, b.valid_time, b.model,
                   b."sigma_0km" AS forecast_t2m_c, b.era5_t2m_c, b.avamet_t2m_qc_c AS observed_t2m_c
              FROM read_parquet('{source.as_posix()}') AS b
              INNER JOIN read_csv_auto('{stations}') AS s USING (station_id)"""
    ).fetchdf()
    panel_path = output / "self_test_panel.parquet"
    connection = duckdb.connect()
    connection.register("panel", panel)
    connection.execute(f"COPY panel TO '{panel_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    roughness_path = output / "self_test_roughness.csv"
    pd.read_csv(ROOT / paths["field_smoothness_results_directory"] / "field_smoothness_summary.csv") \
        .rename(columns={"source": "model"}).to_csv(roughness_path, index=False)
    return panel_path, roughness_path


if __name__ == "__main__":
    main()
