#!/usr/bin/env python3
"""Analyse point-to-grid representativeness and the interpolation sensitivity.

Three pre-specified questions share one complete-case panel so that every
comparison runs on the same valid days:

* whether the reference effect survives when the observations are aggregated to
  the ERA5 cell, which turns a grid-to-point comparison into a grid-to-area one;
* whether it depends on how many stations a cell contains;
* whether it depends on extracting the grid bilinearly or at the nearest point.

An extraction method applies to the whole evaluation: a nearest-point reference
is compared against nearest-point forecasts, never against bilinear ones.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "regional_t2m_2020.yaml"
MODELS = ("ifs_hres", "graphcast_hres_init", "pangu_hres_init")
EXTRACTIONS = {"bilinear": ("{model}_t2m_c", "era5_t2m_c"), "nearest": ("{model}_nearest_t2m_c", "era5_nearest_t2m_c")}


def interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    tail = (1 - confidence) / 2
    return tuple(np.quantile(values, [tail, 1 - tail]).tolist())


def block_indices(rng: np.random.Generator, days: int, draws: int, width: int) -> np.ndarray:
    starts = rng.integers(0, days, size=(draws, int(np.ceil(days / width))))
    return ((starts[:, :, None] + np.arange(width)) % days).reshape(draws, -1)[:, :days]


def daily_errors(frame: pd.DataFrame, model_template: str, reference: str) -> pd.DataFrame:
    values = {"valid_day": pd.to_datetime(frame.valid_time).dt.date}
    for model in MODELS:
        values[model] = (frame[model_template.format(model=model)] - frame[reference]).abs()
    return pd.DataFrame(values).groupby("valid_day", as_index=False).mean(numeric_only=True).sort_values("valid_day").reset_index(drop=True)


def cell_panel(frame: pd.DataFrame, minimum_stations: int) -> pd.DataFrame:
    """Collapse stations onto their ERA5 cell at each valid time.

    The forecast and reanalysis values are constant inside a cell because they
    are read at its grid point, so only the observations are averaged.
    """
    grid_columns = [column for column in frame.columns if column.endswith("_nearest_t2m_c")]
    aggregations = {column: (column, "first") for column in grid_columns}
    aggregations["avamet_cell_mean_t2m_c"] = ("avamet_t2m_qc_c", "mean")
    aggregations["stations_reporting"] = ("avamet_t2m_qc_c", "size")
    cells = frame.groupby(["valid_time", "cell_id", "density_stratum"], as_index=False).agg(**aggregations)
    for column in grid_columns:
        spread = frame.groupby(["valid_time", "cell_id"])[column].nunique()
        if int(spread.max()) != 1:
            raise RuntimeError(f"{column} is not constant within a cell; the grid point is not shared")
    return cells[cells.stations_reporting >= minimum_stations].reset_index(drop=True)


def analyse(daily: dict[str, pd.DataFrame], unit: str, extraction: str, stratum: str, cases: int,
            units: int, cfg: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """Bootstrap one panel over ordered valid days and contrast its two references."""
    bootstrap, extension = cfg["bootstrap"], cfg["representativeness_extension"]
    primary_a, primary_b = extension["primary_pair"]
    days = daily["avamet"].valid_day
    for reference, values in daily.items():
        if not values.valid_day.equals(days):
            raise RuntimeError(f"reference {reference} does not share the valid-day blocks")
    scores, effects, summaries = [], [], []
    for width in extension["bootstrap_block_days"]:
        rng = np.random.default_rng(bootstrap["seed"] + width * 100 + sum(map(ord, f"{unit}{extraction}{stratum}")))
        indices = block_indices(rng, len(days), bootstrap["n_resamples"], width)
        sampled, winners = {}, {}
        for reference, values in daily.items():
            point = values[list(MODELS)].mean().to_numpy()
            draws = values[list(MODELS)].to_numpy()[indices].mean(axis=1)
            sampled[reference], winners[reference] = draws, np.argmin(draws, axis=1)
            for position, model in enumerate(MODELS):
                low, high = interval(draws[:, position], bootstrap["ci"])
                scores.append({"unit": unit, "extraction": extraction, "stratum": stratum, "block_days": width, "reference": reference, "model": model,
                               "n_common_cases": cases, "n_units": units, "n_daily_blocks": len(days), "mae_c_day_weighted": point[position],
                               "mae_ci_low_c": low, "mae_ci_high_c": high, "winner_probability": float((winners[reference] == position).mean()),
                               "mae_rank": int(pd.Series(point).rank(method="min").iloc[position])})
        pairs = {}
        for first, second in itertools.combinations(range(len(MODELS)), 2):
            era = sampled["era5"][:, first] - sampled["era5"][:, second]
            ava = sampled["avamet"][:, first] - sampled["avamet"][:, second]
            switch = ava - era
            low, high = interval(switch, bootstrap["ci"])
            row = {"unit": unit, "extraction": extraction, "stratum": stratum, "block_days": width, "model_a": MODELS[first], "model_b": MODELS[second],
                   "era5_delta_mae_a_minus_b_c": daily["era5"][MODELS[first]].mean() - daily["era5"][MODELS[second]].mean(),
                   "avamet_delta_mae_a_minus_b_c": daily["avamet"][MODELS[first]].mean() - daily["avamet"][MODELS[second]].mean(),
                   "avamet_minus_era5_delta_mae_c": switch.mean(), "switch_ci_low_c": low, "switch_ci_high_c": high,
                   "p_switch_gt_zero": float((switch > 0).mean()), "p_pairwise_order_reverses": float((era * ava < 0).mean())}
            pairs[(MODELS[first], MODELS[second])] = row
            effects.append(row)
        primary = pairs[(primary_a, primary_b)]
        summaries.append({"unit": unit, "extraction": extraction, "stratum": stratum, "block_days": width, "n_common_cases": cases, "n_units": units,
                          "n_daily_blocks": len(days), "avamet_winner": MODELS[int(np.argmin(daily["avamet"][list(MODELS)].mean().to_numpy()))],
                          "era5_winner": MODELS[int(np.argmin(daily["era5"][list(MODELS)].mean().to_numpy()))],
                          "p_winners_differ": float((winners["avamet"] != winners["era5"]).mean()),
                          "primary_pair": f"{primary_a}__minus__{primary_b}", "primary_effect_c": primary["avamet_minus_era5_delta_mae_c"],
                          "primary_effect_ci_low_c": primary["switch_ci_low_c"], "primary_effect_ci_high_c": primary["switch_ci_high_c"],
                          "primary_pair_reversal_probability": primary["p_pairwise_order_reverses"]})
    return scores, effects, summaries


def station_panels(frame: pd.DataFrame, cfg: dict) -> list[tuple[str, str, str, pd.DataFrame, int, int]]:
    strata = [("all_eligible", frame)] + [(str(value), subset) for value, subset in frame.groupby("density_stratum", sort=True)]
    panels = []
    for extraction, (model_template, era5_column) in EXTRACTIONS.items():
        for stratum, subset in strata:
            daily = {"era5": daily_errors(subset, model_template, era5_column), "avamet": daily_errors(subset, model_template, "avamet_t2m_qc_c")}
            panels.append(("station", extraction, stratum, daily, len(subset), subset.station_id.nunique()))
    return panels


def cell_panels(frame: pd.DataFrame, cfg: dict) -> list[tuple[str, str, str, pd.DataFrame, int, int]]:
    extension = cfg["representativeness_extension"]
    cells = cell_panel(frame, extension["min_cell_stations_for_aggregate"])
    model_template, era5_column = EXTRACTIONS["nearest"]
    strata = [("multi_station_cells", cells)] + [(str(value), subset) for value, subset in cells.groupby("density_stratum", sort=True)]
    panels = []
    for stratum, subset in strata:
        if subset.cell_id.nunique() < 2:
            continue
        daily = {"era5": daily_errors(subset, model_template, era5_column), "avamet": daily_errors(subset, model_template, "avamet_cell_mean_t2m_c")}
        panels.append(("cell_aggregate", "nearest", stratum, daily, len(subset), subset.cell_id.nunique()))
    return panels


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    extension, paths = cfg["representativeness_extension"], cfg["paths"]
    output = ROOT / paths["representativeness_results_directory"]
    output.mkdir(parents=True, exist_ok=True)
    scores, effects, summaries = [], [], []
    for lead in extension["leads_hours"]:
        bilinear_source = (ROOT / paths["spatial_directory"] / f"lead={lead:03d}" / "batches" / "*.parquet").as_posix()
        nearest_source = (ROOT / paths["representativeness_directory"] / f"lead={lead:03d}" / "nearest.parquet").as_posix()
        if not Path(nearest_source).exists():
            print(f"skipping lead={lead}: nearest fields are not materialised yet", flush=True)
            continue
        frame = duckdb.sql(
            f"""SELECT b.*, n.* EXCLUDE (station_id, init_time, valid_time, lead_h)
                  FROM read_parquet('{bilinear_source}') AS b
                  INNER JOIN read_parquet('{nearest_source}') AS n
                    USING (station_id, init_time, valid_time, lead_h)"""
        ).fetchdf()
        required = ["avamet_t2m_qc_c", "era5_t2m_c", "era5_nearest_t2m_c", *(template.format(model=model) for template, _ in EXTRACTIONS.values() for model in MODELS)]
        frame = frame.dropna(subset=required).reset_index(drop=True)
        if frame.empty:
            raise RuntimeError(f"no complete cases at lead={lead}")
        for unit, extract, stratum, daily, cases, units in [*station_panels(frame, cfg), *cell_panels(frame, cfg)]:
            score, effect, summary = analyse(daily, unit, extract, stratum, cases, units, cfg)
            for rows, collector in ((score, scores), (effect, effects), (summary, summaries)):
                for row in rows:
                    row["lead_h"] = lead
                collector.extend(rows)
        print(f"analysed representativeness lead={lead}; cases={len(frame)}; stations={frame.station_id.nunique()}; cells={frame.cell_id.nunique()}", flush=True)
    if not summaries:
        raise RuntimeError("no lead produced a representativeness panel")
    order = ["lead_h", "unit", "extraction", "stratum", "block_days"]
    pd.DataFrame(scores).sort_values([*order, "reference", "mae_rank"]).to_csv(output / "representativeness_scores.csv", index=False)
    pd.DataFrame(effects).sort_values([*order, "model_a", "model_b"]).to_csv(output / "representativeness_reference_effects.csv", index=False)
    robust = pd.DataFrame(summaries).sort_values(order)
    robust.to_csv(output / "representativeness_robustness.csv", index=False)
    (output / "representativeness_summary.json").write_text(json.dumps(
        {"pilot_name": cfg["pilot_name"], "analysis": "point-to-grid representativeness and interpolation sensitivity",
         "leads_hours": extension["leads_hours"], "grid_resolution_deg": extension["grid_resolution_deg"],
         "min_cell_stations_for_aggregate": extension["min_cell_stations_for_aggregate"],
         "units": ["station", "cell_aggregate"], "extractions": list(EXTRACTIONS),
         "bootstrap": {"method": "circular moving-block bootstrap on ordered valid days", "n_resamples": cfg["bootstrap"]["n_resamples"],
                       "block_days": extension["bootstrap_block_days"], "ci": cfg["bootstrap"]["ci"], "seed": cfg["bootstrap"]["seed"]},
         "status": "completed"}, indent=2), encoding="utf-8")
    primary = robust[robust.block_days == extension["bootstrap_block_days"][0]]
    print(primary.drop(columns=["primary_pair", "block_days"]).to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
