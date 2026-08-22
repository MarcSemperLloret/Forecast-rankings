#!/usr/bin/env python3
"""Analyse the separate 2021–2022 IFS/Pangu temporal sensitivity cohort."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "regional_t2m_2020.yaml"
MODELS = ("ifs_hres", "pangu_hres_init")
REFERENCES = {"era5_station": "era5_t2m_c", "avamet": "avamet_t2m_qc_c"}


def interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    tail = (1 - confidence) / 2
    return tuple(np.quantile(values, [tail, 1 - tail]).tolist())


def block_indices(rng: np.random.Generator, days: int, draws: int, width: int) -> np.ndarray:
    starts = rng.integers(0, days, size=(draws, int(np.ceil(days / width))))
    return ((starts[:, :, None] + np.arange(width)) % days).reshape(draws, -1)[:, :days]


def daily_errors(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    values = {"valid_day": pd.to_datetime(frame.valid_time).dt.date}
    for model in MODELS:
        values[model] = (frame[f"{model}_t2m_c"] - frame[target]).abs()
    return pd.DataFrame(values).groupby("valid_day", as_index=False).mean(numeric_only=True).sort_values("valid_day").reset_index(drop=True)


def analyse(frame: pd.DataFrame, label: str, cfg: dict) -> tuple[list[dict], list[dict]]:
    required = [*REFERENCES.values(), *(f"{model}_t2m_c" for model in MODELS)]
    common = frame.dropna(subset=required)
    if common.empty:
        raise RuntimeError(f"no complete cases for {label}")
    daily = {reference: daily_errors(common, target) for reference, target in REFERENCES.items()}
    if not daily["era5_station"].valid_day.equals(daily["avamet"].valid_day):
        raise RuntimeError("reference day series differ")
    bootstrap, temporal = cfg["bootstrap"], cfg["temporal_replication"]
    scores, robustness = [], []
    for width in temporal["bootstrap_block_days"]:
        rng = np.random.default_rng(bootstrap["seed"] + width * 100 + sum(map(ord, label)))
        indices = block_indices(rng, len(daily["avamet"]), bootstrap["n_resamples"], width)
        draws, winners = {}, {}
        for reference, values in daily.items():
            point = values[list(MODELS)].mean().to_numpy()
            sampled = values[list(MODELS)].to_numpy()[indices].mean(axis=1)
            draws[reference], winners[reference] = sampled, np.argmin(sampled, axis=1)
            for position, model in enumerate(MODELS):
                low, high = interval(sampled[:, position], bootstrap["ci"])
                scores.append({"temporal_panel": label, "block_days": width, "reference": reference, "model": model, "n_common_cases": len(common), "n_stations": common.station_id.nunique(), "n_daily_blocks": len(values), "mae_c_day_weighted": point[position], "mae_ci_low_c": low, "mae_ci_high_c": high, "winner_probability": float((winners[reference] == position).mean()), "mae_rank": int(pd.Series(point).rank(method="min").iloc[position])})
        era = draws["era5_station"][:, 0] - draws["era5_station"][:, 1]
        ava = draws["avamet"][:, 0] - draws["avamet"][:, 1]
        switch = ava - era
        low, high = interval(switch, bootstrap["ci"])
        robustness.append({"temporal_panel": label, "block_days": width, "n_common_cases": len(common), "n_stations": common.station_id.nunique(), "n_daily_blocks": len(daily["avamet"]), "avamet_winner": MODELS[int(np.argmin(daily["avamet"][list(MODELS)].mean().to_numpy()))], "era5_winner": MODELS[int(np.argmin(daily["era5_station"][list(MODELS)].mean().to_numpy()))], "p_winners_differ": float((winners["avamet"] != winners["era5_station"]).mean()), "ifs_minus_pangu_era5_delta_mae_c": era.mean(), "ifs_minus_pangu_avamet_delta_mae_c": ava.mean(), "avamet_minus_era5_delta_mae_c": switch.mean(), "switch_ci_low_c": low, "switch_ci_high_c": high, "p_switch_gt_zero": float((switch > 0).mean()), "p_pairwise_order_reverses": float((era * ava < 0).mean())})
    return scores, robustness


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    temporal, paths = cfg["temporal_replication"], cfg["paths"]
    source = ROOT / paths["temporal_directory"] / "year=*" / "lead=024" / "batches" / "*.parquet"
    columns = ["station_id", "valid_time", "ifs_hres_t2m_c", "pangu_hres_init_t2m_c", "era5_t2m_c", "avamet_t2m_qc_c"]
    selection = ", ".join(columns)
    frame = duckdb.sql(f"SELECT {selection}, analysis_year FROM read_parquet('{source.as_posix()}', hive_partitioning=true)").fetchdf()
    expected = set(temporal["years"])
    baseline = ROOT / temporal["baseline_panel"]
    if baseline.exists():
        # The frozen 2020 panel scored with the same two models, so that the
        # years share one axis.  Its own three-model cohort is untouched.
        earlier = duckdb.sql(
            f"SELECT {selection}, {temporal['baseline_year']} AS analysis_year FROM read_parquet('{baseline.as_posix()}')"
        ).fetchdf()
        frame = pd.concat([earlier, frame], ignore_index=True)
        expected.add(temporal["baseline_year"])
    found = set(frame.analysis_year.unique())
    if found != expected:
        raise RuntimeError(f"expected years {expected}, found {found}")
    score_rows, robust_rows = [], []
    for year, subset in frame.groupby("analysis_year", sort=True):
        score, robust = analyse(subset, str(year), cfg); score_rows.extend(score); robust_rows.extend(robust)
    extracted = frame[frame.analysis_year.isin(temporal["years"])]
    label = "pooled_" + "_".join(str(year) for year in sorted(temporal["years"]))
    score, robust = analyse(extracted, label, cfg); score_rows.extend(score); robust_rows.extend(robust)
    if len(found) > len(temporal["years"]):
        span = f"pooled_{min(found)}_{max(found)}"
        score, robust = analyse(frame, span, cfg); score_rows.extend(score); robust_rows.extend(robust)
    output = ROOT / paths["temporal_results_directory"]
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(score_rows).sort_values(["temporal_panel", "block_days", "reference", "mae_rank"]).to_csv(output / "temporal_ifs_pangu_scores.csv", index=False)
    summary = pd.DataFrame(robust_rows).sort_values(["temporal_panel", "block_days"])
    summary.to_csv(output / "temporal_ifs_pangu_robustness.csv", index=False)
    (output / "temporal_ifs_pangu_summary.json").write_text(json.dumps({"pilot_name": cfg["pilot_name"], "analysis": "two-model IFS/Pangu temporal sensitivity; not a GraphCast substitute", "years": [int(year) for year in sorted(found)], "extracted_years": temporal["years"], "baseline_year_from_frozen_panel": temporal["baseline_year"], "lead_hours": temporal["leads_hours"], "models": list(MODELS), "bootstrap": {"method": "circular moving-block bootstrap on ordered valid days", "n_resamples": cfg["bootstrap"]["n_resamples"], "block_days": temporal["bootstrap_block_days"], "ci": cfg["bootstrap"]["ci"], "seed": cfg["bootstrap"]["seed"]}, "status": "completed"}, indent=2), encoding="utf-8")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
