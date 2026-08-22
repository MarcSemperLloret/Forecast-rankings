#!/usr/bin/env python3
"""Score one exploratory surface-variable cohort with temporal block bootstrap."""
from __future__ import annotations

import argparse
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


def interval(values: np.ndarray, ci: float) -> tuple[float, float]:
    alpha = (1 - ci) / 2
    return tuple(np.quantile(values, [alpha, 1 - alpha]).tolist())


def block_indices(rng: np.random.Generator, n_days: int, n_resamples: int, block_days: int) -> np.ndarray:
    blocks = int(np.ceil(n_days / block_days))
    starts = rng.integers(0, n_days, size=(n_resamples, blocks))
    return ((starts[:, :, None] + np.arange(block_days)) % n_days).reshape(n_resamples, -1)[:, :n_days]


def daily_errors(frame: pd.DataFrame, target: str, suffix: str) -> pd.DataFrame:
    values = {"valid_day": pd.to_datetime(frame.valid_time).dt.date}
    for model in MODELS:
        values[model] = (frame[f"{model}_{suffix}"] - frame[target]).abs()
    return pd.DataFrame(values).groupby("valid_day", as_index=False).mean(numeric_only=True).sort_values("valid_day").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variable", help="key from surface_variable_extension.variables")
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    variables = cfg["surface_variable_extension"]["variables"]
    if args.variable not in variables:
        raise ValueError(f"unknown variable {args.variable!r}; choose {sorted(variables)}")
    variable = variables[args.variable]
    suffix = variable["output_name"]
    input_path = ROOT / cfg["paths"]["surface_variable_directory"] / f"variable={args.variable}" / "annual.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"variable={args.variable} is not consolidated: {input_path}")
    frame = duckdb.sql(f"SELECT * FROM read_parquet('{input_path.as_posix()}')").fetchdf()
    targets = {"era5_station": f"era5_{suffix}", "avamet": f"avamet_{suffix}"}
    required = [*targets.values(), *(f"{model}_{suffix}" for model in MODELS)]
    common = frame.dropna(subset=required).copy()
    if common.empty:
        raise RuntimeError(f"variable={args.variable} has no complete cases")
    daily = {reference: daily_errors(common, target, suffix) for reference, target in targets.items()}
    if not daily["era5_station"].valid_day.equals(daily["avamet"].valid_day):
        raise RuntimeError("references do not share daily blocks")
    bootstrap = cfg["bootstrap"]
    blocks = cfg["lead_time_extension"]["bootstrap_block_days"]
    primary_a, primary_b = cfg["lead_time_extension"]["primary_pair"]
    score_rows, effect_rows, summary_rows = [], [], []
    for block_days in blocks:
        rng = np.random.default_rng(bootstrap["seed"] + 10_000 + block_days + sum(map(ord, args.variable)))
        indices = block_indices(rng, len(daily["era5_station"]), bootstrap["n_resamples"], block_days)
        draws, winners = {}, {}
        for reference in targets:
            values = daily[reference][list(MODELS)].to_numpy()
            point, sampled = values.mean(axis=0), values[indices].mean(axis=1)
            draws[reference], winners[reference] = sampled, np.argmin(sampled, axis=1)
            for index, model in enumerate(MODELS):
                low, high = interval(sampled[:, index], bootstrap["ci"])
                score_rows.append({"variable": args.variable, "unit": variable["unit"], "block_days": block_days, "reference": reference, "model": model,
                                   "n_common_cases": len(common), "n_daily_blocks": len(values), "mae": point[index], "mae_ci_low": low, "mae_ci_high": high,
                                   "winner_probability": float((winners[reference] == index).mean()), "mae_rank": int(pd.Series(point).rank(method="min").iloc[index])})
        pair_rows = {}
        for a, b in itertools.combinations(range(len(MODELS)), 2):
            era5_delta = draws["era5_station"][:, a] - draws["era5_station"][:, b]
            avamet_delta = draws["avamet"][:, a] - draws["avamet"][:, b]
            switched = avamet_delta - era5_delta
            low, high = interval(switched, bootstrap["ci"])
            row = {"variable": args.variable, "unit": variable["unit"], "block_days": block_days, "model_a": MODELS[a], "model_b": MODELS[b],
                   "avamet_minus_era5_delta_mae": switched.mean(), "switch_ci_low": low, "switch_ci_high": high,
                   "p_pairwise_order_reverses": float((era5_delta * avamet_delta < 0).mean())}
            effect_rows.append(row)
            pair_rows[(MODELS[a], MODELS[b])] = row
        primary = pair_rows[(primary_a, primary_b)]
        summary_rows.append({"variable": args.variable, "unit": variable["unit"], "block_days": block_days, "n_common_cases": len(common),
                             "n_daily_blocks": len(daily["era5_station"]), "avamet_winner": MODELS[int(np.argmin(daily["avamet"][list(MODELS)].mean()))],
                             "era5_winner": MODELS[int(np.argmin(daily["era5_station"][list(MODELS)].mean()))], "p_winners_differ": float((winners["avamet"] != winners["era5_station"]).mean()),
                             "primary_effect": primary["avamet_minus_era5_delta_mae"], "primary_ci_low": primary["switch_ci_low"], "primary_ci_high": primary["switch_ci_high"],
                             "primary_pair_reversal_probability": primary["p_pairwise_order_reverses"]})
    result_dir = ROOT / cfg["paths"]["surface_variable_results_directory"] / f"variable={args.variable}"
    result_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(score_rows).sort_values(["block_days", "reference", "mae_rank"]).to_csv(result_dir / "scores.csv", index=False)
    pd.DataFrame(effect_rows).sort_values(["block_days", "model_a", "model_b"]).to_csv(result_dir / "reference_effects.csv", index=False)
    summary = pd.DataFrame(summary_rows).sort_values("block_days")
    summary.to_csv(result_dir / "robustness.csv", index=False)
    manifest = {"pilot_name": cfg["pilot_name"], "variable": args.variable, "label": variable["label"], "unit": variable["unit"],
                "lead_hours": cfg["surface_variable_extension"]["lead_hours"], "n_resamples": bootstrap["n_resamples"], "block_days": blocks,
                "status": "exploratory variable cohort; interpret only with the declared measurement-semantic limitations"}
    (result_dir / "summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
