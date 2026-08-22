#!/usr/bin/env python3
"""Pre-specified annual regional verification and spatial stability checks."""
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
REFERENCES = {"era5_station": "era5_t2m_c", "avamet": "avamet_t2m_qc_c"}


def percentile_interval(values: np.ndarray, ci: float) -> tuple[float, float]:
    alpha = (1 - ci) / 2
    return tuple(np.quantile(values, [alpha, 1 - alpha]).tolist())


def season(values: pd.Series) -> pd.Series:
    month = pd.to_datetime(values).dt.month
    return month.map({12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"})


def daily_errors(frame: pd.DataFrame, target: str, models: list[str]) -> pd.DataFrame:
    values = {"valid_day": pd.to_datetime(frame.valid_time).dt.date}
    for model in models:
        values[model] = (frame[f"{model}_t2m_c"] - frame[target]).abs()
    return pd.DataFrame(values).groupby("valid_day", as_index=False).mean(numeric_only=True)


def scores(frame: pd.DataFrame, reference: str, target: str, models: list[str]) -> pd.DataFrame:
    daily = daily_errors(frame, target, models)
    point = daily[models].mean()
    case_weighted = {model: (frame[f"{model}_t2m_c"] - frame[target]).abs().mean() for model in models}
    return pd.DataFrame({
        "reference": reference,
        "model": models,
        "n_common_cases": len(frame),
        "n_daily_blocks": len(daily),
        "mae_c_day_weighted": [point[model] for model in models],
        "mae_c_case_weighted": [case_weighted[model] for model in models],
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True, help="key from analysis_cohorts in the regional configuration")
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cohorts = cfg["analysis_cohorts"]
    if args.cohort not in cohorts:
        raise ValueError(f"unknown cohort {args.cohort!r}; choose {sorted(cohorts)}")
    cohort = cohorts[args.cohort]
    models = list(cohort["models"])
    baseline_model = cohort["baseline_model"]
    input_path = ROOT / "data" / "interim" / "analysis_cohorts" / f"{args.cohort}.parquet"
    results = ROOT / "results"
    prefix = f"regional_t2m_2020_{args.cohort}"
    frame = duckdb.sql(f"SELECT * FROM read_parquet('{input_path.as_posix()}')").fetchdf()
    required = [*REFERENCES.values(), *(f"{model}_t2m_c" for model in models)]
    common = frame.dropna(subset=required).copy()
    if common.empty:
        raise RuntimeError("no complete cases shared by both references")
    bootstrap_cfg = cfg["bootstrap"]
    rng = np.random.default_rng(bootstrap_cfg["seed"])
    daily_by_reference = {name: daily_errors(common, target, models) for name, target in REFERENCES.items()}
    n_days = len(daily_by_reference["era5_station"])
    if n_days != len(daily_by_reference["avamet"]):
        raise RuntimeError("references do not share the same daily blocks")
    sampled_days = rng.integers(0, n_days, size=(bootstrap_cfg["n_resamples"], n_days))
    point_scores, pairwise_rows, bootstrap_rows = [], [], []
    winner_draws: dict[str, np.ndarray] = {}
    for reference, target in REFERENCES.items():
        daily = daily_by_reference[reference]
        point = daily[models].mean().to_numpy()
        draw_scores = daily[models].to_numpy()[sampled_days].mean(axis=1)
        winners = np.argmin(draw_scores, axis=1)
        winner_draws[reference] = winners
        base = scores(common, reference, target, models)
        base["mae_rank"] = base["mae_c_day_weighted"].rank(method="min").astype(int)
        base["mae_ci_low_c"], base["mae_ci_high_c"], base["winner_probability"] = zip(
            *(percentile_interval(draw_scores[:, index], bootstrap_cfg["ci"]) + ((winners == index).mean(),) for index in range(len(models))),
            strict=True,
        )
        point_scores.append(base)
        for draw, values in enumerate(draw_scores):
            bootstrap_rows.extend({"draw": draw, "reference": reference, "model": model, "mae_c_day_weighted": value} for model, value in zip(models, values, strict=True))
        for a, b in itertools.combinations(range(len(models)), 2):
            difference = draw_scores[:, a] - draw_scores[:, b]
            lower, upper = percentile_interval(difference, bootstrap_cfg["ci"])
            pairwise_rows.append({
                "reference": reference, "model_a": models[a], "model_b": models[b],
                "delta_mae_a_minus_b_c": point[a] - point[b], "delta_ci_low_c": lower,
                "delta_ci_high_c": upper, "p_a_beats_b": float((difference < 0).mean()),
            })
    annual = pd.concat(point_scores, ignore_index=True).sort_values(["reference", "mae_rank", "model"])
    annual.to_csv(results / f"{prefix}_scores.csv", index=False)
    pd.DataFrame(pairwise_rows).sort_values(["reference", "model_a", "model_b"]).to_csv(results / f"{prefix}_pairwise.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(results / f"{prefix}_bootstrap_draws.csv", index=False)

    reference_effects = []
    era5_draws = daily_by_reference["era5_station"][models].to_numpy()[sampled_days].mean(axis=1)
    avamet_draws = daily_by_reference["avamet"][models].to_numpy()[sampled_days].mean(axis=1)
    era5_point, avamet_point = era5_draws.mean(axis=0), avamet_draws.mean(axis=0)
    for a, b in itertools.combinations(range(len(models)), 2):
        era5_difference = era5_draws[:, a] - era5_draws[:, b]
        avamet_difference = avamet_draws[:, a] - avamet_draws[:, b]
        switch = avamet_difference - era5_difference
        lower, upper = percentile_interval(switch, bootstrap_cfg["ci"])
        reference_effects.append({
            "model_a": models[a], "model_b": models[b],
            "era5_delta_mae_a_minus_b_c": era5_point[a] - era5_point[b],
            "avamet_delta_mae_a_minus_b_c": avamet_point[a] - avamet_point[b],
            "avamet_minus_era5_delta_mae_c": switch.mean(),
            "switch_ci_low_c": lower, "switch_ci_high_c": upper,
            "p_switch_gt_zero": float((switch > 0).mean()),
            "p_pairwise_order_reverses": float((era5_difference * avamet_difference < 0).mean()),
        })
    pd.DataFrame(reference_effects).to_csv(results / f"{prefix}_reference_effects.csv", index=False)

    seasonal = []
    common["season"] = season(common.valid_time)
    for label in ("DJF", "MAM", "JJA", "SON"):
        subset = common.loc[common.season == label]
        for reference, target in REFERENCES.items():
            result = scores(subset, reference, target, models)
            result.insert(0, "season", label)
            result["mae_rank"] = result["mae_c_day_weighted"].rank(method="min").astype(int)
            seasonal.append(result)
    pd.concat(seasonal, ignore_index=True).sort_values(["season", "reference", "mae_rank"]).to_csv(results / f"{prefix}_seasonal_scores.csv", index=False)

    loso = []
    for station_id in sorted(common.station_id.unique()):
        subset = common.loc[common.station_id != station_id]
        for reference, target in REFERENCES.items():
            result = scores(subset, reference, target, models)
            result["mae_rank"] = result["mae_c_day_weighted"].rank(method="min").astype(int)
            winner = result.loc[result.mae_rank == 1, "model"].iloc[0]
            baseline_rank = int(result.loc[result.model == baseline_model, "mae_rank"].iloc[0])
            loso.append({"held_out_station": station_id, "reference": reference, "winner": winner, "baseline_model": baseline_model, "baseline_rank": baseline_rank, "n_common_cases": len(subset)})
    loso_frame = pd.DataFrame(loso)
    loso_frame.to_csv(results / f"{prefix}_leave_one_station_out.csv", index=False)

    winner_by_reference = annual.loc[annual.mae_rank == 1].set_index("reference")["model"].to_dict()
    paired_winner_difference = winner_draws["era5_station"] != winner_draws["avamet"]
    summary = {
        "pilot_name": cfg["pilot_name"], "cohort": args.cohort, "cohort_label": cohort["label"], "models": models,
        "scope": "2020 regional T2m +24 h; fixed common sample, daily block bootstrap and leave-one-station-out stability.",
        "common_cases_both_references": len(common), "daily_blocks": n_days, "stations": common.station_id.nunique(),
        "n_resamples": bootstrap_cfg["n_resamples"], "seed": bootstrap_cfg["seed"],
        "annual_winner_by_reference": winner_by_reference,
        "bootstrap_probability_winners_differ": float(paired_winner_difference.mean()),
        "leave_one_station_out_winner_counts": loso_frame.groupby(["reference", "winner"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "leave_one_station_out_baseline_rank_counts": loso_frame.groupby(["reference", "baseline_rank"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "warning": "The bootstrap quantifies temporal sampling uncertainty. Leave-one-station-out is a sensitivity analysis, not independent spatial replication beyond AVAMET's regional domain. Cohort results must not be pooled across different forecast-initialisation protocols.",
    }
    (results / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(annual.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
