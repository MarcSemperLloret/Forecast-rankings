#!/usr/bin/env python3
"""Day-block bootstrap for reference-specific model scores and rankings."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "mvp_t2m.yaml"
INPUT = ROOT / "data" / "interim" / "aligned_t2m_mvp.parquet"
RESULTS = ROOT / "results"
MODELS = ["ifs_hres", "graphcast_hres_init", "pangu_hres_init"]
REFERENCES = {"era5_station": "era5_t2m_c", "avamet": "avamet_t2m_qc_c"}


def daily_absolute_errors(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    values = {"valid_day": pd.to_datetime(frame.valid_time).dt.date}
    for model in MODELS:
        values[model] = (frame[f"{model}_t2m_c"] - frame[target]).abs()
    return pd.DataFrame(values).groupby("valid_day", as_index=False).mean(numeric_only=True)


def bootstrap(daily: pd.DataFrame, n_resamples: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    data = daily[MODELS].to_numpy()
    n_days = len(data)
    score_rows, winner_rows = [], []
    for draw in range(n_resamples):
        sampled = data[rng.integers(0, n_days, size=n_days)].mean(axis=0)
        for model, value in zip(MODELS, sampled, strict=True):
            score_rows.append({"draw": draw, "model": model, "mae_c": value})
        winner_rows.append({"draw": draw, "winner": MODELS[int(sampled.argmin())]})
    return pd.DataFrame(score_rows), pd.DataFrame(winner_rows)


def percentile_interval(values: pd.Series, ci: float) -> tuple[float, float]:
    alpha = (1 - ci) / 2
    return tuple(np.quantile(values, [alpha, 1 - alpha]).tolist())


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    bootstrap_cfg = cfg["bootstrap"]
    frame = duckdb.sql(f"SELECT * FROM read_parquet('{INPUT.as_posix()}')").fetchdf()
    # B→C is the causal comparison in this pilot: every reference must score
    # every model on exactly the same station–initialisation cases. Otherwise a
    # ranking change could be caused by missing AVAMET values instead of truth.
    common_required = [*REFERENCES.values(), *(f"{model}_t2m_c" for model in MODELS)]
    common_frame = frame.dropna(subset=common_required).copy()
    RESULTS.mkdir(parents=True, exist_ok=True)
    summaries, pairwise, draws = [], [], []
    winners_by_reference: dict[str, pd.DataFrame] = {}
    for reference, target in REFERENCES.items():
        daily = daily_absolute_errors(common_frame, target)
        if len(daily) < 2:
            raise RuntimeError(f"{reference}: fewer than two daily blocks")
        score_draws, winner_draws = bootstrap(
            daily, bootstrap_cfg["n_resamples"], bootstrap_cfg["seed"]
        )
        score_draws["reference"] = reference
        winner_draws["reference"] = reference
        draws.append(score_draws)
        winners_by_reference[reference] = winner_draws
        point = daily[MODELS].mean()
        ranks = point.rank(method="min").astype(int)
        for model in MODELS:
            interval = percentile_interval(score_draws.loc[score_draws.model == model, "mae_c"], bootstrap_cfg["ci"])
            summaries.append(
                {
                    "reference": reference,
                    "model": model,
                    "n_common_cases": len(common_frame),
                    "n_daily_blocks": len(daily),
                    "mae_c": point[model],
                    "mae_rank": ranks[model],
                    "mae_ci_low_c": interval[0],
                    "mae_ci_high_c": interval[1],
                    "winner_probability": (winner_draws.winner == model).mean(),
                }
            )
        for model_a, model_b in itertools.combinations(MODELS, 2):
            difference = score_draws.loc[score_draws.model == model_a, "mae_c"].to_numpy() - score_draws.loc[score_draws.model == model_b, "mae_c"].to_numpy()
            interval = percentile_interval(pd.Series(difference), bootstrap_cfg["ci"])
            pairwise.append(
                {
                    "reference": reference,
                    "model_a": model_a,
                    "model_b": model_b,
                    "delta_mae_a_minus_b_c": point[model_a] - point[model_b],
                    "delta_ci_low_c": interval[0],
                    "delta_ci_high_c": interval[1],
                    "p_a_beats_b": float((difference < 0).mean()),
                }
            )
    merged_winners = winners_by_reference["era5_station"].merge(
        winners_by_reference["avamet"], on="draw", suffixes=("_era5", "_avamet"), validate="one_to_one"
    )
    summary = pd.DataFrame(summaries).sort_values(["reference", "mae_rank", "model"])
    pairwise_frame = pd.DataFrame(pairwise).sort_values(["reference", "model_a", "model_b"])
    draw_frame = pd.concat(draws, ignore_index=True)
    summary.to_csv(RESULTS / "bootstrap_scores_t2m_q1.csv", index=False)
    pairwise_frame.to_csv(RESULTS / "bootstrap_pairwise_t2m_q1.csv", index=False)
    draw_frame.to_csv(RESULTS / "bootstrap_draws_t2m_q1.csv", index=False)
    decision = {
        "scope": "Q1 2020 regional pilot; day-block bootstrap preserves within-day station dependence",
        "common_cases_both_references": len(common_frame),
        "n_resamples": bootstrap_cfg["n_resamples"],
        "seed": bootstrap_cfg["seed"],
        "rank_winner_differs_point_estimate": bool(
            summary.loc[summary.reference == "era5_station"].sort_values("mae_rank").iloc[0].model
            != summary.loc[summary.reference == "avamet"].sort_values("mae_rank").iloc[0].model
        ),
        "bootstrap_probability_winners_differ": float((merged_winners.winner_era5 != merged_winners.winner_avamet).mean()),
        "warning": "This bootstrap quantifies temporal sampling uncertainty only; spatial generalisation needs a wider station network.",
    }
    (RESULTS / "bootstrap_summary_t2m_q1.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
