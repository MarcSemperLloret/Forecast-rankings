#!/usr/bin/env python3
"""Analyse pre-specified T2m lead times with circular temporal block bootstrap."""
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
REFERENCES = {"era5_station": "era5_t2m_c", "avamet": "avamet_t2m_qc_c"}


def percentile_interval(values: np.ndarray, ci: float) -> tuple[float, float]:
    alpha = (1 - ci) / 2
    return tuple(np.quantile(values, [alpha, 1 - alpha]).tolist())


def circular_block_indices(rng: np.random.Generator, n_days: int, n_resamples: int, block_days: int) -> np.ndarray:
    n_blocks = int(np.ceil(n_days / block_days))
    starts = rng.integers(0, n_days, size=(n_resamples, n_blocks))
    offsets = np.arange(block_days)
    return ((starts[:, :, None] + offsets) % n_days).reshape(n_resamples, -1)[:, :n_days]


def daily_errors(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    values = {"valid_day": pd.to_datetime(frame.valid_time).dt.date}
    for model in MODELS:
        values[model] = (frame[f"{model}_t2m_c"] - frame[target]).abs()
    return pd.DataFrame(values).groupby("valid_day", as_index=False).mean(numeric_only=True).sort_values("valid_day").reset_index(drop=True)


def panel_path(cfg: dict, lead: int) -> Path:
    if lead == 24:
        return ROOT / cfg["paths"]["annual_aligned_parquet"]
    return ROOT / cfg["paths"]["lead_time_directory"] / f"lead={lead:03d}" / "annual.parquet"


def analyse_lead(frame: pd.DataFrame, lead: int, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    required = [*REFERENCES.values(), *(f"{model}_t2m_c" for model in MODELS)]
    common = frame.dropna(subset=required).copy()
    if common.empty:
        raise RuntimeError(f"lead={lead} has no complete cases")
    daily = {reference: daily_errors(common, target) for reference, target in REFERENCES.items()}
    if not daily["era5_station"].valid_day.equals(daily["avamet"].valid_day):
        raise RuntimeError(f"lead={lead} references do not share daily blocks")
    n_days = len(daily["era5_station"])
    bootstrap = cfg["bootstrap"]
    extension = cfg["lead_time_extension"]
    score_rows, effect_rows, summary_rows = [], [], []
    primary_a, primary_b = extension["primary_pair"]
    for block_days in extension["bootstrap_block_days"]:
        rng = np.random.default_rng(bootstrap["seed"] + lead * 100 + block_days)
        indices = circular_block_indices(rng, n_days, bootstrap["n_resamples"], block_days)
        draws: dict[str, np.ndarray] = {}
        winners: dict[str, np.ndarray] = {}
        for reference in REFERENCES:
            values = daily[reference][list(MODELS)].to_numpy()
            point = values.mean(axis=0)
            sampled = values[indices].mean(axis=1)
            draws[reference] = sampled
            winners[reference] = np.argmin(sampled, axis=1)
            for index, model in enumerate(MODELS):
                lower, upper = percentile_interval(sampled[:, index], bootstrap["ci"])
                score_rows.append({
                    "lead_h": lead, "block_days": block_days, "reference": reference, "model": model,
                    "n_common_cases": len(common), "n_daily_blocks": n_days, "mae_c_day_weighted": point[index],
                    "mae_ci_low_c": lower, "mae_ci_high_c": upper,
                    "winner_probability": float((winners[reference] == index).mean()),
                    "mae_rank": int(pd.Series(point).rank(method="min").iloc[index]),
                })
        effects_by_pair: dict[tuple[str, str], dict] = {}
        for a, b in itertools.combinations(range(len(MODELS)), 2):
            era5_difference = draws["era5_station"][:, a] - draws["era5_station"][:, b]
            avamet_difference = draws["avamet"][:, a] - draws["avamet"][:, b]
            switch = avamet_difference - era5_difference
            lower, upper = percentile_interval(switch, bootstrap["ci"])
            row = {
                "lead_h": lead, "block_days": block_days, "model_a": MODELS[a], "model_b": MODELS[b],
                "era5_delta_mae_a_minus_b_c": daily["era5_station"][MODELS[a]].mean() - daily["era5_station"][MODELS[b]].mean(),
                "avamet_delta_mae_a_minus_b_c": daily["avamet"][MODELS[a]].mean() - daily["avamet"][MODELS[b]].mean(),
                "avamet_minus_era5_delta_mae_c": switch.mean(), "switch_ci_low_c": lower, "switch_ci_high_c": upper,
                "p_switch_gt_zero": float((switch > 0).mean()),
                "p_pairwise_order_reverses": float((era5_difference * avamet_difference < 0).mean()),
            }
            effects_by_pair[(MODELS[a], MODELS[b])] = row
            effect_rows.append(row)
        primary = effects_by_pair[(primary_a, primary_b)]
        summary_rows.append({
            "lead_h": lead, "block_days": block_days, "n_common_cases": len(common), "n_daily_blocks": n_days,
            "avamet_winner": MODELS[int(np.argmin(daily["avamet"][list(MODELS)].mean().to_numpy()))],
            "era5_winner": MODELS[int(np.argmin(daily["era5_station"][list(MODELS)].mean().to_numpy()))],
            "p_winners_differ": float((winners["avamet"] != winners["era5_station"]).mean()),
            "primary_pair": f"{primary_a}__minus__{primary_b}",
            "primary_effect_c": primary["avamet_minus_era5_delta_mae_c"],
            "primary_effect_ci_low_c": primary["switch_ci_low_c"],
            "primary_effect_ci_high_c": primary["switch_ci_high_c"],
            "primary_pair_reversal_probability": primary["p_pairwise_order_reverses"],
        })
    return pd.DataFrame(score_rows), pd.DataFrame(effect_rows), summary_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leads", nargs="+", type=int, help="subset of pre-specified lead hours; default all")
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    planned = cfg["lead_time_extension"]["leads_hours"]
    leads = args.leads or planned
    unknown = set(leads).difference(planned)
    if unknown:
        raise ValueError(f"unplanned lead hours: {sorted(unknown)}")
    result_dir = ROOT / cfg["paths"]["lead_time_results_directory"]
    result_dir.mkdir(parents=True, exist_ok=True)
    all_scores, all_effects, all_summaries = [], [], []
    for lead in leads:
        input_path = panel_path(cfg, lead)
        if not input_path.exists():
            raise FileNotFoundError(f"lead={lead} is not consolidated: {input_path}")
        frame = duckdb.sql(f"SELECT * FROM read_parquet('{input_path.as_posix()}')").fetchdf()
        scores, effects, summaries = analyse_lead(frame, lead, cfg)
        all_scores.append(scores)
        all_effects.append(effects)
        all_summaries.extend(summaries)
        print(f"analysed lead={lead}; cases={summaries[0]['n_common_cases']}; days={summaries[0]['n_daily_blocks']}", flush=True)
    scores = pd.concat(all_scores, ignore_index=True).sort_values(["lead_h", "block_days", "reference", "mae_rank"])
    effects = pd.concat(all_effects, ignore_index=True).sort_values(["lead_h", "block_days", "model_a", "model_b"])
    summaries = pd.DataFrame(all_summaries).sort_values(["lead_h", "block_days"])
    scores.to_csv(result_dir / "hres_multilead_scores.csv", index=False)
    effects.to_csv(result_dir / "hres_multilead_reference_effects.csv", index=False)
    summaries.to_csv(result_dir / "hres_multilead_robustness.csv", index=False)
    manifest = {
        "pilot_name": cfg["pilot_name"], "cohort": "hres_initialised", "leads_hours": leads,
        "bootstrap": {"method": "circular moving-block bootstrap on ordered valid days", "block_days": cfg["lead_time_extension"]["bootstrap_block_days"],
                      "n_resamples": cfg["bootstrap"]["n_resamples"], "ci": cfg["bootstrap"]["ci"], "seed": cfg["bootstrap"]["seed"]},
        "primary_pair": cfg["lead_time_extension"]["primary_pair"],
        "status": "completed only for supplied consolidated lead-time panels",
    }
    (result_dir / "hres_multilead_summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summaries.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
