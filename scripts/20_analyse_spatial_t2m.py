#!/usr/bin/env python3
"""Analyse the all-eligible-station spatial T2m replication."""
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
REFERENCES = {"era5_station": "era5_t2m_c", "avamet": "avamet_t2m_qc_c"}


def interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    tail = (1 - confidence) / 2
    return tuple(np.quantile(values, [tail, 1 - tail]).tolist())


def block_indices(rng: np.random.Generator, days: int, draws: int, width: int) -> np.ndarray:
    starts = rng.integers(0, days, size=(draws, int(np.ceil(days / width))))
    return ((starts[:, :, None] + np.arange(width)) % days).reshape(draws, -1)[:, :days]


def daily_errors(frame: pd.DataFrame, reference: str) -> pd.DataFrame:
    errors = {"valid_day": pd.to_datetime(frame.valid_time).dt.date}
    for model in MODELS:
        errors[model] = (frame[f"{model}_t2m_c"] - frame[reference]).abs()
    return pd.DataFrame(errors).groupby("valid_day", as_index=False).mean(numeric_only=True).sort_values("valid_day").reset_index(drop=True)


def analyse_group(frame: pd.DataFrame, lead: int, group_type: str, group: str, cfg: dict) -> tuple[list[dict], list[dict], list[dict]]:
    needed = [*REFERENCES.values(), *(f"{model}_t2m_c" for model in MODELS)]
    common = frame.dropna(subset=needed).copy()
    if common.empty:
        raise RuntimeError(f"no complete cases: lead={lead}; {group_type}={group}")
    daily = {name: daily_errors(common, target) for name, target in REFERENCES.items()}
    if not daily["era5_station"].valid_day.equals(daily["avamet"].valid_day):
        raise RuntimeError("references do not share the same valid-day blocks")
    bootstrap, extension = cfg["bootstrap"], cfg["spatial_replication"]
    primary_a, primary_b = extension["primary_pair"]
    score_rows, effect_rows, summary_rows = [], [], []
    for width in extension["bootstrap_block_days"]:
        rng = np.random.default_rng(bootstrap["seed"] + lead * 1000 + width * 10 + sum(map(ord, group)))
        indices = block_indices(rng, len(daily["era5_station"]), bootstrap["n_resamples"], width)
        sampled, winners = {}, {}
        for reference, values in daily.items():
            point = values[list(MODELS)].mean().to_numpy()
            draws = values[list(MODELS)].to_numpy()[indices].mean(axis=1)
            sampled[reference], winners[reference] = draws, np.argmin(draws, axis=1)
            for position, model in enumerate(MODELS):
                low, high = interval(draws[:, position], bootstrap["ci"])
                score_rows.append({"lead_h": lead, "stratum_type": group_type, "stratum": group, "block_days": width, "reference": reference, "model": model,
                                   "n_common_cases": len(common), "n_stations": common.station_id.nunique(), "n_daily_blocks": len(values), "mae_c_day_weighted": point[position],
                                   "mae_ci_low_c": low, "mae_ci_high_c": high, "winner_probability": float((winners[reference] == position).mean()), "mae_rank": int(pd.Series(point).rank(method="min").iloc[position])})
        pairs = {}
        for first, second in itertools.combinations(range(len(MODELS)), 2):
            era = sampled["era5_station"][:, first] - sampled["era5_station"][:, second]
            ava = sampled["avamet"][:, first] - sampled["avamet"][:, second]
            switch = ava - era
            low, high = interval(switch, bootstrap["ci"])
            row = {"lead_h": lead, "stratum_type": group_type, "stratum": group, "block_days": width, "model_a": MODELS[first], "model_b": MODELS[second],
                   "era5_delta_mae_a_minus_b_c": daily["era5_station"][MODELS[first]].mean() - daily["era5_station"][MODELS[second]].mean(),
                   "avamet_delta_mae_a_minus_b_c": daily["avamet"][MODELS[first]].mean() - daily["avamet"][MODELS[second]].mean(), "avamet_minus_era5_delta_mae_c": switch.mean(),
                   "switch_ci_low_c": low, "switch_ci_high_c": high, "p_switch_gt_zero": float((switch > 0).mean()), "p_pairwise_order_reverses": float((era * ava < 0).mean())}
            pairs[(MODELS[first], MODELS[second])] = row
            effect_rows.append(row)
        primary = pairs[(primary_a, primary_b)]
        summary_rows.append({"lead_h": lead, "stratum_type": group_type, "stratum": group, "block_days": width, "n_common_cases": len(common), "n_stations": common.station_id.nunique(), "n_daily_blocks": len(daily["avamet"]),
                             "avamet_winner": MODELS[int(np.argmin(daily["avamet"][list(MODELS)].mean().to_numpy()))], "era5_winner": MODELS[int(np.argmin(daily["era5_station"][list(MODELS)].mean().to_numpy()))], "p_winners_differ": float((winners["avamet"] != winners["era5_station"]).mean()),
                             "primary_pair": f"{primary_a}__minus__{primary_b}", "primary_effect_c": primary["avamet_minus_era5_delta_mae_c"], "primary_effect_ci_low_c": primary["switch_ci_low_c"], "primary_effect_ci_high_c": primary["switch_ci_high_c"], "primary_pair_reversal_probability": primary["p_pairwise_order_reverses"]})
    return score_rows, effect_rows, summary_rows


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    output = ROOT / cfg["paths"]["spatial_results_directory"]
    output.mkdir(parents=True, exist_ok=True)
    scores, effects, summaries = [], [], []
    for lead in cfg["spatial_replication"]["leads_hours"]:
        source = ROOT / cfg["paths"]["spatial_directory"] / f"lead={lead:03d}" / "batches" / "*.parquet"
        if not list(source.parent.glob("*.parquet")):
            raise FileNotFoundError(f"no spatial batches for lead={lead}")
        frame = duckdb.sql(f"SELECT * FROM read_parquet('{source.as_posix()}')").fetchdf()
        groups = [("all_eligible", "all", frame)]
        for label in ("coast_stratum", "altitude_stratum"):
            groups.extend((label, str(value), subset) for value, subset in frame.groupby(label, sort=True))
        for group_type, group, subset in groups:
            score, effect, summary = analyse_group(subset, lead, group_type, group, cfg)
            scores.extend(score); effects.extend(effect); summaries.extend(summary)
        print(f"analysed spatial lead={lead}; cases={len(frame)}; stations={frame.station_id.nunique()}", flush=True)
    pd.DataFrame(scores).sort_values(["lead_h", "stratum_type", "stratum", "block_days", "reference", "mae_rank"]).to_csv(output / "spatial_scores.csv", index=False)
    pd.DataFrame(effects).sort_values(["lead_h", "stratum_type", "stratum", "block_days", "model_a", "model_b"]).to_csv(output / "spatial_reference_effects.csv", index=False)
    robust = pd.DataFrame(summaries).sort_values(["lead_h", "stratum_type", "stratum", "block_days"])
    robust.to_csv(output / "spatial_robustness.csv", index=False)
    (output / "spatial_summary.json").write_text(json.dumps({"pilot_name": cfg["pilot_name"], "analysis": "all-eligible-station spatial replication", "leads_hours": cfg["spatial_replication"]["leads_hours"], "bootstrap": {"method": "circular moving-block bootstrap on ordered valid days", "n_resamples": cfg["bootstrap"]["n_resamples"], "block_days": cfg["spatial_replication"]["bootstrap_block_days"], "ci": cfg["bootstrap"]["ci"], "seed": cfg["bootstrap"]["seed"]}, "status": "completed"}, indent=2), encoding="utf-8")
    print(robust.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
