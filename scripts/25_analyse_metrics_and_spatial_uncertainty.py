#!/usr/bin/env python3
"""Add error metrics and a nested temporal--spatial cluster bootstrap."""
from __future__ import annotations

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


def daily_metric(frame: pd.DataFrame, reference: str, metric: str) -> pd.DataFrame:
    values = {"valid_day": pd.to_datetime(frame.valid_time).dt.date}
    for model in MODELS:
        error = frame[f"{model}_t2m_c"] - frame[REFERENCES[reference]]
        if metric == "mae":
            score = error.abs()
        elif metric == "rmse":
            score = error**2
        elif metric == "bias":
            score = error
        elif metric == "medae":
            score = error.abs()
        else:
            raise ValueError(metric)
        values[model] = score
    daily = pd.DataFrame(values).groupby("valid_day", as_index=False)
    if metric == "rmse":
        return daily.mean(numeric_only=True).assign(**{model: lambda item, model=model: np.sqrt(item[model]) for model in MODELS}).sort_values("valid_day").reset_index(drop=True)
    if metric == "medae":
        return daily.median(numeric_only=True).sort_values("valid_day").reset_index(drop=True)
    return daily.mean(numeric_only=True).sort_values("valid_day").reset_index(drop=True)


def analyse_metrics(frame: pd.DataFrame, lead: int, cfg: dict) -> tuple[list[dict], list[dict]]:
    required = [*REFERENCES.values(), *(f"{model}_t2m_c" for model in MODELS)]
    common = frame.dropna(subset=required)
    bootstrap, extension = cfg["bootstrap"], cfg["metric_and_spatial_uncertainty_extension"]
    primary_a, primary_b = extension["primary_pair"]
    scores, summary = [], []
    for metric in extension["metrics"]:
        daily = {reference: daily_metric(common, reference, metric) for reference in REFERENCES}
        if not daily["era5_station"].valid_day.equals(daily["avamet"].valid_day):
            raise RuntimeError("reference day series differ")
        for width in extension["temporal_block_days"]:
            rng = np.random.default_rng(bootstrap["seed"] + lead * 10000 + width * 100 + sum(map(ord, metric)))
            indices = block_indices(rng, len(daily["avamet"]), bootstrap["n_resamples"], width)
            draws, winners = {}, {}
            for reference, values in daily.items():
                point = values[list(MODELS)].mean().to_numpy()
                sampled = values[list(MODELS)].to_numpy()[indices].mean(axis=1)
                draws[reference] = sampled
                if metric != "bias":
                    winners[reference] = np.argmin(sampled, axis=1)
                for position, model in enumerate(MODELS):
                    low, high = interval(sampled[:, position], bootstrap["ci"])
                    scores.append({"lead_h": lead, "metric": metric, "bootstrap": "temporal_circular_blocks", "temporal_block_days": width, "reference": reference, "model": model, "n_common_cases": len(common), "n_stations": common.station_id.nunique(), "n_daily_blocks": len(values), "score": point[position], "score_ci_low": low, "score_ci_high": high, "bootstrap_winner_frequency": float((winners[reference] == position).mean()) if metric != "bias" else np.nan, "rank": int(pd.Series(point).rank(method="min").iloc[position]) if metric != "bias" else np.nan, "ranking_applicable": metric != "bias"})
            if metric == "bias":
                for position, model in enumerate(MODELS):
                    shift = draws["avamet"][:, position] - draws["era5_station"][:, position]
                    low, high = interval(shift, bootstrap["ci"])
                    summary.append({"lead_h": lead, "metric": metric, "bootstrap": "temporal_circular_blocks", "temporal_block_days": width, "n_common_cases": len(common), "n_stations": common.station_id.nunique(), "n_daily_blocks": len(daily["avamet"]), "model": model, "analysis_role": "diagnostic_signed_mean_error_reference_shift; no ranking", "mean_error_era5": daily["era5_station"][model].mean(), "mean_error_avamet": daily["avamet"][model].mean(), "avamet_minus_era5_mean_error": shift.mean(), "avamet_minus_era5_mean_error_ci_low": low, "avamet_minus_era5_mean_error_ci_high": high})
                continue
            era = draws["era5_station"][:, MODELS.index(primary_a)] - draws["era5_station"][:, MODELS.index(primary_b)]
            ava = draws["avamet"][:, MODELS.index(primary_a)] - draws["avamet"][:, MODELS.index(primary_b)]
            switch = ava - era
            low, high = interval(switch, bootstrap["ci"])
            summary.append({"lead_h": lead, "metric": metric, "bootstrap": "temporal_circular_blocks", "temporal_block_days": width, "n_common_cases": len(common), "n_stations": common.station_id.nunique(), "n_daily_blocks": len(daily["avamet"]), "avamet_winner": MODELS[int(np.argmin(daily["avamet"][list(MODELS)].mean().to_numpy()))], "era5_winner": MODELS[int(np.argmin(daily["era5_station"][list(MODELS)].mean().to_numpy()))], "bootstrap_winners_differ_frequency": float((winners["avamet"] != winners["era5_station"]).mean()), "primary_effect": switch.mean(), "primary_effect_ci_low": low, "primary_effect_ci_high": high, "bootstrap_pairwise_ranking_reversal_frequency": float((era * ava < 0).mean())})
    return scores, summary


def cluster_daily_mae(frame: pd.DataFrame, reference: str, spatial_degree: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = frame.copy()
    values["valid_day"] = pd.to_datetime(values.valid_time).dt.date
    values["spatial_block"] = (np.floor(values.latitude / spatial_degree).astype(int).astype(str) + "_" + np.floor(values.longitude / spatial_degree).astype(int).astype(str))
    blocks = np.sort(values.spatial_block.unique())
    weights = values[["station_id", "spatial_block"]].drop_duplicates().spatial_block.value_counts().reindex(blocks).to_numpy(float)
    errors = {model: (values[f"{model}_t2m_c"] - values[REFERENCES[reference]]).abs() for model in MODELS}
    grouped = pd.DataFrame({"valid_day": values.valid_day, "spatial_block": values.spatial_block, **errors}).groupby(["valid_day", "spatial_block"], as_index=False).mean(numeric_only=True)
    complete_days = grouped.groupby("valid_day").spatial_block.nunique()
    days = np.array(sorted(complete_days[complete_days == len(blocks)].index))
    grouped = grouped[grouped.valid_day.isin(days)]
    cube = np.stack([grouped.pivot(index="valid_day", columns="spatial_block", values=model).reindex(index=days, columns=blocks).to_numpy() for model in MODELS], axis=2)
    if not np.isfinite(cube).all():
        raise RuntimeError("incomplete spatial-block daily cube")
    return cube, weights, days


def spatial_temporal_bootstrap(frame: pd.DataFrame, lead: int, cfg: dict) -> list[dict]:
    required = [*REFERENCES.values(), *(f"{model}_t2m_c" for model in MODELS)]
    common = frame.dropna(subset=required)
    bootstrap, extension = cfg["bootstrap"], cfg["metric_and_spatial_uncertainty_extension"]
    first, second = (MODELS.index(model) for model in extension["primary_pair"])
    rows = []
    for spatial_degree in extension["spatial_block_degrees"]:
        cubes, weights, days = {}, None, None
        for reference in REFERENCES:
            cube, weights, days = cluster_daily_mae(common, reference, spatial_degree)
            cubes[reference] = cube
        assert weights is not None and days is not None
        for width in extension["temporal_block_days"]:
            rng = np.random.default_rng(bootstrap["seed"] + lead * 10000 + int(spatial_degree * 1000) + width)
            time_indices = block_indices(rng, len(days), bootstrap["n_resamples"], width)
            spatial_indices = rng.integers(0, len(weights), size=(bootstrap["n_resamples"], len(weights)))
            draws = {reference: np.empty((bootstrap["n_resamples"], len(MODELS))) for reference in REFERENCES}
            for draw in range(bootstrap["n_resamples"]):
                sampled_weights = weights[spatial_indices[draw]]
                for reference in REFERENCES:
                    sampled = cubes[reference][time_indices[draw]][:, spatial_indices[draw], :]
                    draws[reference][draw] = np.average(sampled.reshape(-1, len(MODELS)), axis=0, weights=np.tile(sampled_weights, sampled.shape[0]))
            winners = {reference: np.argmin(values, axis=1) for reference, values in draws.items()}
            era, ava = draws["era5_station"][:, first] - draws["era5_station"][:, second], draws["avamet"][:, first] - draws["avamet"][:, second]
            switch = ava - era
            low, high = interval(switch, bootstrap["ci"])
            rows.append({"lead_h": lead, "metric": "mae", "bootstrap": "nested_temporal_and_spatial_cluster", "spatial_block_degrees": spatial_degree, "n_spatial_blocks": len(weights), "temporal_block_days": width, "n_complete_spatial_days": len(days), "n_common_cases": len(common), "n_stations": common.station_id.nunique(), "avamet_winner": MODELS[int(np.argmin(draws["avamet"].mean(axis=0)))], "era5_winner": MODELS[int(np.argmin(draws["era5_station"].mean(axis=0)))], "bootstrap_winners_differ_frequency": float((winners["avamet"] != winners["era5_station"]).mean()), "primary_effect_c": switch.mean(), "primary_effect_ci_low_c": low, "primary_effect_ci_high_c": high, "bootstrap_pairwise_ranking_reversal_frequency": float((era * ava < 0).mean())})
    return rows


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    output = ROOT / cfg["paths"]["metric_and_spatial_results_directory"]
    output.mkdir(parents=True, exist_ok=True)
    scores, metric_summary, spatial_summary = [], [], []
    for lead in cfg["spatial_replication"]["leads_hours"]:
        source = ROOT / cfg["paths"]["spatial_directory"] / f"lead={lead:03d}" / "batches" / "*.parquet"
        frame = duckdb.sql(f"SELECT * FROM read_parquet('{source.as_posix()}')").fetchdf()
        score, summary = analyse_metrics(frame, lead, cfg); scores.extend(score); metric_summary.extend(summary)
        spatial_summary.extend(spatial_temporal_bootstrap(frame, lead, cfg))
        print(f"analysed metrics and spatial uncertainty lead={lead}", flush=True)
    pd.DataFrame(scores).sort_values(["lead_h", "metric", "temporal_block_days", "reference", "rank"]).to_csv(output / "metric_scores.csv", index=False)
    pd.DataFrame(metric_summary).sort_values(["lead_h", "metric", "temporal_block_days"]).to_csv(output / "metric_robustness.csv", index=False)
    pd.DataFrame(spatial_summary).sort_values(["lead_h", "spatial_block_degrees", "temporal_block_days"]).to_csv(output / "spatiotemporal_cluster_bootstrap.csv", index=False)
    (output / "summary.json").write_text(json.dumps({"mae_status": "primary metric retained", "additional_metrics": cfg["metric_and_spatial_uncertainty_extension"]["metrics"], "spatial_bootstrap": "nested resampling of fixed lat-lon station clusters and circular temporal blocks; cluster weights equal station counts", "probability_label": "bootstrap frequency, not a p-value", "status": "completed"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
