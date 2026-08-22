#!/usr/bin/env python3
"""Quantify T2m ranking sensitivity to station--grid elevation mismatch."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
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


def daily_errors(frame: pd.DataFrame, reference: str, suffix: str) -> pd.DataFrame:
    values = {"valid_day": pd.to_datetime(frame.valid_time).dt.date}
    target = REFERENCES[reference] if not suffix else f"{reference}{suffix}"
    for model in MODELS:
        values[model] = (frame[f"{model}_t2m_c{suffix}"] - frame[target]).abs()
    return pd.DataFrame(values).groupby("valid_day", as_index=False).mean(numeric_only=True).sort_values("valid_day").reset_index(drop=True)


def analyse_subset(frame: pd.DataFrame, lead: int, label: str, suffix: str, cfg: dict) -> tuple[list[dict], list[dict]]:
    reference_columns = list(REFERENCES.values()) if not suffix else [f"{reference}{suffix}" for reference in REFERENCES]
    required = [f"{model}_t2m_c{suffix}" for model in MODELS] + reference_columns
    common = frame.dropna(subset=required)
    if common.empty:
        raise RuntimeError(f"no complete cases for {label} lead={lead}")
    daily = {reference: daily_errors(common, reference, suffix) for reference in REFERENCES}
    if not daily["era5_station"].valid_day.equals(daily["avamet"].valid_day):
        raise RuntimeError("reference day series differ")
    bootstrap = cfg["bootstrap"]
    primary_a, primary_b = "graphcast_hres_init", "pangu_hres_init"
    score_rows, summary_rows = [], []
    for width in cfg["spatial_replication"]["bootstrap_block_days"]:
        rng = np.random.default_rng(bootstrap["seed"] + lead * 1000 + width * 10 + sum(map(ord, label)))
        indices = block_indices(rng, len(daily["avamet"]), bootstrap["n_resamples"], width)
        draws, winners = {}, {}
        for reference, values in daily.items():
            point = values[list(MODELS)].mean().to_numpy()
            sampled = values[list(MODELS)].to_numpy()[indices].mean(axis=1)
            draws[reference], winners[reference] = sampled, np.argmin(sampled, axis=1)
            for position, model in enumerate(MODELS):
                low, high = interval(sampled[:, position], bootstrap["ci"])
                score_rows.append({"lead_h": lead, "sensitivity": label, "block_days": width, "reference": reference, "model": model, "n_common_cases": len(common), "n_stations": common.station_id.nunique(), "n_daily_blocks": len(values), "mae_c_day_weighted": point[position], "mae_ci_low_c": low, "mae_ci_high_c": high, "winner_probability": float((winners[reference] == position).mean()), "mae_rank": int(pd.Series(point).rank(method="min").iloc[position])})
        era = draws["era5_station"][:, MODELS.index(primary_a)] - draws["era5_station"][:, MODELS.index(primary_b)]
        ava = draws["avamet"][:, MODELS.index(primary_a)] - draws["avamet"][:, MODELS.index(primary_b)]
        switch = ava - era
        low, high = interval(switch, bootstrap["ci"])
        summary_rows.append({"lead_h": lead, "sensitivity": label, "block_days": width, "n_common_cases": len(common), "n_stations": common.station_id.nunique(), "n_daily_blocks": len(daily["avamet"]), "avamet_winner": MODELS[int(np.argmin(daily["avamet"][list(MODELS)].mean().to_numpy()))], "era5_winner": MODELS[int(np.argmin(daily["era5_station"][list(MODELS)].mean().to_numpy()))], "p_winners_differ": float((winners["avamet"] != winners["era5_station"]).mean()), "primary_effect_c": switch.mean(), "primary_effect_ci_low_c": low, "primary_effect_ci_high_c": high, "primary_pair_reversal_probability": float((era * ava < 0).mean())})
    return score_rows, summary_rows


def station_effects(frame: pd.DataFrame, lead: int) -> pd.DataFrame:
    required = [*(f"{model}_t2m_c" for model in MODELS), *REFERENCES.values()]
    values = frame.dropna(subset=required).copy()
    rows = []
    for station_id, subset in values.groupby("station_id", sort=True):
        daily = {reference: daily_errors(subset, reference, "") for reference in REFERENCES}
        era = daily["era5_station"]["graphcast_hres_init"].mean() - daily["era5_station"]["pangu_hres_init"].mean()
        ava = daily["avamet"]["graphcast_hres_init"].mean() - daily["avamet"]["pangu_hres_init"].mean()
        rows.append({"lead_h": lead, "station_id": station_id, "station_effect_c": ava - era, "n_common_cases": len(subset), "n_daily_blocks": len(daily["avamet"])})
    return pd.DataFrame(rows)


def plot_effects(stations: pd.DataFrame, output: Path) -> None:
    leads = sorted(stations.lead_h.unique())
    figure, axes = plt.subplots(1, len(leads), figsize=(12, 3.6), sharey=True, layout="constrained")
    markers = {"coast_0_25km": "o", "interior_gt25km": "^"}
    for axis, lead in zip(np.atleast_1d(axes), leads, strict=True):
        values = stations[stations.lead_h == lead]
        for group, marker in markers.items():
            subset = values[values.coast_stratum == group]
            axis.scatter(subset.abs_delta_z_m, subset.station_effect_c, s=24, marker=marker, alpha=0.75, color="#0072B2", edgecolors="black", linewidths=0.25, label=group.replace("_", " "))
        rho = values.abs_delta_z_m.corr(values.station_effect_c, method="spearman")
        axis.axhline(0, color="black", linewidth=0.8, linestyle="--")
        for threshold in (50, 100, 200):
            axis.axvline(threshold, color="#666666", linewidth=0.6, linestyle=":")
        axis.set(title=f"+{lead} h; Spearman ρ={rho:.2f}", xlabel="|Δz| estación − rejilla ERA5 (m)")
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
    axes[0].set_ylabel("Efecto de referencia GraphCast − Pangu (°C)\n[AVAMET menos ERA5]")
    axes[-1].legend(title="Estrato", fontsize=7, title_fontsize=8, frameon=False, loc="best")
    figure.savefig(output.with_suffix(".png"), dpi=300, facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    paths, sensitivity = cfg["paths"], cfg["orography_sensitivity"]
    terrain = pd.read_csv(ROOT / paths["station_grid_elevation_csv"])
    result_dir = ROOT / paths["orography_results_directory"]
    result_dir.mkdir(parents=True, exist_ok=True)
    score_rows, summary_rows, station_rows = [], [], []
    for lead in cfg["spatial_replication"]["leads_hours"]:
        source = ROOT / paths["spatial_directory"] / f"lead={lead:03d}" / "batches" / "*.parquet"
        frame = duckdb.sql(f"SELECT * FROM read_parquet('{source.as_posix()}')").fetchdf().merge(terrain[["station_id", "era5_grid_elevation_m", "delta_z_m", "abs_delta_z_m"]], on="station_id", validate="many_to_one")
        score, summary = analyse_subset(frame, lead, "raw_all_stations", "", cfg); score_rows.extend(score); summary_rows.extend(summary)
        for threshold in sensitivity["abs_delta_z_thresholds_m"]:
            subset = frame[frame.abs_delta_z_m < threshold]
            score, summary = analyse_subset(subset, lead, f"raw_abs_delta_z_lt_{threshold:g}m", "", cfg); score_rows.extend(score); summary_rows.extend(summary)
        for lapse_rate in sensitivity["lapse_rates_k_per_km"]:
            suffix = f"_lapse_{lapse_rate:g}kpkm".replace(".", "p")
            correction = lapse_rate * frame.delta_z_m / 1000
            adjusted = frame.copy()
            for model in MODELS:
                adjusted[f"{model}_t2m_c{suffix}"] = adjusted[f"{model}_t2m_c"] - correction
            adjusted[f"era5_station{suffix}"] = adjusted["era5_t2m_c"] - correction
            adjusted[f"avamet{suffix}"] = adjusted["avamet_t2m_qc_c"]
            score, summary = analyse_subset(adjusted, lead, f"lapse_rate_{lapse_rate:g}_K_per_km_all_stations", suffix, cfg); score_rows.extend(score); summary_rows.extend(summary)
        station_rows.append(station_effects(frame, lead))
        print(f"analysed terrain sensitivity lead={lead}", flush=True)
    stations = pd.concat(station_rows, ignore_index=True).merge(terrain, on="station_id", validate="many_to_one").merge(pd.read_csv(ROOT / paths["spatial_stations_csv"])[["station_id", "coast_stratum"]], on="station_id", validate="many_to_one")
    stations.to_csv(result_dir / "station_level_effect_vs_delta_z.csv", index=False)
    pd.DataFrame(score_rows).sort_values(["lead_h", "sensitivity", "block_days", "reference", "mae_rank"]).to_csv(result_dir / "orography_scores.csv", index=False)
    robustness = pd.DataFrame(summary_rows).sort_values(["lead_h", "sensitivity", "block_days"])
    robustness.to_csv(result_dir / "orography_robustness.csv", index=False)
    correlations = stations.groupby("lead_h").apply(lambda x: x.abs_delta_z_m.corr(x.station_effect_c, method="spearman"), include_groups=False).rename("spearman_rho_station_effect_vs_abs_delta_z").reset_index()
    correlations.to_csv(result_dir / "station_effect_delta_z_spearman.csv", index=False)
    plot_effects(stations, result_dir / "station_effect_vs_abs_delta_z")
    (result_dir / "station_effect_vs_abs_delta_z_alt_text.md").write_text("Tres paneles (+24, +48 y +72 h). Cada punto es una estación; el eje horizontal muestra el desajuste absoluto de elevación entre estación y orografía ERA5 interpolada, y el vertical el cambio de diferencia MAE GraphCast menos Pangu al pasar de ERA5 a AVAMET. Círculos: costa ≤25 km; triángulos: interior >25 km. Las líneas verticales marcan 50, 100 y 200 m; la horizontal, efecto nulo.", encoding="utf-8")
    (result_dir / "orography_summary.json").write_text(json.dumps({"terrain": "ERA5 geopotential_at_surface / standard gravity; model-specific terrain unavailable in published HRES stores", "thresholds_m": sensitivity["abs_delta_z_thresholds_m"], "lapse_rates_k_per_km": sensitivity["lapse_rates_k_per_km"], "method": "same daily circular block bootstrap as spatial replication", "status": "completed"}, indent=2), encoding="utf-8")
    print(robustness[robustness.block_days.eq(7)].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(correlations.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
