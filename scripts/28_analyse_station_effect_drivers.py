#!/usr/bin/env python3
"""Relate the station-level reference effect to pre-listed station covariates.

The effect at a station is the change in the GraphCast-minus-Pangu MAE gap when
the reference moves from ERA5 to AVAMET. It is linear in the daily errors, so
it reduces to one station-by-day matrix whose row means are the station
effects. Resampling days therefore updates the effects and the day-dependent
covariates together, which is what a correlation between them needs.

The relation is descriptive. The covariates come from the frozen configuration
and none is chosen after seeing an outcome.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "regional_t2m_2020.yaml"
MODELS = ("ifs_hres", "graphcast_hres_init", "pangu_hres_init")
STATIC_COVARIATES = ("abs_delta_z_m", "delta_z_m", "altitude_m", "coast_distance_km", "stations_in_cell")


def interval(values: np.ndarray, confidence: float, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    tail = (1 - confidence) / 2
    return np.quantile(values, tail, axis=axis), np.quantile(values, 1 - tail, axis=axis)


def block_indices(rng: np.random.Generator, days: int, draws: int, width: int) -> np.ndarray:
    starts = rng.integers(0, days, size=(draws, int(np.ceil(days / width))))
    return ((starts[:, :, None] + np.arange(width)) % days).reshape(draws, -1)[:, :days]


def day_weights(indices: np.ndarray, days: int) -> np.ndarray:
    """How many times each valid day enters each bootstrap draw."""
    draws = indices.shape[0]
    weights = np.zeros((days, draws))
    np.add.at(weights, (indices.ravel(), np.repeat(np.arange(draws), indices.shape[1])), 1.0)
    return weights


class DayPanel:
    """Station-by-day quantities, and their means under any day weighting."""

    def __init__(self, frame: pd.DataFrame, pair: tuple[str, str]) -> None:
        first, second = pair
        values = frame.copy()
        values["valid_day"] = pd.to_datetime(values.valid_time).dt.date
        observed, reanalysis = values.avamet_t2m_qc_c, values.era5_t2m_c
        gap_avamet = (values[f"{first}_t2m_c"] - observed).abs() - (values[f"{second}_t2m_c"] - observed).abs()
        gap_era5 = (values[f"{first}_t2m_c"] - reanalysis).abs() - (values[f"{second}_t2m_c"] - reanalysis).abs()
        hour = pd.to_datetime(values.valid_time).dt.hour
        quantities = {"effect": gap_avamet - gap_era5, "observed": observed, "observed_squared": observed**2,
                      "difference": reanalysis - observed, "difference_squared": (reanalysis - observed) ** 2,
                      "observed_12utc": observed.where(hour == 12), "observed_00utc": observed.where(hour == 0)}
        daily = pd.DataFrame({"station_id": values.station_id, "valid_day": values.valid_day, **quantities})
        daily = daily.groupby(["station_id", "valid_day"], as_index=False).mean(numeric_only=True)
        self.stations = np.sort(values.station_id.unique())
        self.days = np.sort(values.valid_day.unique())
        self.matrices = {name: daily.pivot(index="station_id", columns="valid_day", values=name)
                         .reindex(index=self.stations, columns=self.days).to_numpy() for name in quantities}

    def means(self, name: str, weights: np.ndarray | None) -> np.ndarray:
        matrix = self.matrices[name]
        present = np.isfinite(matrix)
        filled = np.where(present, matrix, 0.0)
        if weights is None:
            return filled.sum(axis=1) / present.sum(axis=1)
        return (filled @ weights) / (present.astype(float) @ weights)

    def effects_and_covariates(self, weights: np.ndarray | None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        observed = self.means("observed", weights)
        variance = np.clip(self.means("observed_squared", weights) - observed**2, 0.0, None)
        covariates = {"avamet_mean_t2m_c": observed, "avamet_sd_t2m_c": np.sqrt(variance),
                      "avamet_contrast_12_minus_00_c": self.means("observed_12utc", weights) - self.means("observed_00utc", weights),
                      "era5_minus_avamet_bias_c": self.means("difference", weights),
                      "era5_minus_avamet_rmse_c": np.sqrt(self.means("difference_squared", weights))}
        return self.means("effect", weights), covariates


def spearman(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Rank correlation down the station axis, column by column."""
    ranked_first, ranked_second = rankdata(first, axis=0), rankdata(second, axis=0)
    centred_first = ranked_first - ranked_first.mean(axis=0)
    centred_second = ranked_second - ranked_second.mean(axis=0)
    denominator = np.sqrt((centred_first**2).sum(axis=0) * (centred_second**2).sum(axis=0))
    return (centred_first * centred_second).sum(axis=0) / denominator


def standardised_regression(effects: np.ndarray, covariates: pd.DataFrame) -> tuple[pd.Series, float]:
    usable = covariates.notna().all(axis=1).to_numpy()
    design = covariates[usable].to_numpy(float)
    design = (design - design.mean(axis=0)) / design.std(axis=0)
    target = effects[usable]
    target = (target - target.mean()) / target.std()
    coefficients, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(design)), design]), target, rcond=None)
    residual = target - np.column_stack([np.ones(len(design)), design]) @ coefficients
    return pd.Series(coefficients[1:], index=covariates.columns), float(1 - residual.var() / target.var())


def plot_drivers(stations: pd.DataFrame, names: list[str], labels: dict[str, str], lead: int, output: Path) -> None:
    columns = int(np.ceil(len(names) / 2))
    figure, axes = plt.subplots(2, columns, figsize=(3.4 * columns, 6.6), sharey=True, layout="constrained")
    markers = {"coast_0_25km": "o", "interior_gt25km": "^"}
    for axis, name in zip(axes.ravel(), names, strict=True):
        for group, marker in markers.items():
            subset = stations[stations.coast_stratum == group]
            axis.scatter(subset[name], subset.station_effect_c, s=24, marker=marker, alpha=0.75,
                         color="#0072B2", edgecolors="black", linewidths=0.25, label=group.replace("_", " "))
        rho = stations[name].corr(stations.station_effect_c, method="spearman")
        axis.axhline(0, color="black", linewidth=0.8, linestyle="--")
        axis.set(title=f"ρ = {rho:.2f}", xlabel=labels[name])
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
    for axis in axes[:, 0]:
        axis.set_ylabel("Efecto de referencia por estación (°C)\nGraphCast − Pangu, AVAMET menos ERA5")
    axes[0, 0].legend(fontsize=7, frameon=False, loc="best")
    figure.suptitle(f"Dónde la referencia altera más la evaluación relativa (+{lead} h)", fontsize=11)
    figure.savefig(output.with_suffix(".png"), dpi=300, facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    drivers, bootstrap, paths = cfg["station_effect_drivers"], cfg["bootstrap"], cfg["paths"]
    output = ROOT / paths["station_effect_driver_results_directory"]
    output.mkdir(parents=True, exist_ok=True)
    pair = tuple(drivers["primary_pair"])
    names = list(drivers["covariates"])
    terrain = pd.read_csv(ROOT / paths["station_grid_elevation_csv"])[["station_id", "era5_grid_elevation_m", "delta_z_m", "abs_delta_z_m"]]
    geography = pd.read_csv(ROOT / paths["spatial_stations_csv"])[["station_id", "altitude_m", "coast_distance_km", "coast_stratum", "altitude_stratum"]]
    cells = pd.read_csv(ROOT / paths["station_grid_cells_csv"])[["station_id", "cell_id", "stations_in_cell", "density_stratum"]]
    static = terrain.merge(geography, on="station_id", validate="one_to_one").merge(cells, on="station_id", validate="one_to_one")

    station_rows, correlation_rows, day_rows, regression_rows, matrices = [], [], [], [], []
    for lead in drivers["leads_hours"]:
        source = ROOT / paths["spatial_directory"] / f"lead={lead:03d}" / "batches" / "*.parquet"
        frame = duckdb.sql(f"SELECT * FROM read_parquet('{source.as_posix()}')").fetchdf()
        frame = frame.dropna(subset=["avamet_t2m_qc_c", "era5_t2m_c", *(f"{model}_t2m_c" for model in MODELS)])
        panel = DayPanel(frame, pair)
        effects, varying = panel.effects_and_covariates(None)
        stations = pd.DataFrame({"lead_h": lead, "station_id": panel.stations, "station_effect_c": effects, **varying})
        stations = stations.merge(static, on="station_id", validate="one_to_one")
        stations["n_daily_blocks"] = np.isfinite(panel.matrices["effect"]).sum(axis=1)
        station_rows.append(stations)

        observed = stations[names]
        for name in names:
            values = observed[name].to_numpy(float)
            rho = spearman(effects[:, None], values[:, None])[0]
            rng = np.random.default_rng(bootstrap["seed"] + lead * 100 + sum(map(ord, name)))
            resampled = rng.integers(0, len(effects), size=(bootstrap["n_resamples"], len(effects)))
            station_draws = spearman(effects[resampled].T, values[resampled].T)
            low, high = interval(station_draws, bootstrap["ci"])
            correlation_rows.append({"lead_h": lead, "covariate": name, "label": drivers["covariates"][name],
                                     "n_stations": len(effects), "spearman_rho": rho,
                                     "station_bootstrap_ci_low": float(low), "station_bootstrap_ci_high": float(high),
                                     "p_station_bootstrap_same_sign": float((np.sign(station_draws) == np.sign(rho)).mean())})

        for width in drivers["bootstrap_block_days"]:
            rng = np.random.default_rng(bootstrap["seed"] + lead * 1000 + width)
            weights = day_weights(block_indices(rng, len(panel.days), bootstrap["n_resamples"], width), len(panel.days))
            drawn_effects, drawn_varying = panel.effects_and_covariates(weights)
            for name in names:
                drawn = drawn_varying[name] if name in drawn_varying else np.repeat(observed[name].to_numpy(float)[:, None], weights.shape[1], axis=1)
                draws = spearman(drawn_effects, drawn)
                low, high = interval(draws, bootstrap["ci"])
                day_rows.append({"lead_h": lead, "covariate": name, "block_days": width, "spearman_rho_mean": float(draws.mean()),
                                 "day_bootstrap_ci_low": float(low), "day_bootstrap_ci_high": float(high),
                                 "p_day_bootstrap_same_sign": float((np.sign(draws) == np.sign(draws.mean())).mean())})

        betas, r_squared = standardised_regression(effects, observed)
        for name, beta in betas.items():
            regression_rows.append({"lead_h": lead, "covariate": name, "label": drivers["covariates"][name],
                                    "standardised_beta": float(beta), "model_r_squared": r_squared,
                                    "n_stations": int(observed.notna().all(axis=1).sum())})
        matrix = observed.corr(method="spearman").reset_index(names="covariate")
        matrix.insert(0, "lead_h", lead)
        matrices.append(matrix)
        print(f"analysed drivers lead={lead}; stations={len(effects)}; days={len(panel.days)}", flush=True)

    stations = pd.concat(station_rows, ignore_index=True)
    stations.to_csv(output / "station_effects_and_covariates.csv", index=False)
    correlations = pd.DataFrame(correlation_rows).sort_values(["lead_h", "spearman_rho"])
    correlations.to_csv(output / "driver_correlations.csv", index=False)
    pd.DataFrame(day_rows).sort_values(["lead_h", "covariate", "block_days"]).to_csv(output / "driver_correlations_day_bootstrap.csv", index=False)
    regression = pd.DataFrame(regression_rows).sort_values(["lead_h", "standardised_beta"])
    regression.to_csv(output / "driver_standardised_regression.csv", index=False)
    pd.concat(matrices, ignore_index=True).to_csv(output / "covariate_correlation_matrix.csv", index=False)

    lead = drivers["primary_lead_hours"]
    plot_drivers(stations[stations.lead_h == lead], drivers["figure_covariates"], drivers["covariates"], lead, output / "station_effect_vs_drivers")
    (output / "station_effect_vs_drivers_alt_text.md").write_text(
        f"Ocho paneles de dispersión a +{lead} h. Cada punto es una estación AVAMET; el eje vertical es el cambio de "
        "diferencia MAE GraphCast menos Pangu al pasar de la referencia ERA5 a la observacional, y cada panel lleva una "
        "covariable distinta en el horizontal: RMSE ERA5 − AVAMET, sesgo medio ERA5 − AVAMET, Δz con signo frente a la "
        "orografía ERA5, |Δz|, altitud, temperatura media observada, variabilidad temporal observada y distancia a la costa. "
        "Los paneles del sesgo medio y del Δz con signo muestran una pendiente descendente clara; el de |Δz| no. Círculos: "
        "costa ≤25 km; triángulos: interior >25 km. La línea discontinua marca el efecto nulo y el título de cada panel da "
        "su ρ de Spearman.", encoding="utf-8")
    (output / "station_effect_drivers_summary.json").write_text(json.dumps(
        {"pilot_name": cfg["pilot_name"], "analysis": "descriptive drivers of the station-level reference effect",
         "primary_pair": list(pair), "leads_hours": drivers["leads_hours"], "covariates": names,
         "uncertainty": {"station_bootstrap": "resampling stations with replacement",
                         "day_bootstrap": "circular moving-block bootstrap on ordered valid days; effects and day-dependent covariates are recomputed together",
                         "n_resamples": bootstrap["n_resamples"], "ci": bootstrap["ci"], "seed": bootstrap["seed"]},
         "regression": "descriptive standardised least squares; collinearity is reported separately",
         "status": "completed"}, indent=2), encoding="utf-8")
    print(correlations.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(regression[regression.lead_h == lead].to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
