#!/usr/bin/env python3
"""Does the reference's own spatial support set which forecast structure wins?

The controlled degradation shows that blurring a forecast improves its score
against the reanalysis and not against a station. The obvious objection is that
this is support mismatch: a station is a point, the reanalysis is an areal
quantity, so of course the point prefers the unblurred field.

That objection is a hypothesis, and AVAMET is dense enough to test it. Building
observational references of growing spatial support and repeating the blurring
against each one turns the reference scale into the thing that varies.

Prediction declared before running: the optimal blur grows with the support of
the reference.

Three controls guard the reading, because a wider radius also holds more
stations, and more observations average away noise on their own:

* a uniform mean inside the radius;
* a distance-weighted mean, which changes the shape of the aggregation;
* a thinning that keeps the station count fixed at every radius and only lets
  the area grow, repeated many times to give the optimum a distribution.

The centres are the same throughout, and so is the forecast sampling.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "regional_t2m_2020.yaml"


def interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    tail = (1 - confidence) / 2
    return tuple(np.quantile(values, [tail, 1 - tail]).tolist())


def block_indices(rng: np.random.Generator, days: int, draws: int, width: int) -> np.ndarray:
    starts = rng.integers(0, days, size=(draws, int(np.ceil(days / width))))
    return ((starts[:, :, None] + np.arange(width)) % days).reshape(draws, -1)[:, :days]


def distances_km(stations: pd.DataFrame, projection: str) -> np.ndarray:
    to_metres = Transformer.from_crs("EPSG:4326", projection, always_xy=True).transform
    eastings, northings = to_metres(stations.longitude.to_numpy(), stations.latitude.to_numpy())
    coordinates = np.column_stack([eastings, northings]) / 1000.0
    return np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)


def weight_matrix(distance: np.ndarray, radius: float, weighting: str) -> np.ndarray:
    """Rows are centres, columns are stations. Zero outside the radius."""
    if radius == 0:
        return np.eye(len(distance))
    inside = distance <= radius
    if weighting == "uniform":
        return inside.astype(float)
    if weighting == "gaussian":
        return np.where(inside, np.exp(-(distance**2) / (2 * (radius / 2) ** 2)), 0.0)
    raise ValueError(f"unknown weighting: {weighting}")


def aggregate(values: np.ndarray, present: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted mean over each centre's neighbourhood, missing values skipped."""
    totals = np.where(present, values, 0.0) @ weights.T
    counts = present.astype(float) @ weights.T
    return np.where(counts > 0, totals / np.maximum(counts, 1e-12), np.nan)


def wide_matrices(subset: pd.DataFrame, stations: pd.DataFrame, columns: list[str]) -> tuple[dict, pd.DatetimeIndex]:
    frames, index = {}, None
    for column in columns:
        wide = subset.pivot_table(index="valid_time", columns="station_id", values=column, aggfunc="mean")
        wide = wide.reindex(columns=stations.station_id)
        frames[column] = wide.to_numpy()
        index = wide.index
    assert index is not None
    return frames, index


def optimum_of(errors: dict[float, np.ndarray], sigmas: list[float]) -> float:
    means = np.array([np.nanmean(errors[sigma]) for sigma in sigmas])
    return float(np.array(sigmas)[means.argmin()])


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    family, smoothing, bootstrap, paths = (cfg["reference_support_ladder"], cfg["controlled_smoothing"],
                                           cfg["bootstrap"], cfg["paths"])
    output = ROOT / paths["reference_support_results_directory"]
    output.mkdir(parents=True, exist_ok=True)
    lead, sigmas = smoothing["lead_hours"], [float(sigma) for sigma in smoothing["sigma_km"]]
    radii = [float(radius) for radius in family["support_radii_km"]]

    source = ROOT / "data" / "interim" / "controlled_smoothing" / f"blurred_stations_lead{lead:03d}.parquet"
    if not source.exists():
        raise FileNotFoundError("run scripts/34_analyse_controlled_smoothing.py first; it persists the blurred fields")
    panel = duckdb.sql(f"SELECT * FROM read_parquet('{source.as_posix()}')").fetchdf()
    stations = pd.read_csv(ROOT / paths["spatial_stations_csv"])[["station_id", "latitude", "longitude"]].reset_index(drop=True)
    distance = distances_km(stations, family["projection"])

    inside_smallest = distance <= min(radius for radius in radii if radius > 0)
    eligible = inside_smallest.sum(axis=1) >= family["min_stations_per_support"]
    centres = np.flatnonzero(eligible)
    if len(centres) < 10:
        raise RuntimeError(f"only {len(centres)} centres meet the density requirement")

    sigma_columns = [f"sigma_{sigma:g}km" for sigma in sigmas]
    curves, changes, thinned_rows = [], [], []
    for model, subset in panel.groupby("model", sort=True):
        matrices, index = wide_matrices(subset, stations, [*sigma_columns, "avamet_t2m_qc_c", "era5_t2m_c"])
        days = pd.to_datetime(index).date
        present = {name: np.isfinite(values) for name, values in matrices.items()}
        forecast = {sigma: matrices[f"sigma_{sigma:g}km"][:, centres] for sigma in sigmas}

        targets = {}
        for weighting in family["weightings"]:
            for radius in radii:
                weights = weight_matrix(distance, radius, weighting)[centres]
                targets[(weighting, radius)] = aggregate(matrices["avamet_t2m_qc_c"], present["avamet_t2m_qc_c"], weights)
            if weighting == "uniform":
                point = weight_matrix(distance, 0.0, weighting)[centres]
                targets[("uniform", -1.0)] = aggregate(matrices["era5_t2m_c"], present["era5_t2m_c"], point)

        for (weighting, radius), target in targets.items():
            label = "era5_point" if radius < 0 else f"avamet_{radius:g}km"
            daily = {}
            for sigma in sigmas:
                error = np.abs(forecast[sigma] - target)
                frame = pd.DataFrame({"valid_day": days, "mae": np.nanmean(error, axis=1)})
                daily[sigma] = frame.groupby("valid_day", as_index=False).mae.mean().sort_values("valid_day").reset_index(drop=True)
                curves.append({"model": model, "weighting": weighting, "reference": label, "support_km": radius,
                               "sigma_km": sigma, "mae_c": float(daily[sigma].mae.mean()), "n_centres": len(centres),
                               "n_daily_blocks": len(daily[sigma])})
            for width in family["bootstrap_block_days"]:
                rng = np.random.default_rng(bootstrap["seed"] + width + sum(map(ord, f"{model}{weighting}{label}")))
                indices = block_indices(rng, len(daily[sigmas[0]]), bootstrap["n_resamples"], width)
                stacked = np.stack([daily[sigma].mae.to_numpy()[indices].mean(axis=1) for sigma in sigmas], axis=1)
                best = np.array(sigmas)[stacked.argmin(axis=1)]
                low, high = interval(best, bootstrap["ci"])
                changes.append({"model": model, "weighting": weighting, "reference": label, "support_km": radius,
                                "block_days": width,
                                "optimal_sigma_km_point": float(np.array(sigmas)[stacked.mean(axis=0).argmin()]),
                                "optimal_sigma_km_bootstrap_mean": float(best.mean()),
                                "optimal_sigma_ci_low_km": low, "optimal_sigma_ci_high_km": high,
                                "p_optimal_sigma_gt_zero": float((best > 0).mean())})

        # Thinning: the neighbourhood keeps a fixed number of stations at every
        # radius, so the count is held constant and only the area grows.
        keep = family["thinning_stations"]
        for draw in range(family["thinning_draws"]):
            rng = np.random.default_rng(bootstrap["seed"] + draw * 97 + sum(map(ord, model)))
            for radius in radii:
                if radius == 0:
                    thinned = np.eye(len(distance))[centres]
                else:
                    thinned = np.zeros((len(centres), len(distance)))
                    for position, centre in enumerate(centres):
                        candidates = np.flatnonzero(distance[centre] <= radius)
                        chosen = rng.choice(candidates, size=min(keep, len(candidates)), replace=False)
                        thinned[position, chosen] = 1.0
                target = aggregate(matrices["avamet_t2m_qc_c"], present["avamet_t2m_qc_c"], thinned)
                errors = {sigma: np.abs(forecast[sigma] - target) for sigma in sigmas}
                thinned_rows.append({"model": model, "draw": draw, "support_km": radius,
                                     "stations_kept": keep, "optimal_sigma_km": optimum_of(errors, sigmas)})
        print(f"analysed support ladder for {model}; centres={len(centres)}", flush=True)

    curve = pd.DataFrame(curves).sort_values(["model", "weighting", "support_km", "sigma_km"])
    curve.to_csv(output / "support_ladder_curves.csv", index=False)
    optima = pd.DataFrame(changes).sort_values(["model", "weighting", "support_km", "block_days"])
    optima.to_csv(output / "support_ladder_optima.csv", index=False)
    thinned = pd.DataFrame(thinned_rows)
    thinned.to_csv(output / "support_ladder_thinning.csv", index=False)

    thinning_summary = thinned.groupby(["model", "support_km"], as_index=False).optimal_sigma_km.agg(
        ["mean", "median", lambda values: float((values > 0).mean())])
    thinning_summary.columns = ["model", "support_km", "mean_optimal_sigma_km", "median_optimal_sigma_km",
                                "fraction_of_draws_gt_zero"]
    thinning_summary.to_csv(output / "support_ladder_thinning_summary.csv", index=False)

    ladder = optima[(optima.support_km >= 0) & (optima.block_days == family["bootstrap_block_days"][-1])]
    correlations = {}
    for weighting, group in ladder.groupby("weighting"):
        correlations[weighting] = float(group.groupby("model").apply(
            lambda frame: frame.support_km.corr(frame.optimal_sigma_km_point, method="spearman"),
            include_groups=False).mean())
    correlations["thinned"] = float(thinning_summary.groupby("model").apply(
        lambda frame: frame.support_km.corr(frame.mean_optimal_sigma_km, method="spearman"),
        include_groups=False).mean())

    plot_ladder(curve, ladder, thinning_summary, output / "reference_support_ladder")
    (output / "reference_support_ladder_alt_text.md").write_text(
        "Tres paneles. El izquierdo traza el MAE relativo frente al ancho del desenfoque del pronóstico, con una curva por "
        "referencia observacional —estación suelta y medias en 25, 50 y 100 km— más ERA5 en el punto; el mínimo se desplaza "
        "hacia desenfoques mayores conforme crece el soporte. El central resume ese desplazamiento con media uniforme y con "
        "ponderación gaussiana. El derecho repite el resumen manteniendo fijo el número de estaciones por centro y dejando "
        "crecer sólo el área, con la dispersión entre repeticiones del muestreo.", encoding="utf-8")
    (output / "reference_support_summary.json").write_text(json.dumps(
        {"pilot_name": cfg["pilot_name"], "analysis": "reference spatial support versus optimal forecast blur",
         "prediction": "the optimal blur grows with the support of the reference",
         "lead_hours": lead, "support_radii_km": radii, "sigma_km": sigmas, "n_centres": int(len(centres)),
         "weightings": family["weightings"], "thinning_stations": family["thinning_stations"],
         "thinning_draws": family["thinning_draws"],
         "mean_spearman_support_versus_optimal_sigma": correlations,
         "controls_note": "the thinned ladder holds the station count fixed, so a climbing optimum there is due to the "
                          "area covered and not to the number of observations averaged",
         "caveat": "an average of stations inside a radius is not a grid-cell average: it is irregularly sampled and "
                   "weights the places where the network is dense; and the optimum is located on a discrete sigma grid",
         "status": "completed"}, indent=2), encoding="utf-8")
    print(ladder[["model", "weighting", "support_km", "optimal_sigma_km_point", "optimal_sigma_km_bootstrap_mean",
                  "optimal_sigma_ci_low_km", "optimal_sigma_ci_high_km"]].to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(thinning_summary.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(json.dumps(correlations, indent=2))


def plot_ladder(curve: pd.DataFrame, ladder: pd.DataFrame, thinning: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.4, 4.6), layout="constrained")
    subset = curve[(curve.model == "ifs_hres") & (curve.weighting == "uniform")]
    colours = {0.0: "#0072B2", 25.0: "#56B4E9", 50.0: "#009E73", 100.0: "#E69F00", -1.0: "#999999"}
    labels = {0.0: "estación suelta", 25.0: "media 25 km", 50.0: "media 50 km", 100.0: "media 100 km",
              -1.0: "ERA5 en el punto"}
    for support, colour in colours.items():
        line = subset[subset.support_km == support].sort_values("sigma_km")
        if line.empty:
            continue
        axes[0].plot(line.sigma_km, line.mae_c / line.mae_c.iloc[0], marker="o", markersize=4, color=colour,
                     linestyle="--" if support < 0 else "-", label=labels[support])
    axes[0].axhline(1, color="black", linewidth=0.7, linestyle=":")
    axes[0].set(xlabel="Desenfoque del pronóstico σ (km)", ylabel="MAE relativo al campo sin desenfocar",
                title="IFS-HRES según el soporte de la referencia")
    axes[0].legend(fontsize=7, frameon=False)
    styles = {"uniform": "-", "gaussian": "--"}
    for (model, weighting), group in ladder.groupby(["model", "weighting"]):
        group = group.sort_values("support_km")
        axes[1].plot(group.support_km, group.optimal_sigma_km_bootstrap_mean, marker="o", markersize=5,
                     linestyle=styles[weighting], label=f"{model.split('_')[0]} · {weighting}")
    axes[1].set(xlabel="Radio de agregación (km)", ylabel="Desenfoque óptimo σ* (km)",
                title="Media uniforme frente a ponderada")
    axes[1].legend(fontsize=7, frameon=False, ncols=2)
    for model, group in thinning.groupby("model"):
        group = group.sort_values("support_km")
        axes[2].plot(group.support_km, group.mean_optimal_sigma_km, marker="o", markersize=5,
                     label=model.split("_")[0])
    axes[2].set(xlabel="Radio de agregación (km)", ylabel="Desenfoque óptimo σ* medio (km)",
                title="Con número de estaciones fijo")
    axes[2].legend(fontsize=7, frameon=False)
    for axis in axes:
        axis.grid(color="#dddddd", linewidth=0.5)
    figure.savefig(output.with_suffix(".png"), dpi=300, facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
