#!/usr/bin/env python3
"""Why blurring helps against the reanalysis, in which metric, and where.

Three questions the controlled degradation leaves open.

Which metric: benchmarks of this kind are read mostly through RMSE, so the
result has to hold there and not only in MAE. Centred RMSE is carried as a
diagnostic because it removes the mean error and leaves the part that responds
to structure.

Why: the mean squared error splits into a squared bias and a variance term. If
blurring helps against the reanalysis mainly by shrinking the variance term
while the bias barely moves, that is the small-scale mismatch being removed and
nothing else.

Where: the mechanism predicts that blurring should cost the most where there is
subgrid structure for the reference to miss. Stations are stratified by terrain
mismatch and by coast, and the cost of blurring against the observations should
grow with heterogeneity. A flat result would not overturn the main finding; it
would only stop the terrain reading from being pushed too far.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "regional_t2m_2020.yaml"
REFERENCES = {"era5": "era5_t2m_c", "avamet": "avamet_t2m_qc_c"}


def interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    tail = (1 - confidence) / 2
    return tuple(np.quantile(values, [tail, 1 - tail]).tolist())


def block_indices(rng: np.random.Generator, days: int, draws: int, width: int) -> np.ndarray:
    starts = rng.integers(0, days, size=(draws, int(np.ceil(days / width))))
    return ((starts[:, :, None] + np.arange(width)) % days).reshape(draws, -1)[:, :days]


def daily_components(frame: pd.DataFrame, column: str, reference: str) -> pd.DataFrame:
    """Per day: absolute error, squared error and signed error, ready to pool."""
    error = frame[column] - frame[REFERENCES[reference]]
    values = pd.DataFrame({"valid_day": pd.to_datetime(frame.valid_time).dt.date,
                           "absolute": error.abs(), "squared": error**2, "signed": error})
    return values.groupby("valid_day", as_index=False).mean(numeric_only=True).sort_values("valid_day").reset_index(drop=True)


def scores_from(daily: pd.DataFrame, indices: np.ndarray | None) -> dict[str, np.ndarray]:
    """Metrics from pooled daily components, either point estimates or draws."""
    if indices is None:
        absolute, squared, signed = (daily.absolute.mean(), daily.squared.mean(), daily.signed.mean())
    else:
        absolute = daily.absolute.to_numpy()[indices].mean(axis=1)
        squared = daily.squared.to_numpy()[indices].mean(axis=1)
        signed = daily.signed.to_numpy()[indices].mean(axis=1)
    variance = np.maximum(squared - signed**2, 0.0)
    return {"mae": absolute, "rmse": np.sqrt(squared), "centred_rmse": np.sqrt(variance),
            "bias": signed, "bias_squared": signed**2, "error_variance": variance}


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    smoothing, bootstrap, paths = cfg["controlled_smoothing"], cfg["bootstrap"], cfg["paths"]
    output = ROOT / paths["controlled_smoothing_results_directory"]
    output.mkdir(parents=True, exist_ok=True)
    lead, sigmas = smoothing["lead_hours"], [float(sigma) for sigma in smoothing["sigma_km"]]

    source = ROOT / "data" / "interim" / "controlled_smoothing" / f"blurred_stations_lead{lead:03d}.parquet"
    if not source.exists():
        raise FileNotFoundError("run scripts/34_analyse_controlled_smoothing.py first")
    panel = duckdb.sql(f"SELECT * FROM read_parquet('{source.as_posix()}')").fetchdf()
    terrain = pd.read_csv(ROOT / paths["station_grid_elevation_csv"])[["station_id", "abs_delta_z_m"]]
    geography = pd.read_csv(ROOT / paths["spatial_stations_csv"])[["station_id", "coast_stratum"]]
    panel = panel.merge(terrain, on="station_id", validate="many_to_one").merge(geography, on="station_id", validate="many_to_one")
    edges = panel.drop_duplicates("station_id").abs_delta_z_m.quantile([1 / 3, 2 / 3]).to_numpy()
    panel["terrain_stratum"] = np.where(panel.abs_delta_z_m <= edges[0], "delta_z_low",
                                        np.where(panel.abs_delta_z_m <= edges[1], "delta_z_mid", "delta_z_high"))

    metric_rows, decomposition_rows, stratum_rows = [], [], []
    for model, subset in panel.groupby("model", sort=True):
        daily = {sigma: {reference: daily_components(subset, f"sigma_{sigma:g}km", reference) for reference in REFERENCES}
                 for sigma in sigmas}
        for reference in REFERENCES:
            for sigma in sigmas:
                point = scores_from(daily[sigma][reference], None)
                decomposition_rows.append({"model": model, "reference": reference, "sigma_km": sigma,
                                           **{key: float(value) for key, value in point.items()}})
        for width in smoothing["bootstrap_block_days"]:
            rng = np.random.default_rng(bootstrap["seed"] + width + sum(map(ord, model)))
            indices = block_indices(rng, len(daily[sigmas[0]]["era5"]), bootstrap["n_resamples"], width)
            for reference in REFERENCES:
                baseline = scores_from(daily[sigmas[0]][reference], indices)
                for sigma in sigmas[1:]:
                    drawn = scores_from(daily[sigma][reference], indices)
                    for metric in ("mae", "rmse", "centred_rmse", "bias_squared", "error_variance"):
                        change = drawn[metric] - baseline[metric]
                        low, high = interval(change, bootstrap["ci"])
                        metric_rows.append({"model": model, "reference": reference, "metric": metric, "sigma_km": sigma,
                                            "block_days": width, "change": float(change.mean()),
                                            "change_ci_low": low, "change_ci_high": high,
                                            "p_blurring_improves": float((change < 0).mean())})

        for label, column in (("terrain", "terrain_stratum"), ("coast", "coast_stratum")):
            for stratum, part in subset.merge(panel[["station_id", "terrain_stratum"]].drop_duplicates(),
                                              on="station_id", suffixes=("", "_dup")).groupby(column, sort=True):
                local = {sigma: daily_components(part, f"sigma_{sigma:g}km", "avamet") for sigma in sigmas}
                rng = np.random.default_rng(bootstrap["seed"] + sum(map(ord, f"{model}{stratum}")))
                indices = block_indices(rng, len(local[sigmas[0]]), bootstrap["n_resamples"],
                                        smoothing["bootstrap_block_days"][-1])
                baseline = local[sigmas[0]].absolute.to_numpy()[indices].mean(axis=1)
                for sigma in sigmas[1:]:
                    change = local[sigma].absolute.to_numpy()[indices].mean(axis=1) - baseline
                    low, high = interval(change, bootstrap["ci"])
                    stratum_rows.append({"model": model, "stratum_type": label, "stratum": stratum, "sigma_km": sigma,
                                         "n_stations": part.station_id.nunique(), "n_cases": len(part),
                                         "mae_change_c": float(change.mean()), "change_ci_low_c": low,
                                         "change_ci_high_c": high})
        print(f"analysed metrics and terrain for {model}", flush=True)

    metrics = pd.DataFrame(metric_rows).sort_values(["model", "reference", "metric", "sigma_km", "block_days"])
    metrics.to_csv(output / "smoothing_metric_changes.csv", index=False)
    decomposition = pd.DataFrame(decomposition_rows).sort_values(["model", "reference", "sigma_km"])
    decomposition.to_csv(output / "smoothing_decomposition.csv", index=False)
    strata = pd.DataFrame(stratum_rows).sort_values(["model", "stratum_type", "stratum", "sigma_km"])
    strata.to_csv(output / "smoothing_by_terrain.csv", index=False)

    narrow = metrics[(metrics.block_days == smoothing["bootstrap_block_days"][-1]) & (metrics.sigma_km == 12.5)]
    (output / "smoothing_metrics_summary.json").write_text(json.dumps(
        {"pilot_name": cfg["pilot_name"], "analysis": "metric choice, error decomposition and terrain dependence",
         "lead_hours": lead, "sigma_km": sigmas, "metrics": ["mae", "rmse", "centred_rmse"],
         "decomposition": "mean squared error split into squared bias and error variance, pooled over the same days",
         "terrain_strata": "terciles of the absolute station-minus-grid elevation mismatch, plus the frozen coast split",
         "rmse_agrees_with_mae_against_era5": bool(
             (narrow[(narrow.reference == "era5") & (narrow.metric == "rmse")].change.to_numpy() < 0).all()
             and (narrow[(narrow.reference == "era5") & (narrow.metric == "mae")].change.to_numpy() < 0).all()),
         "status": "completed"}, indent=2), encoding="utf-8")
    view = metrics[(metrics.block_days == smoothing["bootstrap_block_days"][-1]) & (metrics.sigma_km.isin([12.5, 25.0]))]
    print(view[view.metric.isin(["mae", "rmse", "centred_rmse"])]
          .to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(decomposition[decomposition.sigma_km <= 25].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(strata[strata.sigma_km == 25.0].to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
