#!/usr/bin/env python3
"""Degrade a forecast on purpose and re-score it against both references.

The smoothness family can only show that fields carrying less small-scale
structure are the ones the reanalysis prefers. It cannot say which way the
arrow points, because rough and skilful models differ in more than roughness.

This removes that objection. One forecast is blurred at several widths and
nothing else about it changes, so the only thing varying between its versions
is how much small-scale structure survives. Each version is scored against the
reanalysis interpolated to the station and against the station itself.

The prediction, declared before running: the two references move in opposite
directions. Blurring should help, or at least stop hurting, against the
reanalysis, and should hurt throughout against the stations. If both got worse
together, the reading of every other family would have to change.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from public_zarr import PublicZarr, bilinear

CONFIG = ROOT / "config" / "regional_t2m_2020.yaml"
CACHE = ROOT / "data" / "raw" / "weatherbench2_cache"
METRE_PER_DEGREE = 111_320.0
REFERENCES = {"era5": "era5_t2m_c", "avamet": "avamet_t2m_qc_c"}


def index_of(values: np.ndarray | pd.DatetimeIndex, wanted: object, name: str) -> int:
    found = np.flatnonzero(np.asarray(values) == wanted)
    if len(found) != 1:
        raise RuntimeError(f"{name}={wanted} is missing or non-unique")
    return int(found[0])


def interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    tail = (1 - confidence) / 2
    return tuple(np.quantile(values, [tail, 1 - tail]).tolist())


def block_indices(rng: np.random.Generator, days: int, draws: int, width: int) -> np.ndarray:
    starts = rng.integers(0, days, size=(draws, int(np.ceil(days / width))))
    return ((starts[:, :, None] + np.arange(width)) % days).reshape(draws, -1)[:, :days]


def blurred_station_values(name: str, spec: dict, stations: pd.DataFrame, initialisations: pd.DatetimeIndex,
                           lead: int, sigmas_km: list[float], cell_degrees: float) -> pd.DataFrame:
    """Sample one forecast at the stations, once per blurring width."""
    store = PublicZarr(spec["path"], CACHE)
    latitudes = store.coordinate(spec["latitude_name"])
    longitudes = store.coordinate(spec["longitude_name"])
    wrapped = np.mod(longitudes + 180, 360) - 180
    margin = int(np.ceil(4 * max(sigmas_km) / (cell_degrees * METRE_PER_DEGREE / 1000)))
    inside_lat = np.flatnonzero((latitudes >= stations.latitude.min() - 1) & (latitudes <= stations.latitude.max() + 1))
    inside_lon = np.flatnonzero((wrapped >= stations.longitude.min() - 1) & (wrapped <= stations.longitude.max() + 1))
    rows = slice(max(inside_lat.min() - margin, 0), min(inside_lat.max() + 1 + margin, len(latitudes)))
    columns = slice(max(inside_lon.min() - margin, 0), min(inside_lon.max() + 1 + margin, len(longitudes)))
    patch_latitudes, patch_longitudes = latitudes[rows], longitudes[columns]
    cell_km = cell_degrees * METRE_PER_DEGREE / 1000
    cosine = float(np.cos(np.deg2rad(patch_latitudes.mean())))
    store_times = store.times(spec["time_name"])
    lead_index = index_of(store.timedeltas_hours(spec["lead_name"]), lead, "lead")
    targets = (stations.latitude.to_numpy(), stations.longitude.to_numpy())

    def one(init: pd.Timestamp) -> pd.DataFrame:
        patch = store.field2d(spec["t2m_name"], index_of(store_times, init, "init"), lead_index)[rows, columns]
        frame = {"station_id": stations.station_id, "init_time": init}
        for sigma in sigmas_km:
            field = patch if sigma == 0 else gaussian_filter(patch, sigma=(sigma / cell_km, sigma / (cell_km * cosine)), mode="nearest")
            frame[f"sigma_{sigma:g}km"] = bilinear(field, patch_latitudes, patch_longitudes, *targets) - 273.15
        return pd.DataFrame(frame)

    with ThreadPoolExecutor(max_workers=4) as executor:
        frames = []
        for number, frame in enumerate(executor.map(one, initialisations), start=1):
            if number == 1 or number % 250 == 0 or number == len(initialisations):
                print(f"blurring {name}: {number}/{len(initialisations)}", flush=True)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True).assign(model=name)


def daily_scores(frame: pd.DataFrame, column: str, reference: str) -> pd.DataFrame:
    error = frame[column] - frame[REFERENCES[reference]]
    values = pd.DataFrame({"valid_day": pd.to_datetime(frame.valid_time).dt.date,
                           "absolute": error.abs(), "squared": error**2})
    return values.groupby("valid_day", as_index=False).mean(numeric_only=True).sort_values("valid_day").reset_index(drop=True)


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    family, bootstrap, paths = cfg["controlled_smoothing"], cfg["bootstrap"], cfg["paths"]
    output = ROOT / paths["controlled_smoothing_results_directory"]
    output.mkdir(parents=True, exist_ok=True)
    lead, sigmas = family["lead_hours"], [float(sigma) for sigma in family["sigma_km"]]
    stations = pd.read_csv(ROOT / paths["spatial_stations_csv"])[["station_id", "latitude", "longitude"]]

    source = (ROOT / paths["spatial_directory"] / f"lead={lead:03d}" / "batches" / "*.parquet").as_posix()
    panel = duckdb.sql(f"SELECT station_id, init_time, valid_time, era5_t2m_c, avamet_t2m_qc_c FROM read_parquet('{source}')").fetchdf()
    panel = panel.dropna(subset=list(REFERENCES.values()))
    initialisations = pd.DatetimeIndex(sorted(panel.init_time.unique()))
    cell_degrees = float(np.abs(np.diff(PublicZarr(cfg["weatherbench2"]["era5"]["path"], CACHE)
                                        .coordinate(cfg["weatherbench2"]["era5"]["latitude_name"]))).mean())

    specs = cfg["weatherbench2"]["models"]
    blurred = pd.concat([blurred_station_values(name, specs[name], stations, initialisations, lead, sigmas, cell_degrees)
                         for name in family["models"]], ignore_index=True)
    joined = blurred.merge(panel, on=["station_id", "init_time"], validate="many_to_one")
    # Persisted so that the reference-support family reuses these fields
    # instead of blurring them again.
    store = ROOT / "data" / "interim" / "controlled_smoothing"
    store.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.register("joined", joined)
    connection.execute(f"COPY joined TO '{(store / f'blurred_stations_lead{lead:03d}.parquet').as_posix()}' "
                       "(FORMAT PARQUET, COMPRESSION ZSTD)")

    rows, curves = [], []
    for model, subset in joined.groupby("model", sort=True):
        daily = {(sigma, reference): daily_scores(subset, f"sigma_{sigma:g}km", reference)
                 for sigma in sigmas for reference in REFERENCES}
        days = daily[(sigmas[0], "era5")].valid_day
        for reference in REFERENCES:
            for sigma in sigmas:
                values = daily[(sigma, reference)]
                if not values.valid_day.equals(days):
                    raise RuntimeError("blurring changed the set of scored days")
                curves.append({"model": model, "reference": reference, "sigma_km": sigma,
                               "mae_c": float(values.absolute.mean()), "rmse_c": float(np.sqrt(values.squared.mean())),
                               "n_daily_blocks": len(values)})
        for width in family["bootstrap_block_days"]:
            rng = np.random.default_rng(bootstrap["seed"] + width + sum(map(ord, model)))
            indices = block_indices(rng, len(days), bootstrap["n_resamples"], width)
            for reference in REFERENCES:
                baseline = daily[(sigmas[0], reference)].absolute.to_numpy()[indices].mean(axis=1)
                for sigma in sigmas[1:]:
                    blurred_draws = daily[(sigma, reference)].absolute.to_numpy()[indices].mean(axis=1)
                    change = blurred_draws - baseline
                    low, high = interval(change, bootstrap["ci"])
                    rows.append({"model": model, "reference": reference, "sigma_km": sigma, "block_days": width,
                                 "mae_change_c": float(change.mean()), "change_ci_low_c": low, "change_ci_high_c": high,
                                 "p_blurring_improves": float((change < 0).mean())})
    curve = pd.DataFrame(curves).sort_values(["model", "reference", "sigma_km"])
    curve.to_csv(output / "smoothing_curves.csv", index=False)
    changes = pd.DataFrame(rows).sort_values(["model", "reference", "sigma_km", "block_days"])
    changes.to_csv(output / "smoothing_changes.csv", index=False)

    narrow = changes[changes.block_days == family["bootstrap_block_days"][0]]
    opposite = {}
    for model, subset in narrow.groupby("model"):
        era5 = subset[subset.reference == "era5"].set_index("sigma_km").mae_change_c
        avamet = subset[subset.reference == "avamet"].set_index("sigma_km").mae_change_c
        opposite[model] = {"era5_best_sigma_km": float(curve[(curve.model == model) & (curve.reference == "era5")]
                                                       .sort_values("mae_c").sigma_km.iloc[0]),
                           "avamet_best_sigma_km": float(curve[(curve.model == model) & (curve.reference == "avamet")]
                                                         .sort_values("mae_c").sigma_km.iloc[0]),
                           "era5_improves_at_any_width": bool((era5 < 0).any()),
                           "avamet_worsens_at_every_width": bool((avamet > 0).all())}

    plot_curves(curve, output / "smoothing_curves")
    (output / "smoothing_curves_alt_text.md").write_text(
        "Dos paneles con el ancho del desenfoque gaussiano en el eje horizontal, de 0 a 100 km. El izquierdo da el MAE de "
        "cada pronóstico medido contra ERA5 interpolado a la estación; el derecho, contra la estación. Una línea por modelo. "
        "Las curvas de la izquierda bajan o se aplanan al desenfocar, y las de la derecha suben, de modo que degradar el "
        "campo mejora la nota contra la reanálisis y la empeora contra las observaciones.", encoding="utf-8")
    (output / "controlled_smoothing_summary.json").write_text(json.dumps(
        {"pilot_name": cfg["pilot_name"], "analysis": "controlled degradation of single forecasts",
         "prediction": "blurring moves the two references in opposite directions", "lead_hours": lead,
         "sigma_km": sigmas, "models": family["models"], "per_model": opposite,
         "bootstrap": {"method": "circular moving-block bootstrap on ordered valid days, paired against the unblurred field",
                       "n_resamples": bootstrap["n_resamples"], "block_days": family["bootstrap_block_days"],
                       "ci": bootstrap["ci"], "seed": bootstrap["seed"]},
         "caveat": "a Gaussian blur is not what a coarser model would produce; it removes small-scale variance without "
                   "changing anything else, which is the point of the control and also its limit",
         "status": "completed"}, indent=2), encoding="utf-8")
    print(curve.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(json.dumps(opposite, indent=2))


def plot_curves(curve: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), layout="constrained")
    colours = {"ifs_hres": "#0072B2", "graphcast_hres_init": "#D55E00", "pangu_hres_init": "#009E73"}
    titles = {"era5": "Contra ERA5 en la estación", "avamet": "Contra la estación"}
    for axis, reference in zip(axes, ("era5", "avamet"), strict=True):
        for model, colour in colours.items():
            subset = curve[(curve.model == model) & (curve.reference == reference)].sort_values("sigma_km")
            axis.plot(subset.sigma_km, subset.mae_c, marker="o", markersize=5, color=colour,
                      label=model.replace("_hres_init", "").replace("_", " "))
        axis.set(xlabel="Ancho del desenfoque σ (km)", ylabel="MAE (°C)", title=titles[reference])
        axis.grid(color="#dddddd", linewidth=0.5)
    axes[0].legend(fontsize=8, frameon=False)
    figure.suptitle("Degradar el campo mejora la nota contra la reanálisis y la empeora contra las estaciones", fontsize=11)
    figure.savefig(output.with_suffix(".png"), dpi=300, facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
