#!/usr/bin/env python3
"""Repeat the two central mechanism results in the other available years.

The whole causal argument rests on 2020, and the first question a reader will
ask is whether that year happened to suit it. This repeats the controlled
degradation and the reference-support ladder in 2021 and 2022 — both available
years, so none can be chosen for its outcome — with every parameter as frozen
and with the design declared in docs/PREREGISTRO_FASE_GLOBAL.md beforehand.

GraphCast-HRES-init is not published for those years and is not substituted.
Both results are per model rather than per pair, so its absence shrinks the
sample without changing what is being asked.

The fields these years need were already fetched by the temporal replication,
and a fetched chunk is a whole global field, so nothing is downloaded here.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml
from pyproj import Transformer
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from public_zarr import PublicZarr, bilinear

CONFIG = ROOT / "config" / "regional_t2m_2020.yaml"
CACHE = ROOT / "data" / "raw" / "weatherbench2_cache"
METRE_PER_DEGREE = 111_320.0


def index_of(values, wanted, name: str) -> int:
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


def model_specs(cfg: dict, year: int) -> dict[str, dict]:
    family = cfg["mechanism_replication"]
    specs = {}
    for name in family["models"]:
        spec = dict(cfg["weatherbench2"]["models"][name])
        if name == "pangu_hres_init":
            spec["path"] = family["pangu_path_template"].format(year=year)
        specs[name] = spec
    return specs


def blurred_values(name: str, spec: dict, stations: pd.DataFrame, inits: pd.DatetimeIndex, lead: int,
                   sigmas: list[float], cell_degrees: float) -> pd.DataFrame:
    store = PublicZarr(spec["path"], CACHE)
    latitudes, longitudes = store.coordinate(spec["latitude_name"]), store.coordinate(spec["longitude_name"])
    wrapped = np.mod(longitudes + 180, 360) - 180
    cell_km = cell_degrees * METRE_PER_DEGREE / 1000
    margin = int(np.ceil(4 * max(sigmas) / cell_km))
    rows_in = np.flatnonzero((latitudes >= stations.latitude.min() - 1) & (latitudes <= stations.latitude.max() + 1))
    cols_in = np.flatnonzero((wrapped >= stations.longitude.min() - 1) & (wrapped <= stations.longitude.max() + 1))
    rows = slice(max(rows_in.min() - margin, 0), min(rows_in.max() + 1 + margin, len(latitudes)))
    columns = slice(max(cols_in.min() - margin, 0), min(cols_in.max() + 1 + margin, len(longitudes)))
    patch_lat, patch_lon = latitudes[rows], longitudes[columns]
    cosine = float(np.cos(np.deg2rad(patch_lat.mean())))
    times = store.times(spec["time_name"])
    lead_index = index_of(store.timedeltas_hours(spec["lead_name"]), lead, "lead")
    targets = (stations.latitude.to_numpy(), stations.longitude.to_numpy())

    def one(init: pd.Timestamp) -> pd.DataFrame:
        patch = store.field2d(spec["t2m_name"], index_of(times, init, "init"), lead_index)[rows, columns]
        frame = {"station_id": stations.station_id, "init_time": init}
        for sigma in sigmas:
            field = patch if sigma == 0 else gaussian_filter(patch, sigma=(sigma / cell_km, sigma / (cell_km * cosine)), mode="nearest")
            frame[f"sigma_{sigma:g}km"] = bilinear(field, patch_lat, patch_lon, *targets) - 273.15
        return pd.DataFrame(frame)

    with ThreadPoolExecutor(max_workers=4) as executor:
        frames = []
        for number, frame in enumerate(executor.map(one, inits), start=1):
            if number == 1 or number % 250 == 0 or number == len(inits):
                print(f"blurring {name}: {number}/{len(inits)}", flush=True)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True).assign(model=name)


def reference_values(cfg: dict, stations: pd.DataFrame, valid_times: pd.DatetimeIndex) -> pd.DataFrame:
    spec = cfg["weatherbench2"]["era5"]
    store = PublicZarr(spec["path"], CACHE)
    latitudes, longitudes = store.coordinate(spec["latitude_name"]), store.coordinate(spec["longitude_name"])
    times = store.times(spec["time_name"])
    targets = (stations.latitude.to_numpy(), stations.longitude.to_numpy())

    def one(valid: pd.Timestamp) -> pd.DataFrame:
        grid = store.field2d(spec["t2m_name"], index_of(times, valid, "valid_time"))
        return pd.DataFrame({"station_id": stations.station_id, "valid_time": valid,
                             "era5_t2m_c": bilinear(grid, latitudes, longitudes, *targets) - 273.15})

    with ThreadPoolExecutor(max_workers=4) as executor:
        frames = list(executor.map(one, valid_times))
    return pd.concat(frames, ignore_index=True)


def observed_values(cfg: dict, stations: pd.DataFrame, valid_times: pd.DatetimeIndex, year: int) -> pd.DataFrame:
    glob = cfg["mechanism_replication"]["avamet_archive_glob"].format(year=year)
    archive = (ROOT / glob).resolve().as_posix()
    connection = duckdb.connect()
    connection.execute("SET TimeZone='UTC'")
    connection.register("wanted", stations[["station_id"]])
    values = connection.execute(
        f"""SELECT a.station_id, a.observed_utc, a.temperature_c
              FROM read_parquet('{archive}', hive_partitioning=true) AS a
              INNER JOIN wanted USING (station_id)
             ORDER BY a.station_id, a.observed_utc"""
    ).fetchdf()
    low, high = cfg["avamet"]["t2m_plausible_c"]
    qc = cfg["avamet"]["qc"]
    physical = values.temperature_c.between(low, high)
    masked = values.temperature_c.where(physical)
    median = masked.groupby(values.station_id).transform(
        lambda item: item.rolling(qc["rolling_window_reports"], center=True, min_periods=3).median())
    values["avamet_t2m_qc_c"] = values.temperature_c.where(
        physical & ((masked - median).abs() <= qc["local_median_deviation_c"]))
    wanted = pd.DataFrame({"observed_utc": valid_times.tz_localize("UTC")})
    values = values.merge(wanted, on="observed_utc", how="inner", validate="many_to_one")
    values["valid_time"] = pd.to_datetime(values.observed_utc, utc=True).dt.tz_localize(None)
    return values.dropna(subset=["avamet_t2m_qc_c"])[["station_id", "valid_time", "avamet_t2m_qc_c"]]


def smoothing_rows(panel: pd.DataFrame, sigmas: list[float], cfg: dict, year: int) -> tuple[list[dict], list[dict]]:
    bootstrap, widths = cfg["bootstrap"], cfg["controlled_smoothing"]["bootstrap_block_days"]
    curves, changes = [], []
    for model, subset in panel.groupby("model", sort=True):
        daily = {}
        for reference, column in (("era5", "era5_t2m_c"), ("avamet", "avamet_t2m_qc_c")):
            for sigma in sigmas:
                error = (subset[f"sigma_{sigma:g}km"] - subset[column]).abs()
                frame = pd.DataFrame({"valid_day": pd.to_datetime(subset.valid_time).dt.date, "mae": error})
                daily[(reference, sigma)] = frame.groupby("valid_day", as_index=False).mae.mean().sort_values("valid_day").reset_index(drop=True)
                curves.append({"year": year, "model": model, "reference": reference, "sigma_km": sigma,
                               "mae_c": float(daily[(reference, sigma)].mae.mean()),
                               "n_daily_blocks": len(daily[(reference, sigma)]), "n_cases": len(subset)})
        for width in widths:
            rng = np.random.default_rng(bootstrap["seed"] + width + year + sum(map(ord, model)))
            indices = block_indices(rng, len(daily[("era5", sigmas[0])]), bootstrap["n_resamples"], width)
            for reference in ("era5", "avamet"):
                baseline = daily[(reference, sigmas[0])].mae.to_numpy()[indices].mean(axis=1)
                for sigma in sigmas[1:]:
                    change = daily[(reference, sigma)].mae.to_numpy()[indices].mean(axis=1) - baseline
                    low, high = interval(change, bootstrap["ci"])
                    changes.append({"year": year, "model": model, "reference": reference, "sigma_km": sigma,
                                    "block_days": width, "mae_change_c": float(change.mean()),
                                    "change_ci_low_c": low, "change_ci_high_c": high,
                                    "p_blurring_improves": float((change < 0).mean())})
    return curves, changes


def ladder_rows(panel: pd.DataFrame, stations: pd.DataFrame, sigmas: list[float], cfg: dict, year: int) -> list[dict]:
    family, bootstrap = cfg["reference_support_ladder"], cfg["bootstrap"]
    radii = [float(radius) for radius in family["support_radii_km"]]
    to_metres = Transformer.from_crs("EPSG:4326", family["projection"], always_xy=True).transform
    eastings, northings = to_metres(stations.longitude.to_numpy(), stations.latitude.to_numpy())
    distance = np.linalg.norm((np.column_stack([eastings, northings]) / 1000.0)[:, None, :]
                              - (np.column_stack([eastings, northings]) / 1000.0)[None, :, :], axis=2)
    rows = []
    for model, subset in panel.groupby("model", sort=True):
        wide = {name: subset.pivot_table(index="valid_time", columns="station_id", values=name, aggfunc="mean")
                .reindex(columns=stations.station_id).to_numpy()
                for name in [*(f"sigma_{sigma:g}km" for sigma in sigmas), "avamet_t2m_qc_c"]}
        index = subset.pivot_table(index="valid_time", columns="station_id", values="avamet_t2m_qc_c", aggfunc="mean").index
        days = pd.to_datetime(index).date
        observed, present = wide["avamet_t2m_qc_c"], np.isfinite(wide["avamet_t2m_qc_c"])
        available = present.any(axis=0)
        inside_smallest = (distance <= min(r for r in radii if r > 0)) & available[None, :]
        centres = np.flatnonzero(inside_smallest.sum(axis=1) >= family["min_stations_per_support"])
        for radius in radii:
            weights = (np.eye(len(distance)) if radius == 0 else (distance <= radius).astype(float))[centres]
            totals = np.where(present, observed, 0.0) @ weights.T
            counts = present.astype(float) @ weights.T
            target = np.where(counts > 0, totals / np.maximum(counts, 1e-12), np.nan)
            daily = {}
            for sigma in sigmas:
                error = np.abs(wide[f"sigma_{sigma:g}km"][:, centres] - target)
                daily[sigma] = pd.DataFrame({"valid_day": days, "mae": np.nanmean(error, axis=1)}) \
                    .groupby("valid_day", as_index=False).mae.mean().sort_values("valid_day").reset_index(drop=True)
            rng = np.random.default_rng(bootstrap["seed"] + year + int(radius) + sum(map(ord, model)))
            indices = block_indices(rng, len(daily[sigmas[0]]), bootstrap["n_resamples"], family["bootstrap_block_days"][-1])
            stacked = np.stack([daily[sigma].mae.to_numpy()[indices].mean(axis=1) for sigma in sigmas], axis=1)
            best = np.array(sigmas)[stacked.argmin(axis=1)]
            low, high = interval(best, bootstrap["ci"])
            rows.append({"year": year, "model": model, "support_km": radius, "n_centres": len(centres),
                         "optimal_sigma_km_point": float(np.array(sigmas)[stacked.mean(axis=0).argmin()]),
                         "optimal_sigma_km_bootstrap_mean": float(best.mean()),
                         "optimal_sigma_ci_low_km": low, "optimal_sigma_ci_high_km": high})
            if radius > 0:
                keep = family["thinning_stations"]
                optima = []
                for draw in range(family["thinning_draws"]):
                    generator = np.random.default_rng(bootstrap["seed"] + draw * 97 + year)
                    thinned = np.zeros((len(centres), len(distance)))
                    for position, centre in enumerate(centres):
                        candidates = np.flatnonzero((distance[centre] <= radius) & available)
                        chosen = generator.choice(candidates, size=min(keep, len(candidates)), replace=False)
                        thinned[position, chosen] = 1.0
                    totals = np.where(present, observed, 0.0) @ thinned.T
                    counts = present.astype(float) @ thinned.T
                    local = np.where(counts > 0, totals / np.maximum(counts, 1e-12), np.nan)
                    means = [np.nanmean(np.abs(wide[f"sigma_{sigma:g}km"][:, centres] - local)) for sigma in sigmas]
                    optima.append(float(np.array(sigmas)[int(np.argmin(means))]))
                rows[-1]["thinned_optimal_sigma_km_mean"] = float(np.mean(optima))
                rows[-1]["thinned_fraction_gt_zero"] = float(np.mean(np.array(optima) > 0))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("year", type=int)
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    family, paths = cfg["mechanism_replication"], cfg["paths"]
    if args.year not in family["years"]:
        raise ValueError("year is outside the sealed replication")
    output = ROOT / paths["mechanism_replication_results_directory"] / f"year={args.year}"
    output.mkdir(parents=True, exist_ok=True)
    lead = family["lead_hours"]
    sigmas = [float(sigma) for sigma in cfg["controlled_smoothing"]["sigma_km"]]
    stations = pd.read_csv(ROOT / paths["spatial_stations_csv"])[["station_id", "latitude", "longitude"]].reset_index(drop=True)

    specs = model_specs(cfg, args.year)
    common = None
    for name, spec in specs.items():
        store = PublicZarr(spec["path"], CACHE)
        times = store.times(spec["time_name"])
        chosen = times[(times >= pd.Timestamp(f"{args.year}-01-01")) & (times <= pd.Timestamp(f"{args.year}-12-31T12:00"))]
        chosen = chosen[chosen.hour.isin(cfg["period"]["cycles_utc"])]
        common = chosen if common is None else common.intersection(chosen)
    assert common is not None
    inits = common.sort_values()
    valid_times = inits + pd.Timedelta(hours=lead)
    cell_degrees = float(np.abs(np.diff(PublicZarr(cfg["weatherbench2"]["era5"]["path"], CACHE)
                                        .coordinate(cfg["weatherbench2"]["era5"]["latitude_name"]))).mean())
    print(f"starting replication year={args.year}; inits={len(inits)}; stations={len(stations)}", flush=True)

    blurred = pd.concat([blurred_values(name, spec, stations, inits, lead, sigmas, cell_degrees)
                         for name, spec in specs.items()], ignore_index=True)
    blurred["valid_time"] = blurred.init_time + pd.Timedelta(hours=lead)
    panel = blurred.merge(reference_values(cfg, stations, valid_times), on=["station_id", "valid_time"], validate="many_to_one")
    panel = panel.merge(observed_values(cfg, stations, valid_times, args.year), on=["station_id", "valid_time"], validate="many_to_one")

    curves, changes = smoothing_rows(panel, sigmas, cfg, args.year)
    curve = pd.DataFrame(curves).sort_values(["model", "reference", "sigma_km"])
    curve.to_csv(output / "smoothing_curves.csv", index=False)
    pd.DataFrame(changes).sort_values(["model", "reference", "sigma_km", "block_days"]).to_csv(output / "smoothing_changes.csv", index=False)
    ladder = pd.DataFrame(ladder_rows(panel, stations, sigmas, cfg, args.year)).sort_values(["model", "support_km"])
    ladder.to_csv(output / "support_ladder.csv", index=False)

    optima = {}
    for (model, reference), group in curve.groupby(["model", "reference"]):
        optima.setdefault(model, {})[reference] = float(group.sort_values("mae_c").sigma_km.iloc[0])
    correlations = {model: float(group[group.support_km >= 0].support_km.corr(
        group[group.support_km >= 0].optimal_sigma_km_point, method="spearman"))
        for model, group in ladder.groupby("model")}
    thinned = {model: float(group.dropna(subset=["thinned_optimal_sigma_km_mean"]).support_km.corr(
        group.dropna(subset=["thinned_optimal_sigma_km_mean"]).thinned_optimal_sigma_km_mean, method="spearman"))
        for model, group in ladder.groupby("model")}
    verdict = {"pilot_name": cfg["pilot_name"], "analysis": "temporal replication of the mechanism", "year": args.year,
               "models": family["models"], "lead_hours": lead, "n_initialisations": len(inits),
               "n_cases": int(len(panel)), "n_stations": int(panel.station_id.nunique()),
               "graphcast_absent": "not published for this year in the WeatherBench stores; not substituted",
               "optimal_sigma_km_by_reference": optima,
               "R012_era5_optimum_above_zero": {model: bool(values["era5"] > 0) for model, values in optima.items()},
               "R012_station_optimum_at_zero": {model: bool(values["avamet"] == 0) for model, values in optima.items()},
               "R013_spearman_support_versus_optimal_sigma": correlations,
               "R013_spearman_under_thinning": thinned,
               "status": "completed"}
    (output / "replication_summary.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(curve.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(ladder.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
