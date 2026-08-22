#!/usr/bin/env python3
"""Materialise one batch for a pre-specified AVAMET surface-variable cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from public_zarr import PublicZarr, bilinear

CONFIG = ROOT / "config" / "regional_t2m_2020.yaml"
CACHE = ROOT / "data" / "raw" / "weatherbench2_cache"


def index_of(values: np.ndarray | pd.DatetimeIndex, wanted: object, name: str) -> int:
    found = np.flatnonzero(np.asarray(values) == wanted)
    if len(found) != 1:
        raise RuntimeError(f"{name}={wanted} is missing or non-unique")
    return int(found[0])


def common_initialisations(specs: dict, start: str, end: str, cycles: list[int]) -> tuple[pd.DatetimeIndex, dict[str, PublicZarr]]:
    stores = {name: PublicZarr(spec["path"], CACHE) for name, spec in specs.items()}
    common: pd.DatetimeIndex | None = None
    for name, store in stores.items():
        times = store.times(specs[name]["time_name"])
        selected = times[(times >= pd.Timestamp(start)) & (times <= pd.Timestamp(end))]
        selected = selected[selected.hour.isin(cycles)]
        common = selected if common is None else common.intersection(selected)
    if common is None or common.empty:
        raise RuntimeError("no common model initialisations in batch")
    return common.sort_values(), stores


def forecast_values(store: PublicZarr, spec: dict, variable: str, stations: pd.DataFrame, initialisations: pd.DatetimeIndex, lead: int, scale: float, workers: int, model: str, output_name: str) -> pd.DataFrame:
    times = store.times(spec["time_name"])
    lead_index = index_of(store.timedeltas_hours(spec["lead_name"]), lead, "lead")
    latitudes, longitudes = store.coordinate(spec["latitude_name"]), store.coordinate(spec["longitude_name"])

    def one(init: pd.Timestamp) -> pd.DataFrame:
        grid = store.field2d(variable, index_of(times, init, "init"), lead_index)
        values = bilinear(grid, latitudes, longitudes, stations.latitude.to_numpy(), stations.longitude.to_numpy()) * scale
        return pd.DataFrame({"station_id": stations.station_id, "init_time": init, f"{model}_{output_name}": values})

    with ThreadPoolExecutor(max_workers=workers) as executor:
        frames = []
        for index, frame in enumerate(executor.map(one, initialisations), start=1):
            if index == 1 or index % 25 == 0 or index == len(initialisations):
                print(f"variable={output_name} model={model}: {index}/{len(initialisations)}", flush=True)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def era5_values(store: PublicZarr, spec: dict, variable: dict, stations: pd.DataFrame, valid_times: pd.DatetimeIndex, scale: float, workers: int, output_name: str) -> pd.DataFrame:
    times = store.times(spec["time_name"])
    latitudes, longitudes = store.coordinate(spec["latitude_name"]), store.coordinate(spec["longitude_name"])

    def one(valid: pd.Timestamp) -> pd.DataFrame:
        time_index = index_of(times, valid, "valid_time")
        if "era5_components" in variable:
            u_name, v_name = variable["era5_components"]
            u = bilinear(store.field2d(u_name, time_index), latitudes, longitudes, stations.latitude.to_numpy(), stations.longitude.to_numpy())
            v = bilinear(store.field2d(v_name, time_index), latitudes, longitudes, stations.latitude.to_numpy(), stations.longitude.to_numpy())
            values = np.hypot(u, v) * scale
        else:
            grid = store.field2d(variable["era5_variable"], time_index)
            values = bilinear(grid, latitudes, longitudes, stations.latitude.to_numpy(), stations.longitude.to_numpy()) * scale
        return pd.DataFrame({"station_id": stations.station_id, "valid_time": valid, f"era5_{output_name}": values})

    with ThreadPoolExecutor(max_workers=workers) as executor:
        frames = []
        for index, frame in enumerate(executor.map(one, valid_times), start=1):
            if index == 1 or index % 25 == 0 or index == len(valid_times):
                print(f"variable={output_name} reference=era5: {index}/{len(valid_times)}", flush=True)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def avamet_values(stations: pd.DataFrame, valid_times: pd.DatetimeIndex, cfg: dict, variable: dict, output_name: str) -> pd.DataFrame:
    archive = (ROOT / cfg["avamet"]["archive_glob"]).resolve().as_posix()
    source_column = variable["avamet_column"]
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    con.register("wanted_stations", stations[["station_id"]])
    values = con.execute(
        f"""SELECT a.station_id, a.observed_utc, a.{source_column} AS source_value,
                   a.temperature_c, a.relative_humidity_pct, a.pressure_hpa, a.wind_mean_kmh
              FROM read_parquet('{archive}', hive_partitioning=true) AS a
              INNER JOIN wanted_stations s USING (station_id)
              ORDER BY a.station_id, a.observed_utc"""
    ).fetchdf()
    low, high = variable["avamet_qc"]["plausible_range"]
    source_valid = values.source_value.between(low, high)
    if variable["avamet_qc"].get("offline_sentinel", False):
        offline = (values.temperature_c == 0) & (values.relative_humidity_pct == 0) & (values.pressure_hpa == 0) & (values.wind_mean_kmh == 0)
        source_valid &= ~offline
    values[f"avamet_{output_name}_qc_valid"] = source_valid.fillna(False)
    values[f"avamet_{output_name}"] = (values.source_value * variable["avamet_scale"]).where(values[f"avamet_{output_name}_qc_valid"])
    wanted = pd.DataFrame({"observed_utc": valid_times.tz_localize("UTC")})
    values = values.merge(wanted, on="observed_utc", how="inner", validate="many_to_one")
    values = values.rename(columns={"observed_utc": "valid_time"})
    values["valid_time"] = pd.to_datetime(values.valid_time, utc=True).dt.tz_localize(None)
    return values[["station_id", "valid_time", f"avamet_{output_name}", f"avamet_{output_name}_qc_valid"]]


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    con = duckdb.connect()
    con.register("frame", frame)
    con.execute(f"COPY frame TO '{path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variable", help="key from surface_variable_extension.variables")
    parser.add_argument("batch", help="quarter key from download.batches")
    parser.add_argument("--force", action="store_true", help="replace an existing batch")
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    extension = cfg["surface_variable_extension"]
    if args.variable not in extension["variables"]:
        raise ValueError(f"unknown variable {args.variable!r}; choose {sorted(extension['variables'])}")
    if args.batch not in cfg["download"]["batches"]:
        raise ValueError(f"unknown batch {args.batch!r}")
    variable = extension["variables"][args.variable]
    start, end = cfg["download"]["batches"][args.batch]
    lead = extension["lead_hours"]
    output_name = variable["output_name"]
    stations_path = ROOT / cfg["paths"]["stations_csv"]
    base_dir = ROOT / cfg["paths"]["surface_variable_directory"] / f"variable={args.variable}"
    batch_dir, availability_dir = base_dir / "batches", base_dir / "availability"
    batch_dir.mkdir(parents=True, exist_ok=True)
    availability_dir.mkdir(parents=True, exist_ok=True)
    destination = batch_dir / f"{args.batch}.parquet"
    if destination.exists() and not args.force:
        print(f"already materialised: {destination}")
        return
    stations = pd.read_csv(stations_path)
    initialisations, stores = common_initialisations(cfg["weatherbench2"]["models"], start, end, cfg["period"]["cycles_utc"])
    print(f"starting variable={args.variable} batch={args.batch}; initialisations={len(initialisations)}", flush=True)
    aligned: pd.DataFrame | None = None
    for model, spec in cfg["weatherbench2"]["models"].items():
        frame = forecast_values(stores[model], spec, variable["model_variable"], stations, initialisations, lead, variable["model_scale"], cfg["download"]["max_workers"], model, output_name)
        frame["valid_time"] = frame.init_time + pd.Timedelta(hours=lead)
        frame["lead_h"] = lead
        aligned = frame if aligned is None else aligned.merge(frame, on=["station_id", "init_time", "valid_time", "lead_h"], validate="one_to_one")
    assert aligned is not None
    valid_times = initialisations + pd.Timedelta(hours=lead)
    era5 = era5_values(PublicZarr(cfg["weatherbench2"]["era5"]["path"], CACHE), cfg["weatherbench2"]["era5"], variable, stations, valid_times, variable["model_scale"], cfg["download"]["max_workers"], output_name)
    aligned = aligned.merge(era5, on=["station_id", "valid_time"], validate="one_to_one")
    aligned = aligned.merge(avamet_values(stations, valid_times, cfg, variable, output_name), on=["station_id", "valid_time"], how="left", validate="one_to_one")
    aligned = aligned.merge(stations, on="station_id", validate="many_to_one")
    write_parquet(aligned, destination)
    manifest = {"pilot_name": cfg["pilot_name"], "variable": args.variable, "label": variable["label"], "unit": variable["unit"], "lead_hours": lead,
                "batch": args.batch, "common_initialisations": len(initialisations), "stations": len(stations), "expected_cases": len(aligned),
                "avamet_qc_valid": int(aligned[f"avamet_{output_name}_qc_valid"].sum()), "output": str(destination.relative_to(ROOT)),
                "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(), "station_table_sha256": hashlib.sha256(stations_path.read_bytes()).hexdigest(),
                "status": "materialised for a separately labelled surface-variable exploratory cohort"}
    (availability_dir / f"{args.batch}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
