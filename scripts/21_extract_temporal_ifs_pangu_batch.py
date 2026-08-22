#!/usr/bin/env python3
"""Materialise a resumable 2021–2022 IFS/Pangu two-model T2m cohort."""
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
MODELS = ("ifs_hres", "pangu_hres_init")


def index_of(values: np.ndarray | pd.DatetimeIndex, wanted: object, name: str) -> int:
    found = np.flatnonzero(np.asarray(values) == wanted)
    if len(found) != 1:
        raise RuntimeError(f"{name}={wanted} is missing or non-unique")
    return int(found[0])


def year_batch(cfg: dict, year: int, batch: str) -> tuple[str, str]:
    start, end = (pd.Timestamp(value).replace(year=year) for value in cfg["download"]["batches"][batch])
    return start.isoformat(), end.isoformat()


def common_initialisations(specs: dict, start: str, end: str, cycles: list[int]) -> tuple[pd.DatetimeIndex, dict[str, PublicZarr]]:
    stores = {name: PublicZarr(spec["path"], CACHE) for name, spec in specs.items()}
    common: pd.DatetimeIndex | None = None
    for name, store in stores.items():
        times = store.times(specs[name]["time_name"])
        chosen = times[(times >= pd.Timestamp(start)) & (times <= pd.Timestamp(end))]
        chosen = chosen[chosen.hour.isin(cycles)]
        common = chosen if common is None else common.intersection(chosen)
    if common is None or common.empty:
        raise RuntimeError("no common initialisations")
    return common.sort_values(), stores


def forecast_values(store: PublicZarr, spec: dict, stations: pd.DataFrame, initialisations: pd.DatetimeIndex, lead: int, workers: int, name: str) -> pd.DataFrame:
    times, leads = store.times(spec["time_name"]), store.timedeltas_hours(spec["lead_name"])
    lead_index = index_of(leads, lead, "lead")
    lats, lons = store.coordinate(spec["latitude_name"]), store.coordinate(spec["longitude_name"])

    def one(init: pd.Timestamp) -> pd.DataFrame:
        grid = store.field2d(spec["t2m_name"], index_of(times, init, "init"), lead_index)
        values = bilinear(grid, lats, lons, stations.latitude.to_numpy(), stations.longitude.to_numpy()) - 273.15
        return pd.DataFrame({"station_id": stations.station_id, "init_time": init, f"{name}_t2m_c": values})

    with ThreadPoolExecutor(max_workers=workers) as executor:
        frames = []
        for number, frame in enumerate(executor.map(one, initialisations), start=1):
            if number == 1 or number % 25 == 0 or number == len(initialisations):
                print(f"model={name}: {number}/{len(initialisations)}", flush=True)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def era5_values(store: PublicZarr, spec: dict, stations: pd.DataFrame, valid_times: pd.DatetimeIndex, workers: int) -> pd.DataFrame:
    times = store.times(spec["time_name"])
    lats, lons = store.coordinate(spec["latitude_name"]), store.coordinate(spec["longitude_name"])

    def one(valid: pd.Timestamp) -> pd.DataFrame:
        grid = store.field2d(spec["t2m_name"], index_of(times, valid, "valid_time"))
        values = bilinear(grid, lats, lons, stations.latitude.to_numpy(), stations.longitude.to_numpy()) - 273.15
        return pd.DataFrame({"station_id": stations.station_id, "valid_time": valid, "era5_t2m_c": values})

    with ThreadPoolExecutor(max_workers=workers) as executor:
        frames = []
        for number, frame in enumerate(executor.map(one, valid_times), start=1):
            if number == 1 or number % 25 == 0 or number == len(valid_times):
                print(f"reference=era5: {number}/{len(valid_times)}", flush=True)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def avamet_values(stations: pd.DataFrame, valid_times: pd.DatetimeIndex, cfg: dict, year: int) -> pd.DataFrame:
    archive = (ROOT / cfg["temporal_replication"]["avamet_archive_glob"].format(year=year)).resolve().as_posix()
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    con.register("wanted_stations", stations[["station_id"]])
    values = con.execute(f"SELECT a.station_id, a.observed_utc, a.temperature_c FROM read_parquet('{archive}', hive_partitioning=true) AS a INNER JOIN wanted_stations s USING (station_id) ORDER BY a.station_id, a.observed_utc").fetchdf()
    low, high = cfg["avamet"]["t2m_plausible_c"]
    qc = cfg["avamet"]["qc"]
    physical = values.temperature_c.between(low, high)
    masked = values.temperature_c.where(physical)
    median = masked.groupby(values.station_id).transform(lambda item: item.rolling(qc["rolling_window_reports"], center=True, min_periods=3).median())
    values["avamet_t2m_qc_valid"] = (physical & ((masked - median).abs() <= qc["local_median_deviation_c"])).fillna(False)
    values["avamet_t2m_qc_c"] = values.temperature_c.where(values.avamet_t2m_qc_valid)
    values = values.merge(pd.DataFrame({"observed_utc": valid_times.tz_localize("UTC")}), on="observed_utc", how="inner", validate="many_to_one")
    values = values.rename(columns={"observed_utc": "valid_time", "temperature_c": "avamet_t2m_c"})
    values["valid_time"] = pd.to_datetime(values.valid_time, utc=True).dt.tz_localize(None)
    return values[["station_id", "valid_time", "avamet_t2m_c", "avamet_t2m_qc_c", "avamet_t2m_qc_valid"]]


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    con = duckdb.connect(); con.register("frame", frame)
    con.execute(f"COPY frame TO '{path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("year", type=int, choices=(2021, 2022))
    parser.add_argument("batch", choices=("q1", "q2", "q3", "q4"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    temporal = cfg["temporal_replication"]
    lead = temporal["leads_hours"][0]
    start, end = year_batch(cfg, args.year, args.batch)
    specs = {"ifs_hres": cfg["weatherbench2"]["models"]["ifs_hres"], "pangu_hres_init": {**cfg["weatherbench2"]["models"]["pangu_hres_init"], "path": temporal["pangu_path_template"].format(year=args.year)}}
    root = ROOT / cfg["paths"]["temporal_directory"] / f"year={args.year}" / f"lead={lead:03d}"
    destination, manifest_path = root / "batches" / f"{args.batch}.parquet", root / "availability" / f"{args.batch}.json"
    destination.parent.mkdir(parents=True, exist_ok=True); manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not args.force:
        print(f"already materialised: {destination}"); return
    stations_path = ROOT / temporal["station_table"]
    stations = pd.read_csv(stations_path)
    initialisations, stores = common_initialisations(specs, start, end, cfg["period"]["cycles_utc"])
    print(f"starting temporal year={args.year} batch={args.batch}; stations={len(stations)}; inits={len(initialisations)}", flush=True)
    aligned: pd.DataFrame | None = None
    for name in MODELS:
        values = forecast_values(stores[name], specs[name], stations, initialisations, lead, cfg["download"]["max_workers"], name)
        values["valid_time"] = values.init_time + pd.Timedelta(hours=lead); values["lead_h"] = lead; values["analysis_year"] = args.year
        aligned = values if aligned is None else aligned.merge(values, on=["station_id", "init_time", "valid_time", "lead_h", "analysis_year"], validate="one_to_one")
    assert aligned is not None
    valid_times = initialisations + pd.Timedelta(hours=lead)
    era5 = era5_values(PublicZarr(cfg["weatherbench2"]["era5"]["path"], CACHE), cfg["weatherbench2"]["era5"], stations, valid_times, cfg["download"]["max_workers"])
    aligned = aligned.merge(era5, on=["station_id", "valid_time"], validate="one_to_one")
    aligned = aligned.merge(avamet_values(stations, valid_times, cfg, args.year), on=["station_id", "valid_time"], how="left", validate="one_to_one").merge(stations, on="station_id", validate="many_to_one")
    write_parquet(aligned, destination)
    manifest = {"pilot_name": cfg["pilot_name"], "analysis": "temporal IFS/Pangu two-model sensitivity", "year": args.year, "lead_hours": lead, "batch": args.batch, "models": list(MODELS), "common_initialisations": len(initialisations), "stations": len(stations), "expected_cases": len(aligned), "avamet_qc_valid": int(aligned.avamet_t2m_qc_valid.sum()), "output": str(destination.relative_to(ROOT)), "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(), "station_table_sha256": hashlib.sha256(stations_path.read_bytes()).hexdigest(), "status": "materialised"}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
