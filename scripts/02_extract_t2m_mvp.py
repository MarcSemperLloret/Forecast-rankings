#!/usr/bin/env python3
"""Extract and align the first AVAMET/ERA5/model T2m MVP from public Zarr data."""
from __future__ import annotations

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

CONFIG_PATH = ROOT / "config" / "mvp_t2m.yaml"
CACHE_ROOT = ROOT / "data" / "raw" / "weatherbench2_cache"


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def find_index(values: pd.DatetimeIndex | np.ndarray, wanted: object, label: str) -> int:
    matches = np.flatnonzero(np.asarray(values) == wanted)
    if len(matches) != 1:
        raise RuntimeError(f"{label} {wanted} is missing or non-unique")
    return int(matches[0])


def common_initialisations(model_specs: dict, start: str, end: str, cycles: list[int]) -> tuple[pd.DatetimeIndex, dict[str, PublicZarr]]:
    stores = {name: PublicZarr(spec["path"], CACHE_ROOT) for name, spec in model_specs.items()}
    common: pd.DatetimeIndex | None = None
    for name, store in stores.items():
        times = store.times(model_specs[name]["time_name"])
        times = times[(times >= pd.Timestamp(start)) & (times <= pd.Timestamp(end))]
        times = times[times.hour.isin(cycles)]
        common = times if common is None else common.intersection(times)
    if common is None or common.empty:
        raise RuntimeError("no common forecast initialisations in the configured window")
    return common.sort_values(), stores


def extract_model(store: PublicZarr, spec: dict, stations: pd.DataFrame,
                  initialisations: pd.DatetimeIndex, lead_h: int, max_workers: int) -> pd.DataFrame:
    times = store.times(spec["time_name"])
    leads = store.timedeltas_hours(spec["lead_name"])
    lead_index = find_index(leads, lead_h, "lead")
    latitudes = store.coordinate(spec["latitude_name"])
    longitudes = store.coordinate(spec["longitude_name"])
    def one(init_time: pd.Timestamp) -> pd.DataFrame:
        grid = store.field2d(spec["t2m_name"], find_index(times, init_time, "init_time"), lead_index)
        values_c = bilinear(grid, latitudes, longitudes, stations.latitude.to_numpy(), stations.longitude.to_numpy()) - 273.15
        return pd.DataFrame({"station_id": stations.station_id, "init_time": init_time, "t2m_c": values_c})

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        frames = []
        for index, frame in enumerate(executor.map(one, initialisations), start=1):
            print(f"  {index}/{len(initialisations)}", flush=True)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def extract_era5(store: PublicZarr, spec: dict, stations: pd.DataFrame,
                 valid_times: pd.DatetimeIndex, max_workers: int) -> pd.DataFrame:
    times = store.times(spec["time_name"])
    latitudes = store.coordinate(spec["latitude_name"])
    longitudes = store.coordinate(spec["longitude_name"])
    def one(valid_time: pd.Timestamp) -> pd.DataFrame:
        grid = store.field2d(spec["t2m_name"], find_index(times, valid_time, "valid_time"))
        values_c = bilinear(grid, latitudes, longitudes, stations.latitude.to_numpy(), stations.longitude.to_numpy()) - 273.15
        return pd.DataFrame({"station_id": stations.station_id, "valid_time": valid_time, "era5_t2m_c": values_c})

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        frames = []
        for index, frame in enumerate(executor.map(one, valid_times), start=1):
            print(f"  ERA5 {index}/{len(valid_times)}", flush=True)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_avamet(stations: pd.DataFrame, valid_times: pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    archive = (ROOT / cfg["avamet"]["archive_glob"]).resolve().as_posix()
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    con.register("wanted_stations", stations[["station_id"]])
    series = con.execute(
        f"""
        SELECT a.station_id, a.observed_utc, a.temperature_c
        FROM read_parquet('{archive}', hive_partitioning=true) AS a
        INNER JOIN wanted_stations AS s USING (station_id)
        ORDER BY a.station_id, a.observed_utc
        """
    ).fetchdf()
    low, high = cfg["avamet"]["t2m_plausible_c"]
    qc = cfg["avamet"]["qc"]
    series["in_physical_range"] = series.temperature_c.between(low, high)
    masked = series.temperature_c.where(series.in_physical_range)
    local_median = masked.groupby(series.station_id).transform(
        lambda values: values.rolling(qc["rolling_window_reports"], center=True, min_periods=3).median()
    )
    series["local_median_deviation_c"] = (masked - local_median).abs()
    series["avamet_t2m_qc_valid"] = (
        series.in_physical_range
        & (series.local_median_deviation_c <= qc["local_median_deviation_c"])
    ).fillna(False)
    series["avamet_t2m_qc_c"] = series.temperature_c.where(series.avamet_t2m_qc_valid)
    wanted = pd.DataFrame({"observed_utc": valid_times.tz_localize("UTC")})
    result = series.merge(wanted, on="observed_utc", how="inner", validate="many_to_one")
    result = result.rename(columns={"observed_utc": "valid_time", "temperature_c": "avamet_t2m_c"})
    result["valid_time"] = pd.to_datetime(result["valid_time"], utc=True).dt.tz_localize(None)
    return result[["station_id", "valid_time", "avamet_t2m_c", "avamet_t2m_qc_c", "avamet_t2m_qc_valid"]]


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    con = duckdb.connect()
    con.register("frame", frame)
    con.execute(f"COPY frame TO '{path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")


def main() -> None:
    cfg = load_config()
    stations_path = ROOT / "data" / "processed" / "stations_mvp.csv"
    if not stations_path.exists():
        raise RuntimeError("run scripts/01_select_stations.py first")
    stations = pd.read_csv(stations_path)
    wb = cfg["weatherbench2"]
    initialisations, stores = common_initialisations(
        wb["models"], cfg["period"]["init_start"], cfg["period"]["init_end"], cfg["period"]["cycles_utc"]
    )
    lead = cfg["leads_hours"][0]
    max_workers = cfg["download"]["max_workers"]
    if cfg["leads_hours"] != [24]:
        raise RuntimeError("the first extractor is intentionally frozen to lead +24 h")
    valid_times = initialisations + pd.Timedelta(hours=lead)
    aligned: pd.DataFrame | None = None
    for name, spec in wb["models"].items():
        print(f"extrayendo {name} ...", flush=True)
        frame = extract_model(stores[name], spec, stations, initialisations, lead, max_workers).rename(columns={"t2m_c": f"{name}_t2m_c"})
        frame["valid_time"] = frame.init_time + pd.Timedelta(hours=lead)
        frame["lead_h"] = lead
        aligned = frame if aligned is None else aligned.merge(frame, on=["station_id", "init_time", "valid_time", "lead_h"], validate="one_to_one")
    assert aligned is not None
    print("extrayendo ERA5 ...", flush=True)
    era5_frame = extract_era5(PublicZarr(wb["era5"]["path"], CACHE_ROOT), wb["era5"], stations, valid_times, max_workers)
    aligned = aligned.merge(era5_frame, on=["station_id", "valid_time"], validate="one_to_one")
    aligned = aligned.merge(load_avamet(stations, valid_times, cfg), on=["station_id", "valid_time"], how="left", validate="one_to_one")
    aligned = aligned.merge(stations, on="station_id", validate="many_to_one")
    interim, results = ROOT / "data" / "interim", ROOT / "results"
    interim.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    write_parquet(aligned, interim / "aligned_t2m_mvp.parquet")
    summary = {
        "pilot_name": cfg["pilot_name"], "stations": len(stations),
        "common_initialisations": len(initialisations),
        "expected_station_forecast_cases": int(len(stations) * len(initialisations)),
        "avamet_rows_matched": int(aligned.avamet_t2m_c.notna().sum()),
        "avamet_rows_qc_valid": int(aligned.avamet_t2m_qc_valid.sum()),
        "models": list(wb["models"]), "lead_hours": lead,
        "valid_time_start": str(valid_times.min()), "valid_time_end": str(valid_times.max()),
        "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "station_table_sha256": hashlib.sha256(stations_path.read_bytes()).hexdigest(),
        "status": "alignment complete; sample is a technical check, not a scoring dataset",
    }
    (results / "availability.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
