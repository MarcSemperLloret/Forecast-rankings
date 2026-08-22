#!/usr/bin/env python3
"""Prepare the 2020 DWD urban hourly-temperature sensitivity stratum.

The DWD ``recent`` archive contains the complete series since each urban site
opened. Its nominal UTC hour represents the one-minute mean ending ten minutes
before that hour; both the nominal verification time and this offset are kept
explicitly in the output.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import duckdb
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = (
    "https://opendata.dwd.de/climate_environment/CDC/observations_germany/"
    "climate_urban/hourly/air_temperature/recent"
)
RAW = ROOT / "data" / "raw" / "dwd_urban_hourly"
OUTPUT = ROOT / "data" / "interim" / "dwd_urban_t2m_2020"
METADATA_NAME = "TU_STADT_Stundenwerte_Beschreibung_Stationen.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(raw: Path) -> list[Path]:
    raw.mkdir(parents=True, exist_ok=True)
    response = requests.get(f"{BASE_URL}/", timeout=60)
    response.raise_for_status()
    names = sorted(set(re.findall(r'href="([^"]+(?:\.zip|Stationen\.txt))"', response.text)))
    if not names:
        raise RuntimeError("DWD directory listing contained no station archives")
    paths = []
    for name in names:
        target = raw / name
        if not target.exists():
            with requests.get(f"{BASE_URL}/{name}", timeout=120, stream=True) as item:
                item.raise_for_status()
                with target.open("wb") as stream:
                    for block in item.iter_content(1024 * 1024):
                        stream.write(block)
        paths.append(target)
    return paths


def read_stations(path: Path) -> pd.DataFrame:
    rows = []
    lines = path.read_text(encoding="latin-1").splitlines()[2:]
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 8:
            raise ValueError(f"cannot parse station metadata line: {line!r}")
        rows.append({
            "station_id": f"dwd_urban_{int(parts[0]):05d}",
            "native_station_id": str(int(parts[0])),
            "open_date": pd.to_datetime(parts[1], format="%Y%m%d"),
            "archive_end_date": pd.to_datetime(parts[2], format="%Y%m%d"),
            "altitude_m": float(parts[3]),
            "latitude": float(parts[4]),
            "longitude": float(parts[5]),
            "station_name": " ".join(parts[6:-1]),
            "state": parts[-1],
        })
    return pd.DataFrame(rows)


def read_observations(archives: list[Path]) -> pd.DataFrame:
    frames = []
    for archive in archives:
        with zipfile.ZipFile(archive) as bundle:
            products = [name for name in bundle.namelist() if name.startswith("produkt_")]
            if len(products) != 1:
                raise ValueError(f"{archive} contains {len(products)} product files")
            raw = bundle.read(products[0])
        frame = pd.read_csv(io.BytesIO(raw), sep=";", skipinitialspace=True)
        frame.columns = frame.columns.str.strip()
        frame = frame.rename(columns={
            "STATIONS_ID": "native_station_id",
            "MESS_DATUM": "valid_time",
            "QUALITAETS_NIVEAU": "qc_level",
            "LUFTTEMPERATUR": "observed_t2m_c",
        })
        frame["native_station_id"] = frame["native_station_id"].astype(str).str.strip().astype(int).astype(str)
        frame["station_id"] = "dwd_urban_" + frame["native_station_id"].astype(int).astype(str).str.zfill(5)
        frame["valid_time"] = pd.to_datetime(frame["valid_time"].astype(str).str.strip(), format="%Y%m%d%H")
        frames.append(frame[["station_id", "valid_time", "qc_level", "observed_t2m_c"]])
    return pd.concat(frames, ignore_index=True)


def prepare(stations: pd.DataFrame, observations: pd.DataFrame, min_samples: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    year = observations[observations.valid_time.dt.year == 2020].copy()
    year["qc_level"] = pd.to_numeric(year.qc_level, errors="coerce")
    year["observed_t2m_c"] = pd.to_numeric(year.observed_t2m_c, errors="coerce")
    before = len(year)
    year = year[(year.qc_level >= 2) & year.observed_t2m_c.between(-90, 60)].copy()
    after_source_qc = len(year)
    year = year.sort_values(["station_id", "valid_time"])
    rolling = year.groupby("station_id").observed_t2m_c.transform(
        lambda values: values.rolling(5, center=True, min_periods=3).median())
    year = year[(year.observed_t2m_c - rolling).abs() <= 4.0].copy()
    after_local_qc = len(year)
    year = year[year.valid_time.dt.hour.isin([0, 12])].copy()
    counts = year.groupby("station_id").size()
    eligible = counts[counts >= min_samples].index
    year = year[year.station_id.isin(eligible)].copy()
    selected = stations[stations.station_id.isin(eligible)].copy()
    selected["network"] = "dwd_urban"
    selected["report_type"] = "URBAN_CLIMATE"
    # Non-WMO-standard siting is not documentary proof of a non-GTS route.
    selected["documented_non_operational"] = False
    selected["operational_exchange"] = pd.NA
    selected["timezone"] = "UTC"
    selected["measurement_time_offset_min"] = -10
    selected["n_2020_nominal_00_12"] = selected.station_id.map(counts).astype(int)
    year["measurement_time_offset_min"] = -10
    year["measurement_window_end"] = year.valid_time - pd.Timedelta(minutes=10)
    year["source_dataset"] = "DWD urban climate hourly air temperature, recent archive"
    report = {
        "year_rows_before_qc": before,
        "year_rows_after_source_qc": after_source_qc,
        "year_rows_after_local_median_qc": after_local_qc,
        "nominal_00_12_rows_after_station_coverage": int(len(year)),
        "stations_in_archive": int(len(stations)),
        "stations_with_any_2020_data": int((stations.open_date.dt.year <= 2020).sum()),
        "stations_meeting_min_samples": int(len(selected)),
        "min_nominal_00_12_samples": min_samples,
        "timestamp_semantics": "nominal UTC hour; DWD documents one-minute mean ending 10 minutes earlier",
    }
    return selected, year, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--min-samples", type=int, default=650)
    args = parser.parse_args()

    inputs = download(args.raw)
    metadata = args.raw / METADATA_NAME
    archives = sorted(args.raw.glob("*.zip"))
    stations = read_stations(metadata)
    observations = read_observations(archives)
    selected, filtered, report = prepare(stations, observations, args.min_samples)

    args.output.mkdir(parents=True, exist_ok=True)
    station_path = args.output / "stations.csv"
    observation_path = args.output / "observations.parquet"
    selected.to_csv(station_path, index=False)
    connection = duckdb.connect()
    connection.register("filtered_observations", filtered)
    connection.execute(f"COPY filtered_observations TO '{observation_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    connection.close()

    report.update({
        "role": "non-overlapping sensitivity candidate with unknown assimilation route; not confirmatory",
        "source_url": BASE_URL,
        "input_sha256": {path.name: sha256(path) for path in inputs},
        "stations_csv": str(station_path),
        "observations_parquet": str(observation_path),
    })
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
