#!/usr/bin/env python3
"""Prepare the 2020 AMeDAS temperature-station inventory from JMA history."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "national_networks" / "jma_amedas" / "metadata"
OUTPUT = ROOT / "data" / "interim" / "jma_amedas_inventory_2020"
HISTORY_URL = "https://www.data.jma.go.jp/stats/data/mdrr/chiten/meta/amdmaster.index4"
FORMAT_URL = "https://www.data.jma.go.jp/stats/data/mdrr/man/amdmasterindex4_format.pdf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with requests.get(url, timeout=120, stream=True) as response:
            response.raise_for_status()
            with path.open("wb") as destination:
                for block in response.iter_content(1024 * 1024):
                    destination.write(block)
    return path


def active_temperature_segments(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="cp932", dtype=str)
    raw = raw[raw["Station Number"].notna()].copy()
    start = pd.to_datetime(raw["Start Date"], errors="coerce")
    end_text = raw["End Date"].replace("9999-99-99", "2100-01-01")
    end = pd.to_datetime(end_text, errors="coerce")
    selected = raw[
        raw.Temperature.eq("1")
        & start.le(pd.Timestamp("2020-12-31"))
        & end.ge(pd.Timestamp("2020-01-01"))
    ].copy()
    selected["segment_start"] = start[selected.index]
    selected["segment_end"] = end[selected.index]
    selected["native_station_id"] = selected["Station Number"].str.strip()
    selected["station_id"] = (
        "jma_amedas_" + selected.native_station_id + "_"
        + selected.segment_start.dt.strftime("%Y%m%d"))
    result = pd.DataFrame({
        "station_id": selected.station_id,
        "native_station_id": selected.native_station_id,
        "site_identifier": selected.native_station_id,
        "network": "jma_amedas",
        "station_name": selected["Station Name.2"].str.strip(),
        "station_name_kanji": selected["Station Name"].str.strip(),
        "latitude": pd.to_numeric(selected.Latitude_Precipitation, errors="coerce"),
        "longitude": pd.to_numeric(selected.Longitude_Precipitation, errors="coerce"),
        "altitude_m": pd.to_numeric(selected.Altitude_Precipitation, errors="coerce"),
        "segment_start": selected.segment_start,
        "segment_end": selected.segment_end.where(
            selected["End Date"].ne("9999-99-99"), pd.NaT),
        "report_type": "AMEDAS_ROUTE_UNKNOWN",
        "operational_exchange": pd.NA,
        "documented_non_operational": False,
        "timezone": "UTC",
    })
    if result[["latitude", "longitude", "altitude_m"]].isna().any().any():
        raise ValueError("active AMeDAS temperature inventory has missing coordinates/elevation")
    if result.station_id.duplicated().any():
        raise ValueError("duplicate AMeDAS history segment IDs")
    return result.sort_values(["native_station_id", "segment_start"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    station_path = args.output / "stations.csv"
    manifest_path = args.output / "manifest.json"
    if (station_path.exists() or manifest_path.exists()) and not args.force:
        raise FileExistsError(f"outputs exist in {args.output}; use --force")

    history = get(HISTORY_URL, args.raw / "amdmaster.index4")
    specification = get(FORMAT_URL, args.raw / "amdmasterindex4_format.pdf")
    stations = active_temperature_segments(history)
    args.output.mkdir(parents=True, exist_ok=True)
    stations.to_csv(station_path, index=False)
    report = {
        "role": "pre-download 2020 inventory; observations not yet materialised",
        "source": {
            "history_url": HISTORY_URL,
            "history_bytes": history.stat().st_size,
            "history_sha256": sha256(history),
            "format_url": FORMAT_URL,
            "format_sha256": sha256(specification),
        },
        "temperature_segments_active_during_2020": int(len(stations)),
        "native_stations": int(stations.native_station_id.nunique()),
        "segments_with_open_end": int(stations.segment_end.isna().sum()),
        "assimilation_route": "unknown pending station-level GTS/WIS provenance",
        "observation_download": (
            "blocked pending a documented bulk historical endpoint or an approved "
            "throttled retrieval plan; the public search UI is not a bulk API"
        ),
        "stations_csv": {
            "path": str(station_path),
            "bytes": station_path.stat().st_size,
            "sha256": sha256(station_path),
        },
    }
    manifest_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
