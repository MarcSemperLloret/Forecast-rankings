#!/usr/bin/env python3
"""Prepare exact 00/12 UTC MIDAS Open air temperature observations for 2020."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "national_networks" / "midas_open_hourly_2020"
OUTPUT = ROOT / "data" / "interim" / "midas_open_t2m_2020"
OPERATIONAL_REPORT_TYPES = {"SYNOP", "METAR", "GTS"}
REPORT_PRIORITY = {
    "AWSHRLY": 0,
    "NCM": 1,
    "HCM": 2,
    "DLY3208": 3,
    "SYNOP": 4,
    "METAR": 5,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata_and_header(path: Path) -> tuple[dict[str, list[str]], int]:
    metadata: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for index, row in enumerate(csv.reader(stream)):
            if row and row[0] == "ob_time":
                return metadata, index
            if len(row) >= 3 and row[1] == "G":
                metadata[row[0]] = row[2:]
    raise ValueError(f"BADC-CSV data header not found: {path}")


def scalar(metadata: dict[str, list[str]], key: str, default: str = "") -> str:
    values = metadata.get(key, [])
    return values[0].strip() if values else default


def number(metadata: dict[str, list[str]], key: str, index: int = 0) -> float:
    values = metadata.get(key, [])
    return float(values[index]) if len(values) > index and values[index] else float("nan")


def unique_values(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame:
        return []
    values = frame[column].dropna().astype(str).str.strip()
    return sorted(value for value in values.unique() if value and value != "NA")


def identifier(frame: pd.DataFrame, identifier_type: str) -> str | pd._libs.missing.NAType:
    rows = frame[frame["id_type"].fillna("").str.upper() == identifier_type]
    values = unique_values(rows, "id")
    return "|".join(values) if values else pd.NA


def read_station(path: Path) -> tuple[dict[str, object], pd.DataFrame, dict[str, int], Counter]:
    metadata, header = metadata_and_header(path)
    columns = [
        "ob_time",
        "id",
        "id_type",
        "met_domain_name",
        "version_num",
        "src_id",
        "air_temperature",
        "air_temperature_q",
        "air_temperature_j",
    ]
    frame = pd.read_csv(
        path,
        skiprows=header,
        usecols=columns,
        dtype="string",
        na_values=["NA"],
        keep_default_na=True,
        low_memory=False,
    )
    counts = {
        "raw_rows": len(frame),
        "commissioning_rows": 0,
        "physical_rows": 0,
        "conflicting_station_time_rows": 0,
        "deduplicated_rows": 0,
        "local_qc_rows": 0,
        "nominal_00_12_rows": 0,
    }
    frame["valid_time"] = pd.to_datetime(frame["ob_time"], errors="coerce", utc=True)
    frame["observed_t2m_c"] = pd.to_numeric(frame["air_temperature"], errors="coerce")
    frame["src_id_numeric"] = pd.to_numeric(frame["src_id"], errors="coerce")
    commissioning = frame["src_id_numeric"].eq(99999)
    counts["commissioning_rows"] = int(commissioning.sum())
    frame = frame[
        ~commissioning
        & frame["valid_time"].notna()
        & frame["observed_t2m_c"].between(-90, 60)
    ].copy()
    counts["physical_rows"] = len(frame)

    report_types = unique_values(frame, "met_domain_name")
    archive_qc_counts = Counter(
        frame["air_temperature_q"].fillna("NA").astype(str).value_counts().to_dict()
    )
    international = frame[["id", "id_type"]].drop_duplicates().copy()

    frame["temperature_range"] = frame.groupby("valid_time")["observed_t2m_c"].transform(
        lambda values: values.max() - values.min()
    )
    conflicting = frame["temperature_range"] > 0.2
    counts["conflicting_station_time_rows"] = int(conflicting.sum())
    frame = frame[~conflicting].copy()
    frame["report_priority"] = (
        frame["met_domain_name"].fillna("").str.upper().map(REPORT_PRIORITY).fillna(99)
    )
    frame["version_priority"] = -pd.to_numeric(frame["version_num"], errors="coerce").fillna(-1)
    frame = (
        frame.sort_values(["valid_time", "report_priority", "version_priority", "met_domain_name"])
        .drop_duplicates("valid_time", keep="first")
        .sort_values("valid_time")
        .copy()
    )
    counts["deduplicated_rows"] = len(frame)
    rolling = frame["observed_t2m_c"].rolling(5, center=True, min_periods=3).median()
    frame = frame[(frame["observed_t2m_c"] - rolling).abs() <= 4.0].copy()
    counts["local_qc_rows"] = len(frame)
    frame = frame[
        frame["valid_time"].dt.year.eq(2020)
        & frame["valid_time"].dt.hour.isin([0, 12])
        & frame["valid_time"].dt.minute.eq(0)
        & frame["valid_time"].dt.second.eq(0)
    ].copy()
    counts["nominal_00_12_rows"] = len(frame)

    native_id = scalar(metadata, "midas_station_id")
    if not native_id:
        raise ValueError(f"missing MIDAS station ID: {path}")
    station_id = f"midas_{native_id}"
    station = {
        "station_id": station_id,
        "native_station_id": native_id,
        "site_identifier": native_id,
        "network": "midas_open_uk",
        "station_name": scalar(metadata, "observation_station"),
        "county": scalar(metadata, "historic_county_name"),
        "latitude": number(metadata, "location", 0),
        "longitude": number(metadata, "location", 1),
        "altitude_m": number(metadata, "height", 0),
        "report_type": report_types[0] if len(report_types) == 1 else "MIXED",
        "report_types_observed": "|".join(report_types),
        "operational_exchange": bool(set(report_types) & OPERATIONAL_REPORT_TYPES),
        "documented_non_operational": False,
        "wmo_id": identifier(international, "WMO"),
        "icao_id": identifier(international, "ICAO"),
        "timezone": "UTC",
        "source_file": path.relative_to(RAW).as_posix(),
    }
    frame["station_id"] = station_id
    observations = frame[["station_id", "valid_time", "observed_t2m_c"]].copy()
    observations["valid_time"] = observations["valid_time"].dt.tz_localize(None)
    return station, observations, counts, archive_qc_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--min-samples", type=int, default=650)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    outputs = [args.output / name for name in (
        "stations_all.csv", "stations.csv", "observations.parquet", "manifest.json")]
    if any(path.exists() for path in outputs) and not args.force:
        raise FileExistsError(f"outputs exist in {args.output}; use --force")

    files = sorted((args.raw / "files").rglob("*_qcv-1_2020.csv"))
    if not files:
        raise FileNotFoundError(f"no MIDAS 2020 files under {args.raw / 'files'}")
    stations: list[dict[str, object]] = []
    observations: list[pd.DataFrame] = []
    totals: Counter = Counter()
    qc_codes: Counter = Counter()
    for index, path in enumerate(files, 1):
        station, station_observations, counts, station_qc = read_station(path)
        stations.append(station)
        observations.append(station_observations)
        totals.update(counts)
        qc_codes.update(station_qc)
        if index % 50 == 0 or index == len(files):
            print(f"Prepared {index}/{len(files)} station files", flush=True)

    station_frame = pd.DataFrame(stations)
    if station_frame["station_id"].duplicated().any():
        duplicates = station_frame.loc[
            station_frame["station_id"].duplicated(False), "station_id"
        ].tolist()
        raise ValueError(f"duplicate MIDAS station identifiers: {duplicates[:10]}")
    observation_frame = pd.concat(observations, ignore_index=True)
    if observation_frame.duplicated(["station_id", "valid_time"]).any():
        raise ValueError("duplicate MIDAS station-time rows after preparation")
    coverage = observation_frame.groupby("station_id").size()
    eligible = coverage[coverage >= args.min_samples]
    selected = station_frame[station_frame["station_id"].isin(eligible.index)].copy()
    selected["n_2020_nominal_00_12"] = selected["station_id"].map(eligible).astype(int)
    filtered = observation_frame[observation_frame["station_id"].isin(eligible.index)].copy()

    args.output.mkdir(parents=True, exist_ok=True)
    station_frame.to_csv(args.output / "stations_all.csv", index=False)
    selected.to_csv(args.output / "stations.csv", index=False)
    connection = duckdb.connect()
    connection.register("observations", filtered)
    connection.execute(
        f"COPY observations TO '{(args.output / 'observations.parquet').as_posix()}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    connection.close()
    totals.update({
        "source_files": len(files),
        "stations_in_archive_2020": len(station_frame),
        "stations_meeting_min_samples": len(selected),
        "rows_after_station_coverage": len(filtered),
        "min_nominal_00_12_samples": args.min_samples,
        "stations_with_operational_report_route": int(selected["operational_exchange"].sum()),
    })
    report = {
        **dict(totals),
        "role": "UK national-network candidate; station/report-route audit required",
        "source_dataset": "MIDAS Open UK hourly weather observations v202107",
        "doi": "10.5285/3bd7221d4844435dad2fa030f26ab5fd",
        "raw_manifest": str(args.raw / "manifest.json"),
        "timestamp_semantics": "MIDAS ob_time interpreted as UTC; exact 00 and 12 retained",
        "archive_qc": "QC version 1 current-best annual files",
        "archive_air_temperature_q_counts": dict(sorted(qc_codes.items())),
        "local_qc": (
            "src_id 99999 removed; physical range -90 to 60 C; conflicting duplicate "
            "station-times (>0.2 C range) removed; centred five-report median deviation <=4 C"
        ),
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in outputs[:-1]
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
