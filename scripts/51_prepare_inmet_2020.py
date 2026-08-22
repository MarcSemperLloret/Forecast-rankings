#!/usr/bin/env python3
"""Download and prepare exact 00/12 UTC INMET automatic T2m for 2020."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import duckdb
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://portal.inmet.gov.br/uploads/dadoshistoricos/2020.zip"
RAW = ROOT / "data" / "raw" / "national_networks" / "inmet_2020"
OUTPUT = ROOT / "data" / "interim" / "inmet_t2m_2020"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(raw: Path) -> tuple[Path, Path]:
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "2020.zip"
    if not archive.exists():
        with requests.get(SOURCE_URL, timeout=180, stream=True) as response:
            response.raise_for_status()
            with archive.open("wb") as destination:
                for block in response.iter_content(1024 * 1024):
                    destination.write(block)
    extracted = raw / "extracted"
    extracted.mkdir(exist_ok=True)
    if not any(extracted.glob("*.CSV")):
        with zipfile.ZipFile(archive) as bundle:
            root = extracted.resolve()
            for member in bundle.infolist():
                target = (extracted / member.filename).resolve()
                if root not in target.parents and target != root:
                    raise ValueError(f"unsafe ZIP member: {member.filename}")
            bundle.extractall(extracted)
    return archive, extracted


def metadata_and_frame(path: Path) -> tuple[dict, pd.DataFrame]:
    raw = path.read_bytes()
    lines = raw.decode("latin-1").splitlines()
    if len(lines) < 10:
        raise ValueError(f"short INMET file: {path}")
    metadata = {}
    for line in lines[:8]:
        key, value, *_ = line.split(";")
        metadata[key.rstrip(":").strip()] = value.strip()
    frame = pd.read_csv(io.BytesIO(raw), sep=";", skiprows=8, encoding="latin-1", dtype=str)
    frame.columns = frame.columns.str.strip()
    temperature_columns = [name for name in frame if "BULBO SECO" in name]
    if len(temperature_columns) != 1:
        raise ValueError(f"temperature column not unique in {path}: {temperature_columns}")
    return metadata, frame[["Data", "Hora UTC", temperature_columns[0]]].rename(
        columns={temperature_columns[0]: "observed_t2m_c"})


def decimal(value: object) -> float:
    return float(str(value).replace(",", "."))


def prepare(extracted: Path, min_samples: int) -> tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    stations = []
    observations = []
    counts = {
        "raw_rows": 0,
        "physical_rows": 0,
        "local_qc_rows": 0,
        "nominal_00_12_rows": 0,
    }
    files = sorted(extracted.glob("*.CSV"))
    if not files:
        raise FileNotFoundError(f"no INMET station CSV files in {extracted}")
    for path in files:
        metadata, frame = metadata_and_frame(path)
        native_id = metadata["CODIGO (WMO)"]
        station_id = f"inmet_{native_id.lower()}"
        stations.append({
            "station_id": station_id,
            "native_station_id": native_id,
            "site_identifier": native_id,
            "network": "inmet_hourly",
            "station_name": metadata["ESTACAO"],
            "region": metadata["REGIAO"],
            "state": metadata["UF"],
            "latitude": decimal(metadata["LATITUDE"]),
            "longitude": decimal(metadata["LONGITUDE"]),
            "altitude_m": decimal(metadata["ALTITUDE"]),
            "report_type": "INMET_AUTOMATIC",
            "operational_exchange": pd.NA,
            "documented_non_operational": False,
            "timezone": "UTC",
            "source_file": path.name,
        })
        counts["raw_rows"] += len(frame)
        frame["observed_t2m_c"] = pd.to_numeric(
            frame.observed_t2m_c.str.replace(",", ".", regex=False), errors="coerce")
        frame["valid_time"] = pd.to_datetime(
            frame.Data.str.strip() + " " + frame["Hora UTC"].str.strip(),
            format="%Y/%m/%d %H%M UTC", errors="coerce", utc=True,
        ).dt.tz_localize(None)
        frame = frame[
            frame.valid_time.notna() & frame.observed_t2m_c.between(-90, 60)
        ].sort_values("valid_time").copy()
        counts["physical_rows"] += len(frame)
        rolling = frame.observed_t2m_c.rolling(5, center=True, min_periods=3).median()
        frame = frame[(frame.observed_t2m_c - rolling).abs() <= 4.0].copy()
        counts["local_qc_rows"] += len(frame)
        frame = frame[frame.valid_time.dt.hour.isin([0, 12])].copy()
        counts["nominal_00_12_rows"] += len(frame)
        frame["station_id"] = station_id
        observations.append(frame[["station_id", "valid_time", "observed_t2m_c"]])

    station_frame = pd.DataFrame(stations)
    observation_frame = pd.concat(observations, ignore_index=True)
    if station_frame.station_id.duplicated().any():
        duplicates = station_frame.loc[station_frame.station_id.duplicated(False), "station_id"].tolist()
        raise ValueError(f"duplicate native station identifiers: {duplicates[:10]}")
    if observation_frame.duplicated(["station_id", "valid_time"]).any():
        raise ValueError("duplicate INMET station-time rows")
    coverage = observation_frame.groupby("station_id").size()
    eligible = coverage[coverage >= min_samples]
    selected = station_frame[station_frame.station_id.isin(eligible.index)].copy()
    selected["n_2020_nominal_00_12"] = selected.station_id.map(eligible).astype(int)
    filtered = observation_frame[observation_frame.station_id.isin(eligible.index)].copy()
    counts.update({
        "source_files": len(files),
        "stations_in_archive": len(station_frame),
        "stations_meeting_min_samples": len(selected),
        "rows_after_station_coverage": len(filtered),
        "min_nominal_00_12_samples": min_samples,
    })
    return station_frame, selected, filtered, counts


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

    archive, extracted = download(args.raw)
    all_stations, stations, observations, report = prepare(extracted, args.min_samples)
    args.output.mkdir(parents=True, exist_ok=True)
    all_stations.to_csv(args.output / "stations_all.csv", index=False)
    stations.to_csv(args.output / "stations.csv", index=False)
    connection = duckdb.connect()
    connection.register("observations", observations)
    connection.execute(
        f"COPY observations TO '{(args.output / 'observations.parquet').as_posix()}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)")
    connection.close()
    report.update({
        "role": "national automatic network candidate; assimilation route unresolved",
        "source_url": SOURCE_URL,
        "archive": {"bytes": archive.stat().st_size, "sha256": sha256(archive)},
        "timestamp_semantics": "INMET field Hora UTC; exact 00 and 12 UTC retained",
        "qc": "physical range -90 to 60 C plus centred five-report median deviation <=4 C",
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in outputs[:-1]
        },
    })
    (args.output / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
