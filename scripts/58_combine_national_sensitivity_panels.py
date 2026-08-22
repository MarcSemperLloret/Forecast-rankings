#!/usr/bin/env python3
"""Combine completed national sensitivity panels without creating new model units."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = [
    ROOT / "data" / "interim" / "midas_unknown_weatherbench2_2020" / "panel.parquet",
    ROOT / "data" / "interim" / "inmet_unknown_weatherbench2_2020" / "panel.parquet",
]
DEFAULT_OUTPUT = ROOT / "data" / "interim" / "national_unknown_combined_weatherbench2_2020"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "panel.parquet"
    manifest_path = args.output / "manifest.json"
    if (target.exists() or manifest_path.exists()) and not args.force:
        raise FileExistsError(f"outputs exist in {args.output}; use --force")
    inputs = [path.resolve() for path in args.inputs]
    missing = [path for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing panel input: {missing[0]}")
    source_sql = "[" + ",".join(f"'{path.as_posix()}'" for path in inputs) + "]"
    connection = duckdb.connect()
    columns = connection.execute(
        f"DESCRIBE SELECT * FROM read_parquet({source_sql}, union_by_name=true)"
    ).fetchdf().column_name.tolist()
    required = {
        "station_id", "latitude", "longitude", "altitude_m", "network", "valid_time",
        "model", "model_family", "cohort", "forecast_t2m_c", "era5_t2m_c", "observed_t2m_c",
    }
    if missing_columns := required.difference(columns):
        raise ValueError(f"combined panel lacks columns: {sorted(missing_columns)}")
    connection.execute(
        f"""
        COPY (
            SELECT * FROM read_parquet({source_sql}, union_by_name=true)
            ORDER BY model, valid_time, network, station_id
        ) TO '{target.resolve().as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    counts = connection.execute(
        f"""
        SELECT count(*), count(DISTINCT station_id), count(DISTINCT network),
               count(DISTINCT model), count(DISTINCT valid_time),
               count(*) - count(DISTINCT (station_id, valid_time, model)),
               count(*) FILTER (WHERE forecast_t2m_c IS NULL OR era5_t2m_c IS NULL
                                      OR observed_t2m_c IS NULL)
        FROM read_parquet('{target.resolve().as_posix()}')
        """
    ).fetchone()
    by_network = connection.execute(
        f"""
        SELECT network, count(DISTINCT station_id) AS stations, count(*) AS rows
        FROM read_parquet('{target.resolve().as_posix()}')
        GROUP BY network ORDER BY network
        """
    ).fetchdf().to_dict(orient="records")
    connection.close()
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "combined national-network sensitivity; not an additional independent replication",
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in inputs],
        "counts": {
            "rows": int(counts[0]),
            "stations": int(counts[1]),
            "networks": int(counts[2]),
            "source_variants": int(counts[3]),
            "valid_times": int(counts[4]),
            "duplicate_station_time_model_rows": int(counts[5]),
            "missing_required_rows": int(counts[6]),
        },
        "by_network": by_network,
        "output": {"path": str(target), "bytes": target.stat().st_size, "sha256": sha256(target)},
        "validation_passed": counts[5] == 0 and counts[6] == 0,
        "interpretation_limit": (
            "MIDAS and INMET share model fields, ERA5 reference, year and analysis pipeline. "
            "Combining them changes spatial support but does not create an independent replicate."
        ),
    }
    manifest_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["validation_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
