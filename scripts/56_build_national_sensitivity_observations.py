#!/usr/bin/env python3
"""Build a traceable national-network observation stratum after risk audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--stations", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--risk-class", default="unknown")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "observations.parquet"
    manifest_path = args.output / "manifest.json"
    if (target.exists() or manifest_path.exists()) and not args.force:
        raise FileExistsError(f"outputs exist in {args.output}; use --force")

    stations = pd.read_csv(args.stations, dtype={"station_id": "string"})
    audit = pd.read_csv(args.audit, dtype={"station_id": "string"})
    required_station = {"station_id", "latitude", "longitude", "altitude_m", "network"}
    required_audit = {"station_id", "assimilation_risk_class"}
    if missing := required_station.difference(stations.columns):
        raise ValueError(f"station metadata missing columns: {sorted(missing)}")
    if missing := required_audit.difference(audit.columns):
        raise ValueError(f"audit missing columns: {sorted(missing)}")
    selected_audit = audit[audit["assimilation_risk_class"] == args.risk_class].copy()
    if selected_audit.empty:
        raise ValueError(f"no stations have risk class {args.risk_class!r}")
    metadata = stations.merge(
        selected_audit[["station_id", "assimilation_risk_class"]],
        on="station_id",
        validate="one_to_one",
    )
    metadata["potentially_assimilated"] = metadata["assimilation_risk_class"].eq(
        "potentially_assimilated"
    )
    connection = duckdb.connect()
    connection.register("selected_metadata", metadata)
    observation_path = args.observations.resolve().as_posix()
    target_path = target.resolve().as_posix()
    connection.execute(
        f"""
        COPY (
            SELECT o.station_id, m.latitude, m.longitude, m.altitude_m, m.network,
                   m.potentially_assimilated, m.assimilation_risk_class,
                   o.valid_time, o.observed_t2m_c
            FROM read_parquet('{observation_path}') o
            JOIN selected_metadata m USING (station_id)
            ORDER BY valid_time, station_id
        ) TO '{target_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    counts = connection.execute(
        f"""
        SELECT count(*), count(DISTINCT station_id), count(DISTINCT valid_time),
               min(valid_time), max(valid_time),
               count(*) - count(DISTINCT (station_id, valid_time)),
               count(*) FILTER (WHERE observed_t2m_c IS NULL OR
                                latitude IS NULL OR longitude IS NULL)
        FROM read_parquet('{target_path}')
        """
    ).fetchone()
    connection.close()
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "national-network sensitivity; not proof of ERA5 independence",
        "selected_assimilation_risk_class": args.risk_class,
        "inputs": {
            "observations": {"path": str(args.observations), "sha256": sha256(args.observations)},
            "stations": {"path": str(args.stations), "sha256": sha256(args.stations)},
            "audit": {"path": str(args.audit), "sha256": sha256(args.audit)},
        },
        "counts": {
            "rows": int(counts[0]),
            "stations": int(counts[1]),
            "valid_times": int(counts[2]),
            "first_valid_time": counts[3].isoformat(),
            "last_valid_time": counts[4].isoformat(),
            "duplicate_station_time_rows": int(counts[5]),
            "missing_required_rows": int(counts[6]),
        },
        "output": {"path": str(target), "bytes": target.stat().st_size, "sha256": sha256(target)},
        "validation_passed": counts[5] == 0 and counts[6] == 0,
        "interpretation_limit": (
            "No Weather5K/ISD overlap was detected for this stratum, but absence from all ERA5 "
            "assimilation routes has not been documented."
        ),
    }
    manifest_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["validation_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
