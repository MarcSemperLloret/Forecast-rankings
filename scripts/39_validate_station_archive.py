#!/usr/bin/env python3
"""Run the archive checks over a station table and report what fails.

Written now, against the pilot's own tables, so that it is already working and
already tested when the global archive lands. It takes any table with
station_id, latitude, longitude and optionally altitude_m and network, so the
same command serves AVAMET today and GHCN plus eight other networks later.

Exit code is non-zero if any check fails, so it can gate a pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from validation import (check_coordinates, check_elevation, check_temperature_units, check_timestamps,
                        find_duplicate_stations, summarise)


def load(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        # Station identifiers are labels, not numbers.  Explicit string types
        # preserve NOAA/ISD leading zeroes across CSV round trips.
        header = pd.read_csv(path, nrows=0)
        string_columns = {
            name: "string" for name in ("station_id", "site_identifier", "site_id")
            if name in header.columns
        }
        return pd.read_csv(path, dtype=string_columns)
    return duckdb.sql(f"SELECT * FROM read_parquet('{path.as_posix()}')").fetchdf()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stations", type=Path, help="station table with station_id, latitude, longitude")
    parser.add_argument("--observations", type=Path, default=None,
                        help="optional observation table with a timestamp and a temperature column")
    parser.add_argument("--timestamp-column", default="observed_utc")
    parser.add_argument("--temperature-column", default="temperature_c")
    parser.add_argument("--series-column", default="station_id",
                        help="comma-separated series keys used with time for uniqueness")
    parser.add_argument("--expected-hours", default=None, help="comma-separated UTC hours the archive should contain")
    parser.add_argument("--duplicate-distance-km", type=float, default=0.5)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    stations = load(args.stations)
    findings = [*check_coordinates(stations), *check_elevation(stations)]

    if args.observations is not None:
        observations = load(args.observations)
        expected = {int(hour) for hour in args.expected_hours.split(",")} if args.expected_hours else None
        series_columns = [name.strip() for name in args.series_column.split(",") if name.strip()]
        if series_columns and all(name in observations for name in series_columns):
            series_ids = observations[series_columns].astype(str).agg("|".join, axis=1)
        else:
            series_ids = None
        findings.extend(check_timestamps(observations[args.timestamp_column], expected, series_ids))
        findings.extend(check_temperature_units(observations[args.temperature_column]))

    duplicates = find_duplicate_stations(stations, args.duplicate_distance_km)
    same_site = duplicates[duplicates.same_site] if not duplicates.empty else duplicates
    cross_network = same_site[same_site.network_a != same_site.network_b] if not same_site.empty else same_site
    report = summarise(findings)
    report["duplicates"] = {"candidate_pairs": int(len(duplicates)), "same_site_pairs": int(len(same_site)),
                            "cross_network_pairs": int(len(cross_network)),
                            "distance_threshold_km": args.duplicate_distance_km}
    report["stations"] = int(len(stations))
    report["source"] = str(args.stations)

    for item in findings:
        print(item)
    print(f"[{'WARN' if len(same_site) else 'PASS':7}] duplicates: {len(same_site)} same-site pairs "
          f"({len(cross_network)} across networks) within {args.duplicate_distance_km} km")
    if not same_site.empty:
        print(same_site.head(20).to_string(index=False))

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not duplicates.empty:
            duplicates.to_csv(args.report.with_name(args.report.stem + "_duplicates.csv"), index=False)
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
