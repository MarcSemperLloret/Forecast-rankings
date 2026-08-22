#!/usr/bin/env python3
"""Harmonise WEATHER-5K temperature observations for an engineering dry run.

The source is scanned once into a DuckDB staging table.  The output contains
only exact 00/12 UTC TMP observations: TMP's own MASK flag must be one and
TIME_DIFF must be exactly zero.  This script intentionally labels the result
as a dry run because NOAA ISD observations may have entered reanalysis data
assimilation and therefore are not automatically independent confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "weather5k_dry_run_2020.yaml"


def resolve_from_root(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sql_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def source_sql(paths: list[Path]) -> str:
    if len(paths) == 1:
        return sql_string(paths[0].as_posix())
    return "[" + ", ".join(sql_string(path.as_posix()) for path in paths) + "]"


def remove_previous_outputs(output_root: Path) -> None:
    names = [
        "weather5k_harmonize.duckdb",
        "observations.parquet",
        "stations.csv",
        "coverage.csv",
        "manifest.json",
    ]
    for name in names:
        target = output_root / name
        if target.exists():
            if not target.is_file():
                raise RuntimeError(f"Refusing to replace non-file output: {target}")
            target.unlink()


def load_selected_metadata(
    station_index: Path, source_root: Path, sample_files: int | None
) -> tuple[pd.DataFrame, list[Path]]:
    metadata = pd.read_csv(
        station_index,
        dtype={"station_id": "string", "isd_station_id": "string"},
    )
    required = {
        "station_id",
        "isd_station_id",
        "latitude",
        "longitude",
        "elevation_m",
        "relative_path",
    }
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Station index lacks columns: {sorted(missing)}")
    metadata = metadata.sort_values("station_id").reset_index(drop=True)
    if sample_files is not None:
        if sample_files < 1:
            raise ValueError("--sample-files must be at least 1")
        metadata = metadata.head(sample_files).copy()
    paths = [source_root / f"{station_id}.csv" for station_id in metadata.station_id]
    absent = [path for path in paths if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"Missing {len(absent)} source CSVs; first: {absent[0]}")
    metadata["station_id"] = metadata.station_id.astype(str)
    metadata["isd_station_id"] = metadata.isd_station_id.astype(str)
    return metadata, paths


def scalar(connection: duckdb.DuckDBPyConnection, query: str) -> Any:
    return connection.execute(query).fetchone()[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override the configured output directory (required for disposable samples).",
    )
    parser.add_argument(
        "--sample-files", type=int, default=None, help="Process the first N station files only."
    )
    parser.add_argument("--force", action="store_true", help="Replace this script's known outputs.")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_cfg = config["source"]
    period = config["period"]
    qc = config["quality_control"]
    temperature = config["temperature"]
    output_cfg = config["output"]

    source_root = resolve_from_root(source_cfg["station_csv_root"])
    station_index = resolve_from_root(source_cfg["station_index"])
    file_index = resolve_from_root(source_cfg["file_index"])
    metadata_json = resolve_from_root(source_cfg["metadata_json"])
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else resolve_from_root(output_cfg["root"])
    )
    output_root.mkdir(parents=True, exist_ok=True)
    known_outputs = [
        output_root / "weather5k_harmonize.duckdb",
        output_root / "observations.parquet",
        output_root / "stations.csv",
        output_root / "coverage.csv",
        output_root / "manifest.json",
    ]
    if any(path.exists() for path in known_outputs):
        if not args.force:
            raise FileExistsError(f"Output exists in {output_root}; use --force to replace it")
        remove_previous_outputs(output_root)

    metadata, paths = load_selected_metadata(station_index, source_root, args.sample_files)
    selected_bytes = sum(path.stat().st_size for path in paths)
    year = int(period["year"])
    cycles = [int(hour) for hour in period["cycles_utc"]]
    if any(hour < 0 or hour > 23 for hour in cycles):
        raise ValueError(f"Invalid UTC cycles: {cycles}")
    cycle_sql = ", ".join(str(hour) for hour in cycles)
    plausible_low, plausible_high = map(float, temperature["plausible_c"])
    lat_low, lat_high = map(float, qc["valid_latitude"])
    lon_low, lon_high = map(float, qc["valid_longitude"])
    elev_low, elev_high = map(float, qc["valid_elevation_m"])
    min_samples = int(qc["min_exact_utc_samples"])
    required_mask = str(int(qc["require_tmp_mask_value"]))
    required_time_diff = float(qc["require_time_diff_hours"])
    rolling_window = int(qc["rolling_window_reports"])
    rolling_min_valid = int(qc["rolling_min_valid_reports"])
    local_deviation = float(qc["local_median_deviation_c"])
    if rolling_window != 5:
        raise ValueError("This audited implementation is fixed to a centred five-report window")
    compression = str(output_cfg["parquet_compression"])

    print(
        f"Scanning {len(paths):,} files ({selected_bytes / 1024**3:.2f} GiB) for "
        f"{year} at UTC hours {cycles}...",
        flush=True,
    )
    database = output_root / "weather5k_harmonize.duckdb"
    connection = duckdb.connect(str(database))
    connection.execute("PRAGMA threads=4")
    connection.register("metadata_frame", metadata)
    connection.execute("CREATE TABLE station_metadata AS SELECT * FROM metadata_frame")

    columns = """{
        'DATE': 'VARCHAR', 'LONGITUDE': 'DOUBLE', 'LATITUDE': 'DOUBLE',
        'TMP': 'DOUBLE', 'DEW': 'DOUBLE', 'WND_ANGLE': 'DOUBLE',
        'WND_RATE': 'DOUBLE', 'SLP': 'DOUBLE', 'MASK': 'VARCHAR',
        'TIME_DIFF': 'DOUBLE'
    }"""
    connection.execute(
        f"""
        CREATE TABLE hourly_rows AS
        SELECT
            parse_filename(filename, true) AS station_id,
            try_cast(DATE AS TIMESTAMP) AS valid_time,
            TMP AS observed_t2m_c,
            MASK AS source_mask,
            TIME_DIFF AS source_time_diff_hours
        FROM read_csv(
            {source_sql(paths)},
            columns={columns},
            header=true,
            filename=true,
            parallel=true
        )
        WHERE DATE >= '{year - 1}-12-29 00:00:00'
          AND DATE < '{year + 1}-01-04 00:00:00'
        """
    )
    mask_expression = (
        f"regexp_extract(source_mask, '^\\[\\s*([01])', 1) = '{required_mask}'"
    )
    exact_expression = f"source_time_diff_hours = {required_time_diff}"
    plausible_expression = (
        f"observed_t2m_c BETWEEN {plausible_low} AND {plausible_high}"
    )
    connection.execute(
        f"""
        CREATE TABLE cycle_rows AS
        WITH flagged AS (
            SELECT *,
                   {mask_expression} AS tmp_mask_observed,
                   {exact_expression} AS exact_time,
                   {plausible_expression} AS temperature_plausible
            FROM hourly_rows
        ), valid_reports AS (
            SELECT station_id, valid_time,
                   median(observed_t2m_c) OVER (
                       PARTITION BY station_id ORDER BY valid_time
                       ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING
                   ) AS rolling_median_t2m_c,
                   count(observed_t2m_c) OVER (
                       PARTITION BY station_id ORDER BY valid_time
                       ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING
                   ) AS rolling_window_valid_reports
            FROM flagged
            WHERE tmp_mask_observed AND exact_time AND temperature_plausible
        ), cycles AS (
            SELECT f.*, r.rolling_median_t2m_c, r.rolling_window_valid_reports
            FROM flagged f
            LEFT JOIN valid_reports r USING (station_id, valid_time)
            WHERE f.valid_time >= TIMESTAMP '{year}-01-01 00:00:00'
              AND f.valid_time < TIMESTAMP '{year + 1}-01-01 00:00:00'
              AND hour(f.valid_time) IN ({cycle_sql})
        )
        SELECT *,
               (
                   tmp_mask_observed AND exact_time AND temperature_plausible
                   AND rolling_window_valid_reports >= {rolling_min_valid}
                   AND abs(observed_t2m_c - rolling_median_t2m_c) <= {local_deviation}
               ) AS source_qc_valid
        FROM cycles
        """
    )
    connection.execute(
        f"""
        CREATE TABLE coverage AS
        SELECT
            m.station_id,
            m.isd_station_id,
            m.latitude,
            m.longitude,
            m.elevation_m AS altitude_m,
            count(c.valid_time) AS n_cycle_rows,
            count(c.valid_time) FILTER (WHERE {mask_expression}) AS n_tmp_observed_mask,
            count(c.valid_time) FILTER (
                WHERE {mask_expression} AND {exact_expression}
            ) AS n_exact_time_rows,
            count(c.valid_time) FILTER (WHERE c.source_qc_valid) AS n_usable_rows,
            count(c.valid_time) FILTER (
                WHERE NOT ({mask_expression}) OR source_mask IS NULL
            ) AS n_rejected_tmp_mask,
            count(c.valid_time) FILTER (
                WHERE {mask_expression} AND NOT ({exact_expression})
            ) AS n_rejected_time_diff,
            count(c.valid_time) FILTER (
                WHERE {mask_expression} AND {exact_expression}
                  AND NOT ({plausible_expression})
            ) AS n_rejected_temperature,
            count(c.valid_time) FILTER (
                WHERE {mask_expression} AND {exact_expression} AND {plausible_expression}
                  AND NOT c.source_qc_valid
            ) AS n_rejected_local_median,
            (
                m.latitude BETWEEN {lat_low} AND {lat_high}
                AND m.longitude BETWEEN {lon_low} AND {lon_high}
                AND m.elevation_m BETWEEN {elev_low} AND {elev_high}
            ) AS metadata_valid
        FROM station_metadata m
        LEFT JOIN cycle_rows c USING (station_id)
        GROUP BY ALL
        """
    )
    connection.execute(
        f"""
        CREATE TABLE eligible_observations AS
        SELECT
            c.station_id,
            v.isd_station_id AS site_identifier,
            v.latitude,
            v.longitude,
            v.altitude_m,
            '{output_cfg['network']}' AS network,
            {str(bool(output_cfg['potentially_assimilated'])).lower()}
                AS potentially_assimilated,
            c.valid_time,
            c.observed_t2m_c,
            c.source_mask,
            c.source_time_diff_hours,
            c.rolling_median_t2m_c,
            c.rolling_window_valid_reports,
            c.source_qc_valid,
            '{source_cfg['dataset']}' AS source_dataset,
            'TMP_MASK=1;TIME_DIFF=0;RANGE;CENTERED_MEDIAN_5_MAXDEV_4C' AS qc_contract
        FROM cycle_rows c
        JOIN coverage v USING (station_id)
        WHERE v.metadata_valid
          AND v.n_usable_rows >= {min_samples}
          AND c.source_qc_valid
        ORDER BY c.station_id, c.valid_time
        """
    )

    observations_path = output_root / "observations.parquet"
    stations_path = output_root / "stations.csv"
    coverage_path = output_root / "coverage.csv"
    connection.execute(
        f"COPY eligible_observations TO {sql_string(observations_path.as_posix())} "
        f"(FORMAT PARQUET, COMPRESSION {compression})"
    )
    connection.execute(
        f"""
        COPY (
            SELECT DISTINCT station_id, site_identifier, latitude, longitude,
                   altitude_m, network, potentially_assimilated, source_dataset
            FROM eligible_observations ORDER BY station_id
        ) TO {sql_string(stations_path.as_posix())} (HEADER, DELIMITER ',')
        """
    )
    connection.execute(
        f"COPY (SELECT * FROM coverage ORDER BY station_id) TO "
        f"{sql_string(coverage_path.as_posix())} (HEADER, DELIMITER ',')"
    )

    duplicate_groups = scalar(
        connection,
        """SELECT count(*) FROM (
               SELECT station_id, valid_time FROM eligible_observations
               GROUP BY station_id, valid_time HAVING count(*) > 1
           )""",
    )
    invalid_hours = scalar(
        connection,
        f"SELECT count(*) FROM eligible_observations WHERE hour(valid_time) NOT IN ({cycle_sql})",
    )
    invalid_years = scalar(
        connection,
        f"SELECT count(*) FROM eligible_observations WHERE year(valid_time) <> {year}",
    )
    summary = connection.execute(
        """
        SELECT
            (SELECT count(*) FROM cycle_rows) AS cycle_rows,
            (SELECT count(*) FROM eligible_observations) AS output_rows,
            (SELECT count(DISTINCT station_id) FROM eligible_observations) AS output_stations,
            (SELECT min(valid_time) FROM eligible_observations) AS first_valid_time,
            (SELECT max(valid_time) FROM eligible_observations) AS last_valid_time,
            (SELECT min(observed_t2m_c) FROM eligible_observations) AS min_t2m_c,
            (SELECT max(observed_t2m_c) FROM eligible_observations) AS max_t2m_c
        """
    ).fetchone()
    connection.close()

    validations = {
        "duplicate_station_time_groups": int(duplicate_groups),
        "rows_with_invalid_utc_hour": int(invalid_hours),
        "rows_outside_target_year": int(invalid_years),
        "passed": duplicate_groups == 0 and invalid_hours == 0 and invalid_years == 0,
    }
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": config["analysis_role"],
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "source": {
            "dataset": source_cfg["dataset"],
            "root": str(source_root),
            "selected_files": len(paths),
            "selected_bytes": selected_bytes,
            "selection": (
                f"first_{args.sample_files}_station_ids_sorted"
                if args.sample_files is not None
                else "all_station_ids_in_station_index"
            ),
            "fingerprint_scope": "index hashes and aggregate file size; source CSV contents not hashed",
            "station_index": str(station_index),
            "station_index_sha256": sha256(station_index),
            "file_index": str(file_index),
            "file_index_sha256": sha256(file_index),
            "metadata_json": str(metadata_json),
            "metadata_json_sha256": sha256(metadata_json),
            "upstream_repository": source_cfg["upstream_repository"],
            "upstream_paper": source_cfg["upstream_paper"],
            "dataset_license": source_cfg["dataset_license"],
            "underlying_source": source_cfg["underlying_source"],
        },
        "filter_contract": {
            "year": year,
            "cycles_utc": cycles,
            "tmp_mask_value": int(required_mask),
            "time_diff_hours": required_time_diff,
            "plausible_t2m_c": [plausible_low, plausible_high],
            "rolling_window_reports": rolling_window,
            "rolling_min_valid_reports": rolling_min_valid,
            "local_median_deviation_c": local_deviation,
            "minimum_usable_rows_per_station": min_samples,
        },
        "counts": {
            "cycle_rows": int(summary[0]),
            "output_rows": int(summary[1]),
            "output_stations": int(summary[2]),
        },
        "bounds": {
            "first_valid_time": summary[3].isoformat() if summary[3] else None,
            "last_valid_time": summary[4].isoformat() if summary[4] else None,
            "min_t2m_c": float(summary[5]) if summary[5] is not None else None,
            "max_t2m_c": float(summary[6]) if summary[6] is not None else None,
        },
        "validations": validations,
        "outputs": {
            path.name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (observations_path, stations_path, coverage_path, database)
        },
        "interpretation_warning": (
            "Engineering dry run only. ISD observations may overlap data assimilated by reanalyses; "
            "do not treat this output as independent confirmatory evidence."
        ),
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"Wrote {summary[1]:,} rows for {summary[2]:,} stations to {observations_path}",
        flush=True,
    )
    print(f"Validation passed: {validations['passed']}", flush=True)
    if not validations["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
