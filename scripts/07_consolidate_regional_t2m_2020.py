#!/usr/bin/env python3
"""Consolidate the four materialised regional batches and audit completeness."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "regional_t2m_2020.yaml"
BATCHES = ("q1", "q2", "q3", "q4")
MODELS = ("ifs_hres", "graphcast_hres_init", "pangu_hres_init")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="replace prior annual parquet")
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    batch_dir = ROOT / cfg["paths"]["batch_directory"]
    availability_dir = ROOT / cfg["paths"]["availability_directory"]
    missing = [batch for batch in BATCHES if not (batch_dir / f"{batch}.parquet").exists() or not (availability_dir / f"{batch}.json").exists()]
    if missing:
        raise RuntimeError(f"missing materialised batches or manifests: {missing}")
    output = ROOT / cfg["paths"]["annual_aligned_parquet"]
    if output.exists() and not args.force:
        raise RuntimeError(f"annual output exists: {output}; use --force to rebuild")
    output.parent.mkdir(parents=True, exist_ok=True)
    source = f"{batch_dir.as_posix()}/*.parquet"
    con = duckdb.connect()
    audit = con.execute(
        f"""SELECT
              count(*) AS case_count,
              count(DISTINCT station_id) AS station_count,
              count(DISTINCT init_time) AS initialisation_count,
              count(DISTINCT valid_time) AS valid_time_count,
              count(*) - count(DISTINCT (station_id, init_time)) AS duplicate_case_count,
              count(*) FILTER (WHERE era5_t2m_c IS NULL) AS missing_era5,
              count(*) FILTER (WHERE avamet_t2m_qc_valid) AS avamet_qc_valid,
              {", ".join(f"count(*) FILTER (WHERE {model}_t2m_c IS NULL) AS missing_{model}" for model in MODELS)}
            FROM read_parquet('{source}')"""
    ).fetchone()
    columns = [description[0] for description in con.description]
    summary = dict(zip(columns, audit, strict=True))
    expected_cases = sum(json.loads((availability_dir / f"{batch}.json").read_text(encoding="utf-8"))["expected_cases"] for batch in BATCHES)
    failures = []
    if summary["case_count"] != expected_cases:
        failures.append(f"case count {summary['case_count']} != expected {expected_cases}")
    if summary["duplicate_case_count"]:
        failures.append(f"duplicate station-initialisation cases: {summary['duplicate_case_count']}")
    missing_columns = [name for name, value in summary.items() if name.startswith("missing_") and value]
    if missing_columns:
        failures.append(f"missing forecast/reference values: {missing_columns}")
    if failures:
        raise RuntimeError("; ".join(failures))
    con.execute(f"COPY (SELECT * FROM read_parquet('{source}') ORDER BY init_time, station_id) TO '{output.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    common = con.execute(
        f"""SELECT count(*) FROM read_parquet('{output.as_posix()}')
            WHERE avamet_t2m_qc_c IS NOT NULL
              AND era5_t2m_c IS NOT NULL
              AND {' AND '.join(f'{model}_t2m_c IS NOT NULL' for model in MODELS)}"""
    ).fetchone()[0]
    summary.update({
        "pilot_name": cfg["pilot_name"], "batches": list(BATCHES), "expected_cases": expected_cases,
        "same_case_intersection_both_references": common,
        "output": str(output.relative_to(ROOT)), "status": "consolidated and audited; ready for pre-specified regional scoring",
    })
    (availability_dir / "annual.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
