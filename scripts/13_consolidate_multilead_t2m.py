#!/usr/bin/env python3
"""Consolidate and audit one materialised additional T2m lead time."""
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
    parser.add_argument("lead", type=int, help="materialised additional lead time")
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if args.lead not in cfg["lead_time_extension"]["additional_leads_hours"]:
        raise ValueError(f"lead must be one of {cfg['lead_time_extension']['additional_leads_hours']}")
    base_dir = ROOT / cfg["paths"]["lead_time_directory"] / f"lead={args.lead:03d}"
    batch_dir, availability_dir = base_dir / "batches", base_dir / "availability"
    missing = [batch for batch in BATCHES if not (batch_dir / f"{batch}.parquet").exists() or not (availability_dir / f"{batch}.json").exists()]
    if missing:
        raise RuntimeError(f"lead={args.lead} lacks batches/manifests: {missing}")
    output = base_dir / "annual.parquet"
    source = f"{batch_dir.as_posix()}/*.parquet"
    con = duckdb.connect()
    audit = con.execute(
        f"""SELECT count(*) AS case_count, count(DISTINCT station_id) AS station_count,
                   count(DISTINCT init_time) AS initialisation_count, count(DISTINCT valid_time) AS valid_time_count,
                   count(*) - count(DISTINCT (station_id, init_time)) AS duplicate_case_count,
                   count(*) FILTER (WHERE era5_t2m_c IS NULL) AS missing_era5,
                   count(*) FILTER (WHERE avamet_t2m_qc_valid) AS avamet_qc_valid,
                   {", ".join(f"count(*) FILTER (WHERE {model}_t2m_c IS NULL) AS missing_{model}" for model in MODELS)}
            FROM read_parquet('{source}')"""
    ).fetchone()
    summary = dict(zip([item[0] for item in con.description], audit, strict=True))
    expected = sum(json.loads((availability_dir / f"{batch}.json").read_text(encoding="utf-8"))["expected_cases"] for batch in BATCHES)
    failures = []
    if summary["case_count"] != expected:
        failures.append("case count does not match manifests")
    if summary["duplicate_case_count"]:
        failures.append("duplicate station-initialisation cases")
    if any(value for key, value in summary.items() if key.startswith("missing_")):
        failures.append("missing forecast/reference values")
    if failures:
        raise RuntimeError("; ".join(failures))
    con.execute(f"COPY (SELECT * FROM read_parquet('{source}') ORDER BY init_time, station_id) TO '{output.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    common = con.execute(
        f"""SELECT count(*) FROM read_parquet('{output.as_posix()}')
              WHERE avamet_t2m_qc_c IS NOT NULL AND era5_t2m_c IS NOT NULL
                AND {' AND '.join(f'{model}_t2m_c IS NOT NULL' for model in MODELS)}"""
    ).fetchone()[0]
    summary.update({"pilot_name": cfg["pilot_name"], "lead_hours": args.lead, "batches": list(BATCHES), "expected_cases": expected,
                    "same_case_intersection_both_references": common, "output": str(output.relative_to(ROOT)),
                    "status": "consolidated and audited; ready for lead-time robustness analysis"})
    (availability_dir / "annual.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
