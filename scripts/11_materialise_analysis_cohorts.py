#!/usr/bin/env python3
"""Create immutable, protocol-specific analysis panels from model extensions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "regional_t2m_2020.yaml"


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    con = duckdb.connect()
    con.register("frame", frame)
    con.execute(f"COPY frame TO '{path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cohort", nargs="?", help="configured cohort; omit to materialise all")
    parser.add_argument("--force", action="store_true", help="replace an existing derived panel")
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cohorts = cfg["analysis_cohorts"]
    selected = [args.cohort] if args.cohort else list(cohorts)
    unknown = set(selected).difference(cohorts)
    if unknown:
        raise ValueError(f"unknown cohort(s): {sorted(unknown)}")

    base_path = ROOT / cfg["paths"]["annual_aligned_parquet"]
    extension_dir = ROOT / cfg["paths"]["model_extension_directory"]
    output_dir = ROOT / "data" / "interim" / "analysis_cohorts"
    output_dir.mkdir(parents=True, exist_ok=True)
    availability = ROOT / cfg["paths"]["availability_directory"]
    base = duckdb.sql(f"SELECT * FROM read_parquet('{base_path.as_posix()}')").fetchdf()
    key = ["station_id", "init_time", "valid_time"]
    if base.duplicated(key).any():
        raise RuntimeError("base panel has duplicate station-initialisation rows")

    for name in selected:
        spec = cohorts[name]
        output = output_dir / f"{name}.parquet"
        manifest_path = availability / f"analysis_cohort_{name}.json"
        if output.exists() and not args.force:
            print(f"already materialised: {output}")
            continue
        panel = base.copy()
        for extension in spec["extensions"]:
            extension_path = extension_dir / f"{extension}.parquet"
            if not extension_path.exists():
                raise FileNotFoundError(f"missing extension: {extension_path}")
            extra = duckdb.sql(f"SELECT * FROM read_parquet('{extension_path.as_posix()}')").fetchdf()
            value = f"{extension}_t2m_c"
            if set(extra.columns) != {*key, value}:
                raise RuntimeError(f"unexpected columns in {extension_path.name}: {list(extra.columns)}")
            if len(extra) != len(base) or extra.duplicated(key).any():
                raise RuntimeError(f"extension {extension} does not have a one-to-one fixed panel")
            panel = panel.merge(extra, on=key, how="inner", validate="one_to_one")
        value_columns = [f"{model}_t2m_c" for model in spec["models"]]
        missing = [column for column in value_columns if column not in panel]
        if missing:
            raise RuntimeError(f"cohort {name} misses model columns {missing}")
        if len(panel) != len(base) or panel[value_columns].isna().any().any():
            raise RuntimeError(f"cohort {name} changed the fixed panel or has missing forecasts")
        write_parquet(panel, output)
        manifest = {
            "pilot_name": cfg["pilot_name"],
            "cohort": name,
            "label": spec["label"],
            "models": spec["models"],
            "extensions": spec["extensions"],
            "cases": len(panel),
            "stations": int(panel.station_id.nunique()),
            "initialisations": int(panel.init_time.nunique()),
            "base_panel_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
            "output": str(output.relative_to(ROOT)),
            "status": "fixed-panel cohort materialised; analyse only within this protocol",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
