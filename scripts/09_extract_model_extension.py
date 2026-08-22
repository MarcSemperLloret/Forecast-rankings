#!/usr/bin/env python3
"""Extract one pre-specified extra model on the fixed regional 2020 panel."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from public_zarr import PublicZarr, bilinear

CONFIG = ROOT / "config" / "regional_t2m_2020.yaml"
CACHE = ROOT / "data" / "raw" / "weatherbench2_cache"


def index_of(values: np.ndarray | pd.DatetimeIndex, wanted: object, label: str) -> int:
    indices = np.flatnonzero(np.asarray(values) == wanted)
    if len(indices) != 1:
        raise RuntimeError(f"{label}={wanted} is missing or non-unique")
    return int(indices[0])


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    con = duckdb.connect()
    con.register("frame", frame)
    con.execute(f"COPY frame TO '{path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="pre-specified key from model_extensions")
    parser.add_argument("--force", action="store_true", help="replace an existing extension")
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    extensions = cfg["model_extensions"]
    if args.model not in extensions:
        raise ValueError(f"unknown extension {args.model!r}; choose {sorted(extensions)}")
    spec = extensions[args.model]
    base_path = ROOT / cfg["paths"]["annual_aligned_parquet"]
    output_dir = ROOT / cfg["paths"]["model_extension_directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{args.model}.parquet"
    manifest_path = ROOT / cfg["paths"]["availability_directory"] / f"extension_{args.model}.json"
    if output.exists() and not args.force:
        print(f"already materialised: {output}")
        return
    base = duckdb.sql(f"SELECT station_id, init_time, valid_time, latitude, longitude FROM read_parquet('{base_path.as_posix()}') ORDER BY init_time, station_id").fetchdf()
    stations = base[["station_id", "latitude", "longitude"]].drop_duplicates().sort_values("station_id").reset_index(drop=True)
    if base.duplicated(["station_id", "init_time"]).any():
        raise RuntimeError("the base panel contains duplicate station-initialisation cases")
    initialisations = pd.DatetimeIndex(pd.to_datetime(base.init_time.unique())).sort_values()
    store = PublicZarr(spec["path"], CACHE)
    times = store.times(spec["time_name"])
    missing_times = initialisations.difference(times)
    if not missing_times.empty:
        raise RuntimeError(f"{args.model} misses {len(missing_times)} base initialisations")
    leads = store.timedeltas_hours(spec["lead_name"])
    lead = cfg["leads_hours"][0]
    lead_index = index_of(leads, lead, "lead")
    latitudes, longitudes = store.coordinate(spec["latitude_name"]), store.coordinate(spec["longitude_name"])

    def one(init_time: pd.Timestamp) -> pd.DataFrame:
        grid = store.field2d(spec["t2m_name"], index_of(times, init_time, "init_time"), lead_index)
        values = bilinear(grid, latitudes, longitudes, stations.latitude.to_numpy(), stations.longitude.to_numpy()) - 273.15
        return pd.DataFrame({"station_id": stations.station_id, "init_time": init_time, f"{args.model}_t2m_c": values})

    print(f"starting extension={args.model}; initialisations={len(initialisations)}", flush=True)
    with ThreadPoolExecutor(max_workers=cfg["download"]["max_workers"]) as executor:
        frames = []
        for index, frame in enumerate(executor.map(one, initialisations), start=1):
            if index == 1 or index % 25 == 0 or index == len(initialisations):
                print(f"{args.model}: {index}/{len(initialisations)}", flush=True)
            frames.append(frame)
    extension = pd.concat(frames, ignore_index=True)
    extension = base[["station_id", "init_time", "valid_time"]].merge(extension, on=["station_id", "init_time"], validate="one_to_one")
    if extension[f"{args.model}_t2m_c"].isna().any():
        raise RuntimeError("extension contains missing model values")
    write_parquet(extension, output)
    manifest = {
        "pilot_name": cfg["pilot_name"], "extension": args.model, "cohort": spec["cohort"],
        "stations": len(stations), "initialisations": len(initialisations), "cases": len(extension),
        "lead_hours": lead, "output": str(output.relative_to(ROOT)),
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "base_panel_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        "status": "materialised on the fixed regional panel; ready for an explicit four-model merge",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
