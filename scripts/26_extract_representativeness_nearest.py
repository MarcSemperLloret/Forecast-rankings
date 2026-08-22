#!/usr/bin/env python3
"""Materialise nearest-grid-point T2m for the point-to-grid representativeness family.

Every field this reads has already been fetched by the spatial replication, so
the pass reuses the local chunk cache and adds no new grid downloads. The grid
point nearest a station is the containing cell, so a single extraction serves
both the cell aggregation and the bilinear-versus-nearest sensitivity.
"""
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
from public_zarr import PublicZarr, nearest_indices

CONFIG = ROOT / "config" / "regional_t2m_2020.yaml"
CACHE = ROOT / "data" / "raw" / "weatherbench2_cache"


def index_of(values: np.ndarray | pd.DatetimeIndex, wanted: object, name: str) -> int:
    found = np.flatnonzero(np.asarray(values) == wanted)
    if len(found) != 1:
        raise RuntimeError(f"{name}={wanted} is missing or non-unique")
    return int(found[0])


def station_indices(store: PublicZarr, spec: dict, stations: pd.DataFrame, era5_grid: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Index each station in one store, after checking it holds the ERA5 cells.

    The stores agree on the cells but not on their storage order: HRES and
    GraphCast run latitude south to north, ERA5 and Pangu north to south. The
    cell set is therefore compared as a sorted set and the indices are resolved
    against each store's own coordinates.
    """
    latitudes = store.coordinate(spec["latitude_name"])
    longitudes = store.coordinate(spec["longitude_name"])
    for own, reference, axis in ((latitudes, era5_grid[0], "latitude"), (np.mod(longitudes, 360), np.mod(era5_grid[1], 360), "longitude")):
        if own.shape != reference.shape or not np.allclose(np.sort(own), np.sort(reference)):
            raise RuntimeError(f"store does not hold the ERA5 {axis} cells; cells are not comparable")
    return nearest_indices(latitudes, longitudes, stations.latitude.to_numpy(), stations.longitude.to_numpy())


def nearest_values(store: PublicZarr, spec: dict, stations: pd.DataFrame, times: pd.DatetimeIndex,
                   rows: np.ndarray, columns: np.ndarray, lead: int | None, workers: int, name: str) -> pd.DataFrame:
    store_times = store.times(spec["time_name"])
    lead_index = None if lead is None else index_of(store.timedeltas_hours(spec["lead_name"]), lead, "lead")
    stamp = "init_time" if lead is not None else "valid_time"

    def one(when: pd.Timestamp) -> pd.DataFrame:
        grid = store.field2d(spec["t2m_name"], index_of(store_times, when, stamp), lead_index)
        return pd.DataFrame({"station_id": stations.station_id, stamp: when, f"{name}_nearest_t2m_c": grid[rows, columns] - 273.15})

    with ThreadPoolExecutor(max_workers=workers) as executor:
        frames = []
        for number, frame in enumerate(executor.map(one, times), start=1):
            if number == 1 or number % 100 == 0 or number == len(times):
                print(f"nearest {name}: {number}/{len(times)}", flush=True)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    con = duckdb.connect()
    con.register("frame", frame)
    con.execute(f"COPY frame TO '{path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lead", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    extension, paths = cfg["representativeness_extension"], cfg["paths"]
    if args.lead not in extension["leads_hours"]:
        raise ValueError("lead is outside the pre-specified representativeness family")
    destination = ROOT / paths["representativeness_directory"] / f"lead={args.lead:03d}" / "nearest.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not args.force:
        print(f"already materialised: {destination}")
        return

    stations_path = ROOT / paths["spatial_stations_csv"]
    stations = pd.read_csv(stations_path)
    source = ROOT / paths["spatial_directory"] / f"lead={args.lead:03d}" / "batches" / "*.parquet"
    if not list(source.parent.glob("*.parquet")):
        raise FileNotFoundError(f"the spatial panel for lead={args.lead} must exist first")
    initialisations = pd.DatetimeIndex(
        duckdb.sql(f"SELECT DISTINCT init_time FROM read_parquet('{source.as_posix()}') ORDER BY init_time").fetchdf().init_time
    )

    specs = cfg["weatherbench2"]["models"]
    era5_spec = cfg["weatherbench2"]["era5"]
    stores = {name: PublicZarr(spec["path"], CACHE) for name, spec in specs.items()}
    era5_store = PublicZarr(era5_spec["path"], CACHE)
    latitudes = era5_store.coordinate(era5_spec["latitude_name"])
    longitudes = era5_store.coordinate(era5_spec["longitude_name"])
    indices = {name: station_indices(store, specs[name], stations, (latitudes, longitudes)) for name, store in stores.items()}
    indices["era5"] = station_indices(era5_store, era5_spec, stations, (latitudes, longitudes))
    rows, columns = indices["era5"]

    cells = stations[["station_id", "latitude", "longitude", "altitude_m"]].copy()
    cells["cell_latitude"] = latitudes[rows]
    cells["cell_longitude"] = np.mod(longitudes[columns] + 180, 360) - 180
    cells["cell_id"] = [f"{lat:+08.4f}_{lon:+09.4f}" for lat, lon in zip(cells.cell_latitude, cells.cell_longitude)]
    cells["stations_in_cell"] = cells.groupby("cell_id").station_id.transform("size")
    strata = extension["station_density_strata"]
    cells["density_stratum"] = pd.Series(
        [next(name for name, (low, high) in strata.items() if low <= count <= high) for count in cells.stations_in_cell],
        index=cells.index,
    )
    (ROOT / paths["station_grid_cells_csv"]).parent.mkdir(parents=True, exist_ok=True)
    cells.to_csv(ROOT / paths["station_grid_cells_csv"], index=False)

    print(f"starting representativeness lead={args.lead}; stations={len(stations)}; cells={cells.cell_id.nunique()}; inits={len(initialisations)}", flush=True)
    workers = cfg["download"]["max_workers"]
    aligned: pd.DataFrame | None = None
    for name, spec in specs.items():
        values = nearest_values(stores[name], spec, stations, initialisations, *indices[name], args.lead, workers, name)
        values["valid_time"] = values.init_time + pd.Timedelta(hours=args.lead)
        values["lead_h"] = args.lead
        aligned = values if aligned is None else aligned.merge(values, on=["station_id", "init_time", "valid_time", "lead_h"], validate="one_to_one")
    assert aligned is not None
    valid_times = initialisations + pd.Timedelta(hours=args.lead)
    era5 = nearest_values(era5_store, era5_spec, stations, valid_times, rows, columns, None, workers, "era5")
    aligned = aligned.merge(era5, on=["station_id", "valid_time"], validate="one_to_one")
    aligned = aligned.merge(cells[["station_id", "cell_id", "cell_latitude", "cell_longitude", "stations_in_cell", "density_stratum"]], on="station_id", validate="many_to_one")
    write_parquet(aligned, destination)

    manifest = {"pilot_name": cfg["pilot_name"], "analysis": "point_to_grid_representativeness", "lead_hours": args.lead,
                "initialisations": len(initialisations), "stations": len(stations), "cells": int(cells.cell_id.nunique()),
                "cells_with_two_or_more_stations": int((cells.drop_duplicates("cell_id").stations_in_cell >= extension["min_cell_stations_for_aggregate"]).sum()),
                "expected_cases": len(aligned), "output": str(destination.relative_to(ROOT)),
                "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(), "station_table_sha256": hashlib.sha256(stations_path.read_bytes()).hexdigest(),
                "status": "materialised from the local chunk cache; no new grid downloads"}
    (destination.parent / "availability.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
