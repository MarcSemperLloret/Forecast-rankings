#!/usr/bin/env python3
"""Materialise station elevation and common ERA5-grid terrain at each station."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from public_zarr import PublicZarr, bilinear

CONFIG = ROOT / "config" / "regional_t2m_2020.yaml"
CACHE = ROOT / "data" / "raw" / "weatherbench2_cache"


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    paths, sensitivity = cfg["paths"], cfg["orography_sensitivity"]
    stations = pd.read_csv(ROOT / paths["spatial_stations_csv"])
    era_spec = cfg["weatherbench2"]["era5"]
    era = PublicZarr(era_spec["path"], CACHE)
    lats, lons = era.coordinate(era_spec["latitude_name"]), era.coordinate(era_spec["longitude_name"])
    # The forecast fields share ERA5's 0.25-degree coordinate system, but do
    # not expose a surface-height variable. Fail if that coordinate premise is false.
    coordinate_checks = {}
    for name, spec in cfg["weatherbench2"]["models"].items():
        store = PublicZarr(spec["path"], CACHE)
        model_lats, model_lons = store.coordinate(spec["latitude_name"]), store.coordinate(spec["longitude_name"])
        # IFS and GraphCast store latitude south-to-north, while ERA5 and Pangu
        # store it north-to-south; the horizontal locations are nevertheless identical.
        same_latitudes = np.array_equal(lats, model_lats) or np.array_equal(lats, model_lats[::-1])
        same = same_latitudes and np.array_equal(lons, model_lons)
        coordinate_checks[name] = bool(same)
        if not same:
            raise RuntimeError(f"{name} does not share the ERA5 horizontal coordinate grid")
    geopotential = era.static_field2d("geopotential_at_surface")
    terrain_m = bilinear(geopotential, lats, lons, stations.latitude.to_numpy(), stations.longitude.to_numpy()) / sensitivity["gravity_m_s2"]
    output = stations[["station_id", "latitude", "longitude", "altitude_m"]].copy()
    output["era5_grid_elevation_m"] = terrain_m
    output["delta_z_m"] = output.altitude_m - output.era5_grid_elevation_m
    output["abs_delta_z_m"] = output.delta_z_m.abs()
    destination = ROOT / paths["station_grid_elevation_csv"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    summary = {"terrain_source": sensitivity["terrain_source"], "terrain_variable": "geopotential_at_surface", "conversion": f"geopotential / {sensitivity['gravity_m_s2']} m s-2", "station_count": len(output), "coordinate_equality": coordinate_checks, "model_specific_surface_terrain_available": False, "station_table_sha256": hashlib.sha256((ROOT / paths["spatial_stations_csv"]).read_bytes()).hexdigest(), "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "delta_z_summary_m": output[["delta_z_m", "abs_delta_z_m"]].describe().to_dict()}
    destination.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(output.describe().to_string())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
