#!/usr/bin/env python3
"""Materialise pre-specified static physical covariates for global sites/cells."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import save_npz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from public_zarr import PublicZarr, bilinear
from spherical_roughness import build_geodesic_gaussian_matrix, self_test

DEFAULT_SITES = ROOT / "results" / "weather5k_weatherbench2_dry_run_2020" / "sites.csv"
DEFAULT_OUTPUT = ROOT / "data" / "interim" / "weather5k_physical_covariates_2020"
ERA5 = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def climate_belt(latitude: pd.Series) -> pd.Series:
    absolute = latitude.abs()
    return pd.Series(
        np.select(
            [absolute < 23.5, absolute < 45.0, absolute < 66.5],
            ["tropical", "subtropical", "midlatitude"],
            default="polar",
        ),
        index=latitude.index,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=Path, default=DEFAULT_SITES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=ROOT / "data" / "raw" / "weatherbench2_cache")
    parser.add_argument("--coastal-scale-km", type=float, default=100.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    sites_path, output = args.sites.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    site_path, cell_path = output / "site_covariates.csv", output / "cell_covariates.csv"
    kernel_path, manifest_path = output / "landsea_kernel.npz", output / "manifest.json"
    targets = [site_path, cell_path, kernel_path, manifest_path]
    if any(path.exists() for path in targets) and not args.force:
        raise FileExistsError(f"outputs exist in {output}; use --force")
    if args.force:
        for path in targets:
            if path.exists():
                if not path.is_file():
                    raise RuntimeError(f"refusing to replace non-file {path}")
                path.unlink()

    sites = pd.read_csv(sites_path, dtype={"cell_id": "string", "site_id": "string"})
    latitude = sites.mean_latitude.to_numpy(float)
    longitude = sites.mean_longitude.to_numpy(float)
    store = PublicZarr(ERA5, args.cache.resolve())
    latitudes = store.coordinate("latitude")
    longitudes = store.coordinate("longitude")
    static_names = [
        "geopotential_at_surface",
        "standard_deviation_of_orography",
        "standard_deviation_of_filtered_subgrid_orography",
        "slope_of_sub_gridscale_orography",
        "land_sea_mask",
    ]
    fields = {name: store.static_field2d(name).astype(np.float64) for name in static_names}
    sampled = {
        name: bilinear(field, latitudes, longitudes, latitude, longitude)
        for name, field in fields.items()
    }
    kernel, diagnostics = build_geodesic_gaussian_matrix(
        latitudes,
        longitudes,
        latitude,
        longitude,
        float(args.coastal_scale_km),
        4.0,
    )
    save_npz(kernel_path, kernel, compressed=True)
    land = fields["land_sea_mask"].ravel()
    local_land_mean = np.asarray(kernel @ land).ravel()
    local_land_second = np.asarray(kernel @ np.square(land)).ravel()
    local_land_sd = np.sqrt(np.maximum(local_land_second - np.square(local_land_mean), 0.0))

    result = sites.copy()
    result["absolute_latitude_deg"] = np.abs(latitude)
    result["climate_belt"] = climate_belt(result.mean_latitude)
    result["era5_grid_elevation_m"] = sampled["geopotential_at_surface"] / 9.80665
    result["station_minus_era5_elevation_m"] = result.altitude_m - result.era5_grid_elevation_m
    result["absolute_elevation_mismatch_m"] = result.station_minus_era5_elevation_m.abs()
    result["era5_orography_sd_m"] = sampled["standard_deviation_of_orography"]
    result["era5_filtered_subgrid_orography_sd_m"] = sampled[
        "standard_deviation_of_filtered_subgrid_orography"
    ]
    result["era5_subgrid_orography_slope"] = sampled["slope_of_sub_gridscale_orography"]
    result["era5_land_fraction_at_site"] = sampled["land_sea_mask"]
    result["era5_land_fraction_100km"] = local_land_mean
    result["era5_landsea_heterogeneity_100km"] = local_land_sd
    result["local_solar_hour_at_00utc"] = np.mod(longitude / 15.0, 24.0)
    result["local_solar_hour_at_12utc"] = np.mod(12.0 + longitude / 15.0, 24.0)
    result.to_csv(site_path, index=False)

    cell = (
        result.groupby("cell_id", as_index=False)
        .agg(
            sites=("site_id", "nunique"),
            mean_latitude=("mean_latitude", "mean"),
            mean_longitude=("mean_longitude", "mean"),
            absolute_latitude_deg=("absolute_latitude_deg", "mean"),
            altitude_m=("altitude_m", "mean"),
            station_minus_era5_elevation_m=("station_minus_era5_elevation_m", "mean"),
            absolute_elevation_mismatch_m=("absolute_elevation_mismatch_m", "mean"),
            era5_orography_sd_m=("era5_orography_sd_m", "mean"),
            era5_filtered_subgrid_orography_sd_m=("era5_filtered_subgrid_orography_sd_m", "mean"),
            era5_subgrid_orography_slope=("era5_subgrid_orography_slope", "mean"),
            era5_land_fraction_100km=("era5_land_fraction_100km", "mean"),
            era5_landsea_heterogeneity_100km=("era5_landsea_heterogeneity_100km", "mean"),
        )
    )
    cell["climate_belt"] = climate_belt(cell.mean_latitude)
    cell.to_csv(cell_path, index=False)
    validation = {
        "sites": int(len(result)),
        "cells": int(len(cell)),
        "missing_covariate_values": int(result.iloc[:, len(sites.columns):].isna().sum().sum()),
        "maximum_kernel_row_sum_error": float(
            np.max(np.abs(np.asarray(kernel.sum(axis=1)).ravel() - 1.0))
        ),
        "landsea_sd_range": [float(local_land_sd.min()), float(local_land_sd.max())],
    }
    validation["passed"] = (
        validation["missing_covariate_values"] == 0
        and validation["maximum_kernel_row_sum_error"] < 1e-12
        and validation["landsea_sd_range"][0] >= 0
        and validation["landsea_sd_range"][1] <= 0.5 + 1e-9
    )
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "pre-specified physical heterogeneity covariates; outcomes not read",
        "sites": {"path": sites_path.relative_to(ROOT).as_posix(), "sha256": sha256(sites_path)},
        "era5_store": ERA5,
        "definitions": {
            "gravity_m_s2": 9.80665,
            "coastal_kernel_sigma_km": float(args.coastal_scale_km),
            "coastal_kernel_truncation_sigma": 4.0,
            "landsea_heterogeneity": "sqrt(E_k[lsm^2] - E_k[lsm]^2)",
            "climate_belts_abs_latitude_deg": [23.5, 45.0, 66.5],
            "directional_prediction": (
                "gridded-reference advantage becomes more negative as absolute elevation "
                "mismatch, subgrid orographic variability, or land-sea heterogeneity increase"
            ),
        },
        "kernel": {
            "path": kernel_path.relative_to(ROOT).as_posix(),
            "sha256": sha256(kernel_path),
            "diagnostics": diagnostics.as_dict(),
        },
        "synthetic_geodesic_self_test": self_test(),
        "validation": validation,
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (site_path, cell_path)
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))
    if not validation["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
