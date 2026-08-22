#!/usr/bin/env python3
"""Materialise dry-run model-cell-day roughness with geodesic Gaussian kernels."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import yaml
from scipy.sparse import load_npz, save_npz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from public_zarr import PublicZarr, bilinear
from spherical_roughness import (
    build_geodesic_gaussian_matrix,
    coordinate_signature,
    direct_gaussian_values,
    equal_area_cell_centres,
    self_test,
    target_signature,
)
from tessellation import EqualAreaGrid

MODEL_CONFIG = ROOT / "config" / "weather5k_model_dry_run_2020.yaml"
ANALYSIS_CONFIG = ROOT / "config" / "regional_t2m_2020.yaml"
DEFAULT_PANEL = ROOT / "data" / "interim" / "weather5k_weatherbench2_dry_run_2020" / "panel.parquet"
DEFAULT_SITES = ROOT / "results" / "weather5k_weatherbench2_dry_run_2020" / "sites.csv"
DEFAULT_OUTPUT = ROOT / "data" / "interim" / "weather5k_global_roughness_dry_run_2020"


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_from_root(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def verify_seal() -> str:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "40_freeze_analysis_parameters.py"), "--verify"],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    return result.stdout.strip()


def index_map(values: pd.DatetimeIndex) -> dict[int, int]:
    return {pd.Timestamp(value).value: index for index, value in enumerate(values)}


def field_specs(config: dict[str, Any]) -> dict[str, dict]:
    specs = dict(config["weatherbench2"]["models"])
    specs.update(config["model_extensions"])
    return specs


def cached_kernel(
    kernel_root: Path,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    cell_ids: np.ndarray,
    target_latitudes: np.ndarray,
    target_longitudes: np.ndarray,
    sigma_km: float,
    truncation_sigma: float,
) -> tuple[Any, dict]:
    grid_hash = coordinate_signature(latitudes, longitudes)
    targets_hash = target_signature(cell_ids, target_latitudes, target_longitudes)
    stem = f"geodesic_{grid_hash[:12]}_{targets_hash[:12]}_{sigma_km:g}km_{truncation_sigma:g}sigma"
    matrix_path = kernel_root / f"{stem}.npz"
    metadata_path = kernel_root / f"{stem}.json"
    if matrix_path.exists() != metadata_path.exists():
        raise RuntimeError(f"incomplete cached kernel: {stem}")
    if matrix_path.exists():
        matrix = load_npz(matrix_path).tocsr()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        matrix, diagnostics = build_geodesic_gaussian_matrix(
            latitudes,
            longitudes,
            target_latitudes,
            target_longitudes,
            sigma_km,
            truncation_sigma,
        )
        save_npz(matrix_path, matrix, compressed=True)
        metadata = {
            "grid_signature": grid_hash,
            "target_signature": targets_hash,
            "matrix": diagnostics.as_dict(),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    expected_shape = (len(cell_ids), len(latitudes) * len(longitudes))
    if matrix.shape != expected_shape:
        raise RuntimeError(f"cached kernel shape {matrix.shape} != expected {expected_shape}")
    row_error = float(np.max(np.abs(np.asarray(matrix.sum(axis=1)).ravel() - 1.0)))
    if row_error >= 1e-12:
        raise RuntimeError(f"kernel does not preserve constants: max row error {row_error}")
    metadata.update(
        {
            "matrix_path": str(matrix_path),
            "matrix_sha256": sha256(matrix_path),
            "metadata_path": str(metadata_path),
            "metadata_sha256": sha256(metadata_path),
            "maximum_row_sum_error_verified": row_error,
        }
    )
    return matrix, metadata


def actual_grid_validation(
    matrices: dict[float, Any],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    target_latitudes: np.ndarray,
    target_longitudes: np.ndarray,
    scales: list[float],
    truncation_sigma: float,
) -> dict:
    latitude_grid, longitude_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
    synthetic = (
        np.sin(np.deg2rad(latitude_grid))
        + 0.3 * np.cos(2 * np.deg2rad(longitude_grid))
        + 0.1 * np.sin(3 * np.deg2rad(longitude_grid))
    )
    selected = np.unique(np.linspace(0, len(target_latitudes) - 1, 3, dtype=int))
    errors = {}
    for scale in scales:
        sparse_values = np.asarray(matrices[scale][selected] @ synthetic.ravel()).ravel()
        direct_values = direct_gaussian_values(
            synthetic,
            latitudes,
            longitudes,
            target_latitudes[selected],
            target_longitudes[selected],
            scale,
            truncation_sigma,
        )
        errors[f"{scale:g}km"] = float(np.max(np.abs(sparse_values - direct_values)))
    return {
        "targets_checked": selected.tolist(),
        "sparse_vs_direct_max_abs_error": errors,
        # On the real 0.25-degree grid, points can lie numerically on the 4-sigma
        # cutoff. Sparse and brute-force neighbor selection can then differ at
        # roundoff level even though the omitted weight is negligible.
        "tolerance": 1e-9,
        "passed": max(errors.values()) < 1e-9,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--sites", type=Path, default=DEFAULT_SITES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-config", type=Path, default=MODEL_CONFIG)
    parser.add_argument(
        "--analysis-role",
        default="engineering_dry_run_only_not_confirmatory_evidence",
    )
    parser.add_argument(
        "--interpretation-warning",
        default=(
            "Dry run only. Source variants are not independent model units, and "
            "WEATHER-5K/ISD may overlap observations assimilated by ERA5."
        ),
    )
    parser.add_argument("--sample-days", type=int, default=None)
    parser.add_argument("--models", default=None, help="comma-separated subset of source variants")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            raise SystemExit(1)
        return

    seal = verify_seal()
    model_config_path = args.model_config.resolve()
    model_config = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))
    analysis_config = yaml.safe_load(ANALYSIS_CONFIG.read_text(encoding="utf-8"))
    specs = field_specs(analysis_config)
    specs.update(model_config["weatherbench2"].get("model_specs", {}))
    source_rows = model_config["weatherbench2"]["sources"]
    if args.models:
        wanted = {name.strip() for name in args.models.split(",") if name.strip()}
        source_rows = [row for row in source_rows if row["model"] in wanted]
        missing = wanted.difference(row["model"] for row in source_rows)
        if missing:
            raise ValueError(f"unknown model source variants: {sorted(missing)}")
    workers = args.workers or int(model_config["execution"]["workers"])
    lead_hours = int(model_config["weatherbench2"]["lead_hours"])
    cache_root = resolve_from_root(model_config["weatherbench2"]["cache"])
    scales = [
        float(value)
        for value in model_config.get("roughness", {}).get(
            "scales_km", analysis_config["field_smoothness"]["scales_km"]
        )
    ]
    truncation_sigma = 4.0
    panel_path, sites_path, output = args.panel.resolve(), args.sites.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    parts_root, kernel_root = output / "model_parts", output / "kernels"
    parts_root.mkdir(parents=True, exist_ok=True)
    kernel_root.mkdir(parents=True, exist_ok=True)
    final_path, manifest_path = output / "roughness.parquet", output / "manifest.json"
    final_outputs = [final_path, manifest_path]
    known = [*final_outputs, *parts_root.glob("*.parquet")]
    if any(path.exists() for path in final_outputs) and not args.force:
        raise FileExistsError(f"output exists in {output}; use --force")
    if args.force:
        for path in known:
            if path.exists():
                if not path.is_file():
                    raise RuntimeError(f"refusing to replace non-file {path}")
                path.unlink()

    sites = pd.read_csv(sites_path, dtype={"cell_id": "string"})
    cell_ids = np.sort(sites.cell_id.dropna().unique().astype(str))
    grid = EqualAreaGrid(float(analysis_config["spatial_weighting"]["target_cell_km"]))
    target_latitudes, target_longitudes = equal_area_cell_centres(grid, cell_ids)
    connection = duckdb.connect()
    valid_times = pd.DatetimeIndex(
        connection.execute(
            f"SELECT DISTINCT valid_time FROM read_parquet('{panel_path.as_posix()}') ORDER BY valid_time"
        ).fetchdf().valid_time
    )
    if args.sample_days is not None:
        if args.sample_days < 1:
            raise ValueError("--sample-days must be positive")
        selected_days = valid_times.normalize().unique()[: args.sample_days]
        valid_times = valid_times[valid_times.normalize().isin(selected_days)]
    days = pd.DatetimeIndex(valid_times.normalize().unique())
    day_lookup = {day.value: index for index, day in enumerate(days)}
    print(
        f"Roughness support: {len(source_rows)} variants, {len(cell_ids)} cells, "
        f"{len(days)} days, {len(valid_times)} fields per variant",
        flush=True,
    )

    grid_kernels: dict[str, dict[float, Any]] = {}
    kernel_manifest: dict[str, Any] = {}
    source_manifest: dict[str, Any] = {}
    actual_validation: dict[str, Any] = {}
    for source_row in source_rows:
        name = source_row["model"]
        spec = specs[name]
        store = PublicZarr(spec["path"], cache_root)
        latitudes = store.coordinate(spec["latitude_name"])
        longitudes = store.coordinate(spec["longitude_name"])
        signature = coordinate_signature(latitudes, longitudes)
        if signature not in grid_kernels:
            matrices, matrix_meta = {}, {}
            for scale in scales:
                matrices[scale], matrix_meta[f"{scale:g}km"] = cached_kernel(
                    kernel_root,
                    latitudes,
                    longitudes,
                    cell_ids,
                    target_latitudes,
                    target_longitudes,
                    scale,
                    truncation_sigma,
                )
            grid_kernels[signature] = matrices
            kernel_manifest[signature] = matrix_meta
            actual_validation[signature] = actual_grid_validation(
                matrices,
                latitudes,
                longitudes,
                target_latitudes,
                target_longitudes,
                scales,
                truncation_sigma,
            )
            if not actual_validation[signature]["passed"]:
                raise RuntimeError(
                    f"sparse kernel failed direct validation for grid {signature}: "
                    f"{actual_validation[signature]}"
                )
        matrices = grid_kernels[signature]
        destination = parts_root / f"{name}.parquet"
        if destination.exists():
            print(f"Reusing completed roughness part: {destination.name}", flush=True)
            part_counts = connection.execute(
                f"""SELECT count(*), min(n_cycles), max(n_cycles),
                           count(DISTINCT cell_id), count(DISTINCT valid_day)
                    FROM read_parquet('{destination.as_posix()}')"""
            ).fetchone()
            expected_part_rows = len(days) * len(cell_ids)
            if int(part_counts[0]) != expected_part_rows:
                raise RuntimeError(
                    f"incomplete cached roughness part {destination}: "
                    f"{part_counts[0]} != {expected_part_rows}"
                )
            source_manifest[name] = {
                "family": source_row["family"],
                "cohort": source_row["cohort"],
                "grid_signature": signature,
                "rows": int(part_counts[0]),
                "minimum_cycles_per_day": int(part_counts[1]),
                "maximum_cycles_per_day": int(part_counts[2]),
                "path": str(destination),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
                "reused": True,
            }
            continue
        store_times = store.times(spec["time_name"])
        time_lookup = index_map(store_times)
        lead_index_values = store.timedeltas_hours(spec["lead_name"])
        lead_positions = np.flatnonzero(lead_index_values == lead_hours)
        if len(lead_positions) != 1:
            raise RuntimeError(f"{name}: lead {lead_hours} is missing or non-unique")
        lead_index = int(lead_positions[0])
        sum_squares = np.zeros((len(days), len(cell_ids), len(scales)), dtype=np.float64)
        cycle_counts = np.zeros(len(days), dtype=np.int16)

        def one(valid_time: pd.Timestamp) -> tuple[int, np.ndarray]:
            initialisation = valid_time - pd.Timedelta(hours=lead_hours)
            time_index = time_lookup.get(initialisation.value)
            if time_index is None:
                raise RuntimeError(f"{name}: missing initialisation {initialisation}")
            field = store.geographic_field2d(
                spec["t2m_name"],
                time_index,
                lead_index,
                spec["latitude_name"],
                spec["longitude_name"],
            )
            raw = bilinear(
                field,
                latitudes,
                longitudes,
                target_latitudes,
                target_longitudes,
            )
            squared = np.empty((len(cell_ids), len(scales)), dtype=np.float64)
            flattened = np.asarray(field, dtype=np.float64).ravel()
            for position, scale in enumerate(scales):
                smooth = np.asarray(matrices[scale] @ flattened).ravel()
                squared[:, position] = np.square(raw - smooth)
            if not np.all(np.isfinite(squared)):
                raise RuntimeError(f"{name}: non-finite roughness contribution at {valid_time}")
            return day_lookup[valid_time.normalize().value], squared

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for number, (day_index, squared) in enumerate(executor.map(one, valid_times), start=1):
                sum_squares[day_index] += squared
                cycle_counts[day_index] += 1
                if number == 1 or number % 100 == 0 or number == len(valid_times):
                    print(f"{name}: {number}/{len(valid_times)} fields", flush=True)
        if np.any(cycle_counts == 0):
            raise RuntimeError(f"{name}: a selected day received no cycles")
        roughness = np.sqrt(sum_squares / cycle_counts[:, None, None])
        frame = pd.DataFrame(
            {
                "model": name,
                "model_family": source_row["family"],
                "cohort": source_row["cohort"],
                "cell_id": np.tile(cell_ids, len(days)),
                "valid_day": np.repeat(days.date, len(cell_ids)),
                "n_cycles": np.repeat(cycle_counts, len(cell_ids)),
            }
        )
        for position, scale in enumerate(scales):
            frame[f"roughness_{scale:g}km_k"] = roughness[:, :, position].ravel()
        connection.register("roughness_frame", frame)
        connection.execute(
            f"COPY roughness_frame TO '{destination.as_posix()}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        connection.unregister("roughness_frame")
        source_manifest[name] = {
            "family": source_row["family"],
            "cohort": source_row["cohort"],
            "grid_signature": signature,
            "rows": int(len(frame)),
            "minimum_cycles_per_day": int(cycle_counts.min()),
            "maximum_cycles_per_day": int(cycle_counts.max()),
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        }

    part_glob = (parts_root / "*.parquet").as_posix()
    connection.execute(
        f"COPY (SELECT * FROM read_parquet('{part_glob}') ORDER BY model, valid_day, cell_id) "
        f"TO '{final_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    columns = [f"roughness_{scale:g}km_k" for scale in scales]
    nonnegative = " AND ".join(f"{column} >= 0 AND isfinite({column})" for column in columns)
    counts = connection.execute(
        f"""SELECT count(*), count(DISTINCT model), count(DISTINCT cell_id),
                   count(DISTINCT valid_day),
                   count(*) FILTER (WHERE NOT ({nonnegative})),
                   count(*) - count(DISTINCT (model, cell_id, valid_day))
            FROM read_parquet('{final_path.as_posix()}')"""
    ).fetchone()
    summaries = connection.execute(
        f"""SELECT model, {', '.join(f'avg({column}) AS mean_{column}' for column in columns)}
            FROM read_parquet('{final_path.as_posix()}') GROUP BY model ORDER BY model"""
    ).fetchdf()
    connection.close()
    expected_rows = len(source_rows) * len(cell_ids) * len(days)
    validation = {
        "expected_rows": expected_rows,
        "actual_rows": int(counts[0]),
        "source_variants": int(counts[1]),
        "cells": int(counts[2]),
        "days": int(counts[3]),
        "invalid_or_negative_rows": int(counts[4]),
        "duplicate_model_cell_day_rows": int(counts[5]),
        "passed": int(counts[0]) == expected_rows and counts[4] == 0 and counts[5] == 0,
    }
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": args.analysis_role,
        "sealed_parameter_verification": seal,
        "script_sha256": sha256(Path(__file__).resolve()),
        "module_sha256": sha256(ROOT / "src" / "spherical_roughness.py"),
        "model_config": str(model_config_path),
        "model_config_sha256": sha256(model_config_path),
        "analysis_config_sha256": sha256(ANALYSIS_CONFIG),
        "panel": {"path": str(panel_path), "sha256": sha256(panel_path)},
        "sites": {"path": str(sites_path), "sha256": sha256(sites_path)},
        "definition": {
            "target": "area centroid of each occupied frozen equal-area cell",
            "kernel": "exp(-d_great_circle^2 / (2 sigma_km^2)) times cos(source_latitude)",
            "scales_km": scales,
            "truncation_sigma": truncation_sigma,
            "omitted_radial_mass_upper_bound": float(np.exp(-0.5 * truncation_sigma**2)),
            "cell_day_statistic": "sqrt(mean_cycle((field_at_cell - gaussian_mean)^2))",
            "longitude_boundary": "intrinsically periodic through unit-sphere Cartesian distance",
            "pole_handling": "great-circle distance; latitude-longitude quadrature weighted by cos(latitude)",
        },
        "selection": {
            "sample_days": args.sample_days,
            "models_argument": args.models,
            "lead_hours": lead_hours,
            "first_valid_time": valid_times.min().isoformat(),
            "last_valid_time": valid_times.max().isoformat(),
        },
        "kernel_matrices": kernel_manifest,
        "synthetic_self_test": self_test(),
        "actual_grid_sparse_vs_direct_validation": actual_validation,
        "sources": source_manifest,
        "validation": validation,
        "model_mean_roughness": summaries.to_dict(orient="records"),
        "output": {
            "path": str(final_path),
            "bytes": final_path.stat().st_size,
            "sha256": sha256(final_path),
        },
        "interpretation_warning": args.interpretation_warning,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summaries.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(json.dumps(validation, indent=2))
    if not validation["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
