#!/usr/bin/env python3
"""Materialise geodesically smoothed forecasts at national-network stations.

The zero-width field is sampled bilinearly and must reproduce the existing
national panel. Positive widths use normalized, area-weighted Gaussian kernels
defined by great-circle distance on the full latitude-longitude grid. This
avoids the latitude anisotropy and continental-domain assumptions of the
original AVAMET-only implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
    self_test,
    target_signature,
)

DEFAULT_CONFIG = ROOT / "config" / "national_causal_extension_2020.yaml"


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def index_map(values: pd.DatetimeIndex) -> dict[int, int]:
    return {pd.Timestamp(value).value: index for index, value in enumerate(values)}


def field_specs(config: dict[str, Any]) -> dict[str, dict]:
    specs = dict(config["weatherbench2"]["models"])
    specs.update(config["model_extensions"])
    return specs


def cached_kernel(
    root: Path,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    stations: pd.DataFrame,
    sigma_km: float,
    truncation_sigma: float,
) -> tuple[Any, dict]:
    grid_hash = coordinate_signature(latitudes, longitudes)
    station_hash = target_signature(
        stations.station_id.astype(str).to_numpy(),
        stations.latitude.to_numpy(float),
        stations.longitude.to_numpy(float),
    )
    stem = f"station_{grid_hash[:12]}_{station_hash[:12]}_{sigma_km:g}km_{truncation_sigma:g}sigma"
    matrix_path = root / f"{stem}.npz"
    metadata_path = root / f"{stem}.json"
    if matrix_path.exists() != metadata_path.exists():
        raise RuntimeError(f"incomplete cached kernel: {stem}")
    if matrix_path.exists():
        matrix = load_npz(matrix_path).tocsr()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        matrix, diagnostics = build_geodesic_gaussian_matrix(
            latitudes,
            longitudes,
            stations.latitude.to_numpy(float),
            stations.longitude.to_numpy(float),
            sigma_km,
            truncation_sigma,
        )
        save_npz(matrix_path, matrix, compressed=True)
        metadata = {
            "grid_signature": grid_hash,
            "station_signature": station_hash,
            "matrix": diagnostics.as_dict(),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    expected = (len(stations), len(latitudes) * len(longitudes))
    if matrix.shape != expected:
        raise RuntimeError(f"cached kernel shape {matrix.shape} != expected {expected}")
    row_error = float(np.max(np.abs(np.asarray(matrix.sum(axis=1)).ravel() - 1.0)))
    if row_error >= 1e-12:
        raise RuntimeError(f"kernel fails constant preservation: {row_error}")
    metadata.update(
        {
            "matrix_path": matrix_path.relative_to(ROOT).as_posix(),
            "matrix_sha256": sha256(matrix_path),
            "metadata_path": metadata_path.relative_to(ROOT).as_posix(),
            "maximum_row_sum_error_verified": row_error,
        }
    )
    return matrix, metadata


def atomic_parquet(connection: duckdb.DuckDBPyConnection, frame: pd.DataFrame, target: Path) -> None:
    temporary = target.with_suffix(".tmp.parquet")
    if temporary.exists():
        temporary.unlink()
    connection.register("output_frame", frame)
    connection.execute(
        f"COPY output_frame TO '{temporary.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    connection.unregister("output_frame")
    temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--models", default=None, help="comma-separated source variants")
    parser.add_argument("--sample-times", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        report = self_test()
        print(json.dumps(report, indent=2))
        if not report["passed"]:
            raise SystemExit(1)
        return

    config_path = args.config.resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    panel_path = resolve(cfg["inputs"]["panel"])
    model_spec_path = resolve(cfg["inputs"]["model_specification"])
    source_config_path = resolve(cfg["inputs"]["model_sources"])
    output = (args.output.resolve() if args.output else resolve(cfg["outputs"]["materialisation_root"]))
    output.mkdir(parents=True, exist_ok=True)
    parts_root, kernel_root = output / "model_parts", output / "kernels"
    parts_root.mkdir(parents=True, exist_ok=True)
    kernel_root.mkdir(parents=True, exist_ok=True)
    final_path, manifest_path = output / "blurred_panel.parquet", output / "manifest.json"
    if (final_path.exists() or manifest_path.exists()) and not args.force:
        raise FileExistsError(f"final output exists in {output}; use --force")
    if args.force:
        for path in [final_path, manifest_path, *parts_root.glob("*.parquet")]:
            if path.exists():
                if not path.is_file():
                    raise RuntimeError(f"refusing to replace non-file {path}")
                path.unlink()

    model_spec = yaml.safe_load(model_spec_path.read_text(encoding="utf-8"))
    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    specs = field_specs(model_spec)
    source_rows = source_config["weatherbench2"]["sources"]
    configured_models = [*cfg["scope"]["primary_models"], *cfg["scope"]["extension_models"]]
    source_rows = [row for row in source_rows if row["model"] in configured_models]
    if args.models:
        requested = {item.strip() for item in args.models.split(",") if item.strip()}
        source_rows = [row for row in source_rows if row["model"] in requested]
        missing = requested.difference(row["model"] for row in source_rows)
        if missing:
            raise ValueError(f"unknown requested models: {sorted(missing)}")
    if not source_rows:
        raise ValueError("no model source variants selected")

    sigmas = [float(value) for value in cfg["controlled_smoothing_s1"]["sigma_km"]]
    positive_sigmas = [value for value in sigmas if value > 0]
    lead_hours = int(cfg["scope"]["lead_hours"])
    truncation_sigma = 4.0
    cache_root = resolve(cfg["inputs"]["weatherbench2_cache"])
    connection = duckdb.connect()
    stations = connection.execute(
        f"""
        SELECT station_id, any_value(latitude) AS latitude,
               any_value(longitude) AS longitude, any_value(network) AS network
        FROM read_parquet('{panel_path.as_posix()}')
        GROUP BY station_id ORDER BY station_id
        """
    ).fetchdf()
    duplicate_coordinates = stations.groupby(["latitude", "longitude"]).size().gt(1).sum()
    if stations.station_id.duplicated().any():
        raise RuntimeError("station identifiers are not unique")

    grid_kernels: dict[str, dict[float, Any]] = {}
    kernel_manifest: dict[str, Any] = {}
    source_manifest: dict[str, Any] = {}
    validation_errors: list[float] = []
    for source_row in source_rows:
        model = source_row["model"]
        destination = parts_root / f"{model}.parquet"
        if destination.exists() and not args.force and args.sample_times is None:
            source_manifest[model] = {
                "path": str(destination),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
                "reused": True,
            }
            print(f"Reusing completed smoothing part: {destination.name}", flush=True)
            continue
        subset = connection.execute(
            f"""
            SELECT station_id, latitude, longitude, altitude_m, network,
                   potentially_assimilated, assimilation_risk_class, valid_time,
                   model, model_family, cohort, lead_h, forecast_t2m_c,
                   era5_t2m_c, observed_t2m_c
            FROM read_parquet('{panel_path.as_posix()}')
            WHERE model = ? ORDER BY valid_time, station_id
            """,
            [model],
        ).fetchdf()
        valid_times = pd.DatetimeIndex(sorted(subset.valid_time.unique()))
        if args.sample_times is not None:
            if args.sample_times < 1:
                raise ValueError("--sample-times must be positive")
            valid_times = valid_times[: args.sample_times]
            subset = subset[subset.valid_time.isin(valid_times)].copy()
        spec = specs[model]
        store = PublicZarr(spec["path"], cache_root)
        latitudes = store.coordinate(spec["latitude_name"])
        longitudes = store.coordinate(spec["longitude_name"])
        signature = coordinate_signature(latitudes, longitudes)
        if signature not in grid_kernels:
            matrices, metadata = {}, {}
            for sigma in positive_sigmas:
                matrices[sigma], metadata[f"{sigma:g}km"] = cached_kernel(
                    kernel_root,
                    latitudes,
                    longitudes,
                    stations,
                    sigma,
                    truncation_sigma,
                )
            grid_kernels[signature] = matrices
            kernel_manifest[signature] = metadata
        matrices = grid_kernels[signature]
        time_lookup = index_map(store.times(spec["time_name"]))
        lead_values = store.timedeltas_hours(spec["lead_name"])
        lead_positions = np.flatnonzero(lead_values == lead_hours)
        if len(lead_positions) != 1:
            raise RuntimeError(f"{model}: lead {lead_hours} is missing or non-unique")
        lead_index = int(lead_positions[0])
        targets = (stations.latitude.to_numpy(float), stations.longitude.to_numpy(float))

        def one(valid_time: pd.Timestamp) -> pd.DataFrame:
            initialisation = valid_time - pd.Timedelta(hours=lead_hours)
            time_index = time_lookup.get(initialisation.value)
            if time_index is None:
                raise RuntimeError(f"{model}: missing initialisation {initialisation}")
            field = np.asarray(store.field2d(spec["t2m_name"], time_index, lead_index), dtype=np.float64)
            values: dict[str, Any] = {
                "station_id": stations.station_id,
                "valid_time": valid_time,
                "model": model,
                "sigma_0km": bilinear(field, latitudes, longitudes, *targets) - 273.15,
            }
            flattened = field.ravel()
            for sigma in positive_sigmas:
                values[f"sigma_{sigma:g}km"] = np.asarray(matrices[sigma] @ flattened).ravel() - 273.15
            return pd.DataFrame(values)

        frames = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for number, frame in enumerate(executor.map(one, valid_times), start=1):
                frames.append(frame)
                if number == 1 or number % 100 == 0 or number == len(valid_times):
                    print(f"{model}: {number}/{len(valid_times)} fields", flush=True)
        predictions = pd.concat(frames, ignore_index=True)
        joined = subset.merge(
            predictions,
            on=["station_id", "valid_time", "model"],
            how="left",
            validate="one_to_one",
        )
        sigma_columns = [f"sigma_{sigma:g}km" for sigma in sigmas]
        if joined[sigma_columns].isna().any().any():
            raise RuntimeError(f"{model}: smoothed predictions contain missing values")
        max_error = float(np.max(np.abs(joined["sigma_0km"] - joined["forecast_t2m_c"])))
        validation_errors.append(max_error)
        if max_error > 1e-6:
            raise RuntimeError(f"{model}: sigma-zero mismatch {max_error} C exceeds 1e-6 C")
        atomic_parquet(connection, joined, destination)
        source_manifest[model] = {
            "family": source_row["family"],
            "cohort": source_row["cohort"],
            "grid_signature": signature,
            "rows": int(len(joined)),
            "valid_times": int(len(valid_times)),
            "sigma_zero_max_abs_error_c": max_error,
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        }

    selected_names = [row["model"] for row in source_rows]
    selected_paths = [parts_root / f"{name}.parquet" for name in selected_names]
    if not all(path.exists() for path in selected_paths):
        raise RuntimeError("one or more model parts are missing")
    paths_sql = ",".join(f"'{path.as_posix()}'" for path in selected_paths)
    connection.execute(
        f"COPY (SELECT * FROM read_parquet([{paths_sql}]) ORDER BY model, valid_time, station_id) "
        f"TO '{final_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    counts = connection.execute(
        f"""
        SELECT count(*) AS rows, count(DISTINCT model) AS models,
               count(DISTINCT station_id) AS stations,
               count(DISTINCT valid_time) AS valid_times,
               count(*) - count(DISTINCT (model, station_id, valid_time)) AS duplicates
        FROM read_parquet('{final_path.as_posix()}')
        """
    ).fetchone()
    connection.close()
    validation = {
        "rows": int(counts[0]),
        "models": int(counts[1]),
        "stations": int(counts[2]),
        "valid_times": int(counts[3]),
        "duplicate_model_station_time_rows": int(counts[4]),
        "duplicate_coordinate_groups": int(duplicate_coordinates),
        "maximum_sigma_zero_abs_error_c": float(max(validation_errors)) if validation_errors else None,
        "sigma_zero_tolerance_c": 1e-6,
        "passed": counts[4] == 0 and (not validation_errors or max(validation_errors) <= 1e-6),
    }
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": cfg["analysis_role"],
        "config": {"path": config_path.relative_to(ROOT).as_posix(), "sha256": sha256(config_path)},
        "script_sha256": sha256(Path(__file__).resolve()),
        "spherical_module_sha256": sha256(ROOT / "src" / "spherical_roughness.py"),
        "panel": {"path": panel_path.relative_to(ROOT).as_posix(), "sha256": sha256(panel_path)},
        "model_specification_sha256": sha256(model_spec_path),
        "source_config_sha256": sha256(source_config_path),
        "definition": {
            "sigma_km": sigmas,
            "positive_sigma_kernel": "area-weighted geodesic Gaussian",
            "truncation_sigma": truncation_sigma,
            "zero_sigma_sampling": "bilinear sampling of the unchanged source field",
            "station_targets": int(len(stations)),
        },
        "sample_times": args.sample_times,
        "kernels": kernel_manifest,
        "sources": source_manifest,
        "synthetic_kernel_self_test": self_test(),
        "validation": validation,
        "output": {"path": final_path.relative_to(ROOT).as_posix(), "bytes": final_path.stat().st_size, "sha256": sha256(final_path)},
        "interpretation_limit": cfg["interpretation"]["shared_limit"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))
    if not validation["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
