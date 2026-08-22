#!/usr/bin/env python3
"""Extract a traceable WEATHER-5K × WeatherBench2 dry-run panel.

The script uses the exact station-times produced by script 44, intersects them
with every configured forecast source at +24 h and ERA5, then interpolates each
full gridded field bilinearly.  Forecast-source variants retain ``family`` and
``cohort`` columns: they are useful for an engineering dry run but are not
silently promoted to independent model units.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from public_zarr import PublicZarr, bilinear

DEFAULT_CONFIG = ROOT / "config" / "weather5k_model_dry_run_2020.yaml"


def resolve_from_root(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def index_of(values: np.ndarray | pd.DatetimeIndex, wanted: object, name: str) -> int:
    found = np.flatnonzero(np.asarray(values) == wanted)
    if len(found) != 1:
        raise RuntimeError(f"{name}={wanted} is missing or non-unique")
    return int(found[0])


def source_specs(
    specification: dict[str, Any], weatherbench_config: dict[str, Any]
) -> tuple[dict[str, dict], dict]:
    forecasts = dict(specification["weatherbench2"]["models"])
    forecasts.update(specification["model_extensions"])
    forecasts.update(weatherbench_config.get("model_specs", {}))
    reference = weatherbench_config.get(
        "reference_spec", specification["weatherbench2"]["era5"]
    )
    return forecasts, reference


def required_chunk_key(store: PublicZarr, spec: dict, time_index: int,
                       lead_index: int | None) -> str:
    metadata = store.array_metadata(spec["t2m_name"])
    shape, chunks = tuple(metadata["shape"]), tuple(metadata["chunks"])
    indices = [time_index // chunks[0]]
    if len(shape) == 4:
        if lead_index is None:
            raise ValueError("forecast field requires a lead index")
        indices.append(lead_index // chunks[1])
    indices.extend([0, 0])
    return f"{spec['t2m_name']}/" + ".".join(map(str, indices))


def common_valid_times(
    observations: pd.DatetimeIndex,
    stores: dict[str, PublicZarr],
    specs: dict[str, dict],
    reference_store: PublicZarr,
    reference_spec: dict,
    lead_hours: int,
) -> pd.DatetimeIndex:
    common = observations.unique().sort_values()
    lead = pd.Timedelta(hours=lead_hours)
    for name, store in stores.items():
        initialisations = store.times(specs[name]["time_name"])
        valid = initialisations + lead
        common = common.intersection(valid)
    reference_times = reference_store.times(reference_spec["time_name"])
    return common.intersection(reference_times).sort_values()


def cache_audit(
    store: PublicZarr,
    spec: dict,
    valid_times: pd.DatetimeIndex,
    lead_hours: int | None,
) -> dict[str, Any]:
    store_times = store.times(spec["time_name"])
    if lead_hours is None:
        wanted_times = valid_times
        lead_index = None
    else:
        wanted_times = valid_times - pd.Timedelta(hours=lead_hours)
        leads = store.timedeltas_hours(spec["lead_name"])
        lead_index = index_of(leads, lead_hours, "lead")
    keys = [
        required_chunk_key(store, spec, index_of(store_times, when, "time"), lead_index)
        for when in wanted_times
    ]
    paths = [store._cache_path(key) for key in keys]  # audit the reader's exact cache contract
    present = [path for path in paths if path.is_file()]
    return {
        "required_field_chunks": len(paths),
        "cached_field_chunks": len(present),
        "missing_field_chunks_before_run": len(paths) - len(present),
        "cached_bytes_before_run": sum(path.stat().st_size for path in present),
    }


def interpolate_series(
    store: PublicZarr,
    spec: dict,
    base: pd.DataFrame,
    valid_times: pd.DatetimeIndex,
    lead_hours: int | None,
    workers: int,
    label: str,
) -> np.ndarray:
    store_times = store.times(spec["time_name"])
    if lead_hours is None:
        wanted_times = valid_times
        lead_index = None
    else:
        wanted_times = valid_times - pd.Timedelta(hours=lead_hours)
        lead_index = index_of(store.timedeltas_hours(spec["lead_name"]), lead_hours, "lead")
    latitudes = store.coordinate(spec["latitude_name"])
    longitudes = store.coordinate(spec["longitude_name"])
    grouped = base.groupby("valid_time", sort=False).indices
    output = np.full(len(base), np.nan, dtype=np.float64)

    def one(item: tuple[pd.Timestamp, pd.Timestamp]) -> tuple[np.ndarray, np.ndarray]:
        valid, source_time = item
        rows = np.asarray(grouped[valid], dtype=np.int64)
        grid = store.geographic_field2d(
            spec["t2m_name"],
            index_of(store_times, source_time, "time"),
            lead_index,
            spec["latitude_name"],
            spec["longitude_name"],
        )
        values = bilinear(
            grid,
            latitudes,
            longitudes,
            base.latitude.to_numpy()[rows],
            base.longitude.to_numpy()[rows],
        ) - 273.15
        return rows, values

    items = list(zip(valid_times, wanted_times, strict=True))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for number, (rows, values) in enumerate(executor.map(one, items), start=1):
            output[rows] = values
            if number == 1 or number % 100 == 0 or number == len(items):
                print(f"{label}: {number}/{len(items)} fields", flush=True)
    return output


def write_parquet(connection: duckdb.DuckDBPyConnection, frame: pd.DataFrame, path: Path) -> None:
    connection.register("output_frame", frame)
    connection.execute(
        f"COPY output_frame TO '{path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    connection.unregister("output_frame")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--sample-stations", type=int, default=None)
    parser.add_argument("--sample-days", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    wb = config["weatherbench2"]
    execution = config["execution"]
    observations_path = resolve_from_root(config["observations"]["path"])
    observations_manifest = resolve_from_root(config["observations"]["manifest"])
    specification_path = resolve_from_root(wb["specification_config"])
    cache_root = resolve_from_root(wb["cache"])
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else resolve_from_root(execution["output_root"])
    )
    output_root.mkdir(parents=True, exist_ok=True)
    parts_root = output_root / "model_parts"
    parts_root.mkdir(parents=True, exist_ok=True)
    base_path = output_root / "base_cases.parquet"
    panel_path = output_root / "panel.parquet"
    manifest_path = output_root / "manifest.json"

    if args.force:
        for target in [base_path, panel_path, manifest_path, *parts_root.glob("*.parquet")]:
            if target.exists():
                if not target.is_file():
                    raise RuntimeError(f"Refusing to replace non-file output: {target}")
                target.unlink()
    elif panel_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Completed output exists in {output_root}; use --force to replace it")

    specification = yaml.safe_load(specification_path.read_text(encoding="utf-8"))
    all_forecast_specs, reference_spec = source_specs(specification, wb)
    source_rows = wb["sources"]
    selected_names = [row["model"] for row in source_rows]
    missing_specs = [name for name in selected_names if name not in all_forecast_specs]
    if missing_specs:
        raise ValueError(f"No WeatherBench2 specifications for: {missing_specs}")
    forecast_specs = {name: all_forecast_specs[name] for name in selected_names}
    stores = {name: PublicZarr(spec["path"], cache_root) for name, spec in forecast_specs.items()}
    reference_store = PublicZarr(reference_spec["path"], cache_root)

    query = f"SELECT * FROM read_parquet('{observations_path.as_posix()}')"
    if args.sample_stations is not None:
        if args.sample_stations < 1:
            raise ValueError("--sample-stations must be at least 1")
        query += (
            " WHERE station_id IN (SELECT station_id FROM read_parquet("
            f"'{observations_path.as_posix()}') GROUP BY station_id ORDER BY station_id "
            f"LIMIT {args.sample_stations})"
        )
    query += " ORDER BY valid_time, station_id"
    connection = duckdb.connect()
    base = connection.execute(query).fetchdf()
    base["valid_time"] = pd.to_datetime(base.valid_time)
    observation_times = pd.DatetimeIndex(base.valid_time.unique())
    lead_hours = int(wb["lead_hours"])
    valid_times = common_valid_times(
        observation_times,
        stores,
        forecast_specs,
        reference_store,
        reference_spec,
        lead_hours,
    )
    if args.sample_days is not None:
        if args.sample_days < 1:
            raise ValueError("--sample-days must be at least 1")
        selected_days = pd.Index(valid_times.normalize().unique()[: args.sample_days])
        valid_times = valid_times[valid_times.normalize().isin(selected_days)]
    base = base[base.valid_time.isin(valid_times)].reset_index(drop=True)
    if base.empty:
        raise RuntimeError("No common observation/model/reference cases remain")

    cache_before = {
        name: cache_audit(store, forecast_specs[name], valid_times, lead_hours)
        for name, store in stores.items()
    }
    cache_before[wb["reference"]] = cache_audit(
        reference_store, reference_spec, valid_times, None
    )
    print(
        f"Common support: {len(valid_times)} valid times, {base.station_id.nunique()} stations, "
        f"{len(base):,} observed cases",
        flush=True,
    )

    if not base_path.exists():
        base["era5_t2m_c"] = interpolate_series(
            reference_store,
            reference_spec,
            base,
            valid_times,
            None,
            int(execution["workers"]),
            wb["reference"],
        )
        write_parquet(connection, base, base_path)
    else:
        base = connection.execute(
            f"SELECT * FROM read_parquet('{base_path.as_posix()}') ORDER BY valid_time, station_id"
        ).fetchdf()
        base["valid_time"] = pd.to_datetime(base.valid_time)

    source_manifest: dict[str, Any] = {}
    for row in source_rows:
        name = row["model"]
        destination = parts_root / f"{name}.parquet"
        if destination.exists():
            print(f"Reusing completed part: {destination.name}", flush=True)
        else:
            forecast = interpolate_series(
                stores[name],
                forecast_specs[name],
                base,
                valid_times,
                lead_hours,
                int(execution["workers"]),
                name,
            )
            panel_columns = {
                    "station_id": base.station_id,
                    "latitude": base.latitude,
                    "longitude": base.longitude,
                    "altitude_m": base.altitude_m,
                    "network": base.network,
                    "potentially_assimilated": base.potentially_assimilated,
                    "valid_time": base.valid_time,
                    "model": name,
                    "model_family": row["family"],
                    "cohort": row["cohort"],
                    "lead_h": lead_hours,
                    "forecast_t2m_c": forecast,
                    "era5_t2m_c": base.era5_t2m_c,
                    "observed_t2m_c": base.observed_t2m_c,
                }
            if "assimilation_risk_class" in base.columns:
                panel_columns["assimilation_risk_class"] = base.assimilation_risk_class
            frame = pd.DataFrame(panel_columns)
            write_parquet(connection, frame, destination)
        counts = connection.execute(
            f"""SELECT count(*), count(*) FILTER (
                    WHERE forecast_t2m_c IS NULL OR era5_t2m_c IS NULL OR observed_t2m_c IS NULL)
                 FROM read_parquet('{destination.as_posix()}')"""
        ).fetchone()
        source_manifest[name] = {
            "family": row["family"],
            "cohort": row["cohort"],
            "rows": int(counts[0]),
            "rows_with_missing_required_value": int(counts[1]),
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": sha256(destination),
            "cache_before_run": cache_before[name],
        }

    part_glob = (parts_root / "*.parquet").as_posix()
    connection.execute(
        f"COPY (SELECT * FROM read_parquet('{part_glob}') ORDER BY model, valid_time, station_id) "
        f"TO '{panel_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    duplicate_groups = connection.execute(
        f"""SELECT count(*) FROM (
                SELECT station_id, valid_time, model
                FROM read_parquet('{panel_path.as_posix()}')
                GROUP BY ALL HAVING count(*) > 1
             )"""
    ).fetchone()[0]
    panel_counts = connection.execute(
        f"""SELECT count(*), count(DISTINCT station_id), count(DISTINCT valid_time),
                   count(DISTINCT model), min(valid_time), max(valid_time)
            FROM read_parquet('{panel_path.as_posix()}')"""
    ).fetchone()
    connection.close()

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": config["analysis_role"],
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "script_sha256": sha256(Path(__file__).resolve()),
        "observation_input": {
            "path": str(observations_path),
            "sha256": sha256(observations_path),
            "manifest": str(observations_manifest),
            "manifest_sha256": sha256(observations_manifest),
        },
        "weatherbench2_specification": {
            "path": str(specification_path),
            "sha256": sha256(specification_path),
            "lead_hours": lead_hours,
            "reference": wb["reference"],
        },
        "selection": {
            "sample_stations": args.sample_stations,
            "sample_days": args.sample_days,
            "common_valid_times": int(panel_counts[2]),
            "first_valid_time": panel_counts[4].isoformat(),
            "last_valid_time": panel_counts[5].isoformat(),
        },
        "counts": {
            "rows": int(panel_counts[0]),
            "stations": int(panel_counts[1]),
            "source_variants": int(panel_counts[3]),
            "duplicate_station_time_model_groups": int(duplicate_groups),
        },
        "sources": source_manifest,
        "reference_cache_before_run": cache_before[wb["reference"]],
        "outputs": {
            "base_cases": {
                "path": str(base_path), "bytes": base_path.stat().st_size, "sha256": sha256(base_path)
            },
            "panel": {
                "path": str(panel_path), "bytes": panel_path.stat().st_size, "sha256": sha256(panel_path)
            },
        },
        "validation_passed": duplicate_groups == 0
        and all(item["rows_with_missing_required_value"] == 0 for item in source_manifest.values()),
        "interpretation_warning": config["interpretation"]["warning"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"Wrote {panel_counts[0]:,} rows ({panel_counts[3]} source variants) to {panel_path}",
        flush=True,
    )
    print(f"Validation passed: {manifest['validation_passed']}", flush=True)
    if not manifest["validation_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
