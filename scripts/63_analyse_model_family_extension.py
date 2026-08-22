#!/usr/bin/env python3
"""Analyse the prospectively frozen five-family WeatherBench2 extension."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "model_family_extension_2020.yaml"
DEFAULT_ERRORS = ROOT / "results" / "weather5k_model_family_extension_2020" / "cell_day_errors.parquet"
DEFAULT_ROUGHNESS = ROOT / "data" / "interim" / "weather5k_model_family_roughness_2020" / "roughness.parquet"
DEFAULT_OUTPUT = ROOT / "results" / "weather5k_model_family_analysis_2020"


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def slope(x: np.ndarray, y: np.ndarray) -> float:
    centred = x - x.mean()
    denominator = float(np.dot(centred, centred))
    return float(np.dot(centred, y - y.mean()) / denominator) if denominator else float("nan")


def circular_block_counts(
    rng: np.random.Generator, n_days: int, width: int, n_resamples: int
) -> np.ndarray:
    blocks = int(np.ceil(n_days / width))
    counts = np.zeros((n_resamples, n_days), dtype=np.int16)
    offsets = np.arange(width)
    for draw in range(n_resamples):
        starts = rng.integers(0, n_days, size=blocks)
        selected = ((starts[:, None] + offsets) % n_days).ravel()[:n_days]
        counts[draw] = np.bincount(selected, minlength=n_days)
    return counts


def weighted_equal_cell_draws(cube: np.ndarray, day_counts: np.ndarray, chunk: int = 100) -> np.ndarray:
    """Day-block draws; average time within cell, then weight cells equally."""
    models, cells, days = cube.shape
    values = np.nan_to_num(cube, nan=0.0).reshape(models * cells, days)
    mask = np.isfinite(cube).astype(np.float64).reshape(models * cells, days)
    output = np.empty((models, len(day_counts)), dtype=np.float64)
    for start in range(0, len(day_counts), chunk):
        weights = day_counts[start : start + chunk].T.astype(np.float64)
        numerator = values @ weights
        denominator = mask @ weights
        cell_means = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 0,
        ).reshape(models, cells, -1)
        output[:, start : start + weights.shape[1]] = np.nanmean(cell_means, axis=1)
    return output


def rank_reversals(first: np.ndarray, second: np.ndarray) -> int:
    reversals = 0
    for left in range(len(first)):
        for right in range(left + 1, len(first)):
            if np.sign(first[left] - first[right]) != np.sign(second[left] - second[right]):
                reversals += 1
    return reversals


def interval(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.nanmedian(values)),
        "ci_low": float(np.nanquantile(values, 0.025)),
        "ci_high": float(np.nanquantile(values, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--errors", type=Path, default=DEFAULT_ERRORS)
    parser.add_argument("--roughness", type=Path, default=DEFAULT_ROUGHNESS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path, errors_path, roughness_path = (
        args.config.resolve(), args.errors.resolve(), args.roughness.resolve()
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    output_names = ["model_metrics.csv", "slopes.csv", "rank_comparison.csv", "summary.json"]
    if any((output / name).exists() for name in output_names) and not args.force:
        raise FileExistsError(f"outputs exist in {output}; use --force")
    if args.force:
        for name in output_names:
            path = output / name
            if path.exists():
                if not path.is_file():
                    raise RuntimeError(f"refusing to replace non-file {path}")
                path.unlink()

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    scales = [int(value) for value in cfg["roughness"]["scales_km"]]
    primary_scale = int(cfg["roughness"]["primary_scale_km"])
    primary_models = cfg["prespecification"]["primary_models"]
    sensitivity_models = cfg["prespecification"]["sensitivity_models"]
    all_models = primary_models + sensitivity_models

    rough_columns = ", ".join(f"r.roughness_{scale}km_k" for scale in scales)
    connection = duckdb.connect()
    joined = connection.execute(
        f"""
        SELECT e.model, e.model_family, e.cohort, e.cell_id,
               cast(e.valid_day AS DATE) AS valid_day,
               e.era5_mae_c, e.observed_mae_c, e.era5_mse_c2, e.observed_mse_c2,
               {rough_columns}
        FROM read_parquet('{errors_path.as_posix()}') e
        JOIN read_parquet('{roughness_path.as_posix()}') r
          USING (model, cell_id, valid_day)
        WHERE e.model IN ({','.join('?' for _ in all_models)})
        ORDER BY e.model, e.cell_id, e.valid_day
        """,
        all_models,
    ).fetchdf()
    connection.close()
    present = sorted(joined.model.unique())
    if present != sorted(all_models):
        raise RuntimeError(f"model mismatch: {present} != {sorted(all_models)}")

    cells = np.sort(joined.cell_id.astype(str).unique())
    days = pd.DatetimeIndex(sorted(pd.to_datetime(joined.valid_day).unique()))
    cell_map = {value: index for index, value in enumerate(cells)}
    day_map = {pd.Timestamp(value).value: index for index, value in enumerate(days)}
    shape = (len(all_models), len(cells), len(days))
    columns = [
        "era5_mae_c", "observed_mae_c", "era5_mse_c2", "observed_mse_c2",
        *[f"roughness_{scale}km_k" for scale in scales],
    ]
    cubes = {column: np.full(shape, np.nan, dtype=np.float64) for column in columns}
    model_map = {name: index for index, name in enumerate(all_models)}
    mi = joined.model.map(model_map).to_numpy()
    ci = joined.cell_id.astype(str).map(cell_map).to_numpy()
    di = pd.to_datetime(joined.valid_day).map(lambda value: day_map[value.value]).to_numpy()
    for column, cube in cubes.items():
        cube[mi, ci, di] = joined[column].to_numpy(float)
    common = np.ones((len(cells), len(days)), dtype=bool)
    for cube in cubes.values():
        common &= np.all(np.isfinite(cube), axis=0)
    if not common.any():
        raise RuntimeError("no common model-cell-day support")
    for cube in cubes.values():
        cube[:, ~common] = np.nan

    point = {
        column: np.nanmean(np.nanmean(cube, axis=2), axis=1)
        for column, cube in cubes.items()
    }
    point["era5_rmse_c"] = np.sqrt(point["era5_mse_c2"])
    point["observed_rmse_c"] = np.sqrt(point["observed_mse_c2"])
    families = (
        joined.groupby("model", as_index=False)
        .agg(model_family=("model_family", "first"), cohort=("cohort", "first"))
        .set_index("model")
        .loc[all_models]
        .reset_index()
    )
    metrics = families.copy()
    metrics["era5_mae_c"] = point["era5_mae_c"]
    metrics["observed_mae_c"] = point["observed_mae_c"]
    metrics["mae_advantage_c"] = metrics.era5_mae_c - metrics.observed_mae_c
    metrics["era5_rmse_c"] = point["era5_rmse_c"]
    metrics["observed_rmse_c"] = point["observed_rmse_c"]
    metrics["rmse_advantage_c"] = metrics.era5_rmse_c - metrics.observed_rmse_c
    for scale in scales:
        metrics[f"roughness_{scale}km_k"] = point[f"roughness_{scale}km_k"]
    metrics.to_csv(output / "model_metrics.csv", index=False)

    primary_positions = np.array([model_map[name] for name in primary_models])
    slope_rows = []
    for metric in ("mae", "rmse"):
        advantage = metrics[f"{metric}_advantage_c"].to_numpy()
        for scale in scales:
            roughness = metrics[f"roughness_{scale}km_k"].to_numpy()
            x, y = roughness[primary_positions], advantage[primary_positions]
            slope_rows.append({
                "metric": metric,
                "scale_km": scale,
                "slope_c_per_k": slope(x, y),
                "spearman": float(spearmanr(x, y).statistic),
                "families": len(primary_positions),
            })

    rng = np.random.default_rng(args.seed)
    day_counts = circular_block_counts(rng, len(days), 7, args.bootstrap_resamples)
    draws = {column: weighted_equal_cell_draws(cube, day_counts) for column, cube in cubes.items()}
    draws["era5_rmse_c"] = np.sqrt(draws["era5_mse_c2"])
    draws["observed_rmse_c"] = np.sqrt(draws["observed_mse_c2"])
    slope_draws: dict[str, np.ndarray] = {}
    for row in slope_rows:
        metric, scale = row["metric"], int(row["scale_km"])
        advantage_draws = (
            draws[f"era5_{metric}_c"] - draws[f"observed_{metric}_c"]
        )[primary_positions]
        roughness_draws = draws[f"roughness_{scale}km_k"][primary_positions]
        values = np.array([
            slope(roughness_draws[:, draw], advantage_draws[:, draw])
            for draw in range(args.bootstrap_resamples)
        ])
        key = f"{metric}_{scale}km"
        slope_draws[key] = values
        row.update(interval(values))
        row["ci_excludes_zero_positive"] = bool(row["ci_low"] > 0)
    slopes = pd.DataFrame(slope_rows)
    slopes.to_csv(output / "slopes.csv", index=False)
    np.savez_compressed(output / "bootstrap_slopes.npz", **slope_draws)

    rank_rows = []
    primary_metrics = metrics.iloc[primary_positions]
    for metric in ("mae", "rmse"):
        era5 = primary_metrics[f"era5_{metric}_c"].to_numpy()
        observed = primary_metrics[f"observed_{metric}_c"].to_numpy()
        rank_rows.append({
            "metric": metric,
            "spearman_between_reference_rankings": float(spearmanr(era5, observed).statistic),
            "pairwise_rank_reversals": rank_reversals(era5, observed),
            "possible_pairs": int(len(era5) * (len(era5) - 1) / 2),
            "era5_order_best_to_worst": "|".join(primary_metrics.iloc[np.argsort(era5)].model),
            "station_order_best_to_worst": "|".join(primary_metrics.iloc[np.argsort(observed)].model),
        })
    ranks = pd.DataFrame(rank_rows)
    ranks.to_csv(output / "rank_comparison.csv", index=False)

    all_point_positive = bool((slopes.slope_c_per_k > 0).all())
    primary_ci_positive = bool(
        (slopes[slopes.scale_km == primary_scale].ci_low > 0).all()
    )
    sensitivity = metrics[metrics.model.isin(sensitivity_models)]
    sensitivity_no_reversal = bool(
        (sensitivity.mae_advantage_c < 0).all() and (sensitivity.rmse_advantage_c < 0).all()
    )
    success = all_point_positive and sensitivity_no_reversal
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": cfg["analysis_role"],
        "config": {"path": config_path.relative_to(ROOT).as_posix(), "sha256": sha256(config_path)},
        "inputs": {
            "errors": {"path": errors_path.relative_to(ROOT).as_posix(), "sha256": sha256(errors_path)},
            "roughness": {"path": roughness_path.relative_to(ROOT).as_posix(), "sha256": sha256(roughness_path)},
        },
        "support": {
            "models": len(all_models),
            "primary_deterministic_families": len(primary_models),
            "cells": len(cells),
            "days": len(days),
            "common_model_cell_days": int(common.sum()),
            "first_day": days.min().date().isoformat(),
            "last_day": days.max().date().isoformat(),
        },
        "bootstrap": {
            "unit": "valid day",
            "circular_block_days": 7,
            "resamples": args.bootstrap_resamples,
            "seed": args.seed,
            "spatial_weighting": "equal cell after time aggregation within cell",
        },
        "primary_scale_km": primary_scale,
        "slope_results": slopes.to_dict(orient="records"),
        "rank_results": ranks.to_dict(orient="records"),
        "gates": {
            "positive_point_slope_both_metrics_all_scales": all_point_positive,
            "primary_95ci_positive_both_metrics": primary_ci_positive,
            "gencast_advantage_direction_not_reversed": sensitivity_no_reversal,
            "prespecified_mechanistic_breadth_supported": success,
        },
        "interpretation_limit": cfg["interpretation"]["warning"],
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest_names = [*output_names, "bootstrap_slopes.npz"]
    (output / "outputs_manifest.json").write_text(json.dumps({
        "script_sha256": sha256(Path(__file__).resolve()),
        "outputs": {
            name: {"bytes": (output / name).stat().st_size, "sha256": sha256(output / name)}
            for name in manifest_names
        },
    }, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(slopes.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(json.dumps(report["gates"], indent=2))


if __name__ == "__main__":
    main()
