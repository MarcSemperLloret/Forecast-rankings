#!/usr/bin/env python3
"""Global Weather5K validation of the AVAMET reference-sensitivity mechanism.

This is deliberately labelled a global validation, not an independent
confirmation: Weather5K/ISD can overlap observations assimilated by ERA5 and
the source variants are related. The script uses the already audited
model-cell-day error and roughness tables, jointly resamples both axes, and
reports three views without conflating their statistical units:

1. all seven source variants (descriptive only);
2. each multi-model initialisation cohort separately;
3. one cohort-centred row per named model family, excluding singleton cohorts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "regional_t2m_2020.yaml"
DEFAULT_ERRORS = ROOT / "results" / "weather5k_weatherbench2_dry_run_2020" / "cell_day_errors.parquet"
DEFAULT_ROUGHNESS = ROOT / "data" / "interim" / "weather5k_global_roughness_dry_run_2020" / "roughness.parquet"
DEFAULT_SITES = ROOT / "results" / "weather5k_weatherbench2_dry_run_2020" / "sites.csv"
DEFAULT_OUTPUT = ROOT / "results" / "weather5k_global_validation_2020"
SCALES = (25, 50, 100)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_seal() -> str:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "40_freeze_analysis_parameters.py"), "--verify"],
        capture_output=True, text=True,
    )
    if result.returncode:
        raise SystemExit(f"sealed parameters do not match\n{result.stderr}")
    return result.stdout.strip()


def verify_input_hash(path: Path, manifest: Path, keys: tuple[str, ...]) -> dict:
    record = json.loads(manifest.read_text(encoding="utf-8"))
    node = record
    for key in keys:
        node = node[key]
    expected = node["sha256"]
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"input hash mismatch for {path}: {actual} != {expected}")
    return {"path": str(path), "sha256": actual, "manifest": str(manifest)}


def slope(x: np.ndarray, y: np.ndarray) -> float:
    usable = np.isfinite(x) & np.isfinite(y)
    if usable.sum() < 2 or np.ptp(x[usable]) == 0:
        return float("nan")
    return float(np.polyfit(x[usable], y[usable], 1)[0])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def aggregate(cube: np.ndarray, cell_index: np.ndarray | None = None,
              day_index: np.ndarray | None = None) -> np.ndarray:
    """Days within cells, then equal weight across cells."""
    values = cube
    if day_index is not None:
        values = values[:, :, day_index]
    by_cell = np.nanmean(values, axis=2)
    if cell_index is not None:
        by_cell = by_cell[:, cell_index]
    return np.nanmean(by_cell, axis=1)


def reduce_families(values: pd.DataFrame, column: str) -> pd.DataFrame:
    """Centre within comparable cohorts, then average duplicate families.

    Singleton cohorts have no within-cohort contrast and are excluded rather
    than converted into a fabricated zero-effect model unit.
    """
    counts = values.groupby("cohort").model.transform("nunique")
    comparable = values[counts >= 2].copy()
    comparable[f"{column}_centred"] = comparable[column] - comparable.groupby("cohort")[column].transform("mean")
    return (comparable.groupby("model_family", as_index=False)[f"{column}_centred"].mean()
            .rename(columns={f"{column}_centred": column}))


def family_points(metadata: pd.DataFrame, roughness: np.ndarray, advantage: np.ndarray) -> pd.DataFrame:
    base = metadata.copy()
    base["roughness_k"] = roughness
    base["advantage_c"] = advantage
    reduced_r = reduce_families(base, "roughness_k")
    reduced_b = reduce_families(base, "advantage_c")
    return reduced_r.merge(reduced_b, on="model_family", validate="one_to_one")


def point_tables(metadata: pd.DataFrame, era5: np.ndarray, observed: np.ndarray,
                 roughness: dict[int, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    advantage = era5 - observed
    variants = metadata.copy()
    variants["mae_era5_c"] = era5
    variants["mae_observed_c"] = observed
    variants["advantage_c"] = advantage
    for scale in SCALES:
        variants[f"roughness_{scale}km_k"] = roughness[scale]

    family_frames = []
    for scale in SCALES:
        frame = family_points(metadata, roughness[scale], advantage)
        frame["scale_km"] = scale
        family_frames.append(frame)
    families = pd.concat(family_frames, ignore_index=True)

    cohort_rows = []
    for cohort, positions in metadata.groupby("cohort").groups.items():
        index = np.array(list(positions), dtype=int)
        if len(index) < 2:
            continue
        for scale in SCALES:
            cohort_rows.append({
                "cohort": cohort,
                "scale_km": scale,
                "n_variants": len(index),
                "slope_c_per_k": slope(roughness[scale][index], advantage[index]),
                "spearman": spearman(roughness[scale][index], advantage[index]),
            })
    return variants, families, pd.DataFrame(cohort_rows)


def geographic_tables(metadata: pd.DataFrame, cells: list[str], sites: pd.DataFrame,
                      era5_cube: np.ndarray, observed_cube: np.ndarray,
                      roughness_cubes: dict[int, np.ndarray],
                      common: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descriptive latitude-belt sensitivity; it does not alter the verdict."""
    cell_latitude = sites.groupby("cell_id").mean_latitude.mean().reindex(cells)
    if cell_latitude.isna().any():
        missing = cell_latitude[cell_latitude.isna()].index.tolist()[:5]
        raise SystemExit(f"site metadata missing for occupied cells: {missing}")
    latitude = cell_latitude.to_numpy()
    strata = {
        "south_extratropics": latitude < -23.5,
        "tropics": np.abs(latitude) <= 23.5,
        "north_extratropics": latitude > 23.5,
    }
    definitions = {
        "south_extratropics": "latitude < -23.5",
        "tropics": "abs(latitude) <= 23.5",
        "north_extratropics": "latitude > 23.5",
    }
    summary_rows: list[dict] = []
    family_rows: list[pd.DataFrame] = []
    for stratum, selected in strata.items():
        era5 = aggregate(era5_cube[:, selected, :])
        observed = aggregate(observed_cube[:, selected, :])
        roughness = {
            scale: aggregate(roughness_cubes[scale][:, selected, :])
            for scale in SCALES
        }
        _, families, _ = point_tables(metadata, era5, observed, roughness)
        families.insert(0, "stratum", stratum)
        families.insert(1, "n_cells", int(selected.sum()))
        families.insert(2, "common_cell_days", int(common[selected].sum()))
        family_rows.append(families)
        for scale in SCALES:
            frame = families[families.scale_km == scale]
            leave_slopes = []
            for family in frame.model_family:
                kept = frame[frame.model_family != family]
                leave_slopes.append(slope(
                    kept.roughness_k.to_numpy(), kept.advantage_c.to_numpy()))
            leave_array = np.asarray(leave_slopes)
            summary_rows.append({
                "stratum": stratum,
                "latitude_definition": definitions[stratum],
                "n_cells": int(selected.sum()),
                "common_cell_days": int(common[selected].sum()),
                "scale_km": scale,
                "n_families": int(len(frame)),
                "slope_c_per_k": slope(
                    frame.roughness_k.to_numpy(), frame.advantage_c.to_numpy()),
                "spearman": spearman(
                    frame.roughness_k.to_numpy(), frame.advantage_c.to_numpy()),
                "leave_one_family_out_min_slope_c_per_k": float(np.nanmin(leave_array)),
                "leave_one_family_out_max_slope_c_per_k": float(np.nanmax(leave_array)),
                "leave_one_family_out_all_positive": bool((leave_array > 0).all()),
            })
    return pd.DataFrame(summary_rows), pd.concat(family_rows, ignore_index=True)


def circular_blocks(rng: np.random.Generator, n_days: int, width: int) -> np.ndarray:
    starts = rng.integers(0, n_days, size=int(np.ceil(n_days / width)))
    return ((starts[:, None] + np.arange(width)) % n_days).ravel()[:n_days]


def to_family_cube(metadata: pd.DataFrame, cube: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Linear cohort centring and duplicate-family averaging at cell-day grain."""
    members: dict[str, list[np.ndarray]] = {}
    for cohort, positions in metadata.groupby("cohort").groups.items():
        index = np.array(list(positions), dtype=int)
        if len(index) < 2:
            continue
        subset = cube[index]
        centred = subset - np.mean(subset, axis=0)
        for local, position in enumerate(index):
            family = str(metadata.iloc[position].model_family)
            members.setdefault(family, []).append(centred[local])
    families = sorted(members)
    reduced = np.stack([np.mean(np.stack(members[family]), axis=0) for family in families])
    return families, reduced


def bootstrap_weights(n_cells: int, n_days: int, widths: list[int], n_resamples: int,
                      seed: int) -> tuple[np.ndarray, np.ndarray, dict[int, slice]]:
    """Exact multinomial weights for the frozen cell and circular-block draws."""
    total = len(widths) * n_resamples
    cell_weights = np.zeros((n_cells, total), dtype=float)
    day_weights = np.zeros((n_days, total), dtype=float)
    slices = {}
    offset = 0
    for width in widths:
        rng = np.random.default_rng(seed + int(width))
        slices[int(width)] = slice(offset, offset + n_resamples)
        for draw in range(n_resamples):
            column = offset + draw
            sampled_cells = rng.integers(0, n_cells, size=n_cells)
            sampled_days = circular_blocks(rng, n_days, int(width))
            cell_weights[:, column] = np.bincount(sampled_cells, minlength=n_cells)
            day_weights[:, column] = np.bincount(sampled_days, minlength=n_days)
        offset += n_resamples
    return cell_weights, day_weights, slices


def weighted_draws(cube: np.ndarray, cell_weights: np.ndarray, day_weights: np.ndarray,
                   valid_counts: np.ndarray) -> np.ndarray:
    """Days within cells, then equal resampled-cell weight, for every draw."""
    output = np.full((cube.shape[0], day_weights.shape[1]), np.nan)
    usable_cells = valid_counts > 0
    denominator = np.einsum("cd,cd->d", cell_weights, usable_cells)
    for unit in range(cube.shape[0]):
        sums = np.nan_to_num(cube[unit]) @ day_weights
        by_cell = np.divide(
            sums, valid_counts,
            out=np.full_like(sums, np.nan), where=usable_cells,
        )
        numerator = np.einsum("cd,cd->d", cell_weights, np.nan_to_num(by_cell))
        output[unit] = numerator / denominator
    return output


def vector_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """One OLS slope per column, with family units in rows."""
    x_centre = x - x.mean(axis=0, keepdims=True)
    y_centre = y - y.mean(axis=0, keepdims=True)
    denominator = np.sum(x_centre ** 2, axis=0)
    return np.divide(
        np.sum(x_centre * y_centre, axis=0), denominator,
        out=np.full(x.shape[1], np.nan), where=denominator > 0,
    )


def interval(values: np.ndarray, confidence: float) -> dict:
    tail = (1 - confidence) / 2
    return {
        "mean": float(np.nanmean(values)),
        "ci_low": float(np.nanquantile(values, tail)),
        "ci_high": float(np.nanquantile(values, 1 - tail)),
        "p_slope_gt_zero": float(np.nanmean(values > 0)),
        "finite_draws": int(np.isfinite(values).sum()),
    }


def plot_family(families: pd.DataFrame, output: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, scale in zip(axes, SCALES, strict=True):
        frame = families[families.scale_km == scale]
        x, y = frame.roughness_k.to_numpy(), frame.advantage_c.to_numpy()
        axis.scatter(x, y, s=55, color="#1f6f8b")
        for row in frame.itertuples():
            axis.annotate(row.model_family, (row.roughness_k, row.advantage_c),
                          xytext=(4, 4), textcoords="offset points", fontsize=8)
        if len(frame) >= 2:
            line_x = np.linspace(x.min(), x.max(), 100)
            axis.plot(line_x, np.polyval(np.polyfit(x, y, 1), line_x), color="#c94c4c")
        axis.axhline(0, color="0.6", linewidth=0.8)
        axis.set(title=f"{scale} km", xlabel="Rugosidad centrada en cohorte (K)")
    axes[0].set_ylabel("B centrado en cohorte: MAE ERA5 − estación (°C)")
    fig.suptitle(title)
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--errors", type=Path, default=DEFAULT_ERRORS)
    parser.add_argument("--roughness", type=Path, default=DEFAULT_ROUGHNESS)
    parser.add_argument("--sites", type=Path, default=DEFAULT_SITES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--errors-manifest", type=Path, default=None)
    parser.add_argument("--roughness-manifest", type=Path, default=None)
    parser.add_argument("--sites-manifest", type=Path, default=None)
    parser.add_argument(
        "--analysis-label",
        default="Weather5K global validation of the AVAMET reference-sensitivity mechanism",
    )
    parser.add_argument(
        "--analysis-role",
        default="global observational validation; not independent confirmation and not a new held-out test",
    )
    parser.add_argument(
        "--observation-limit",
        default=(
            "WEATHER-5K/ISD may overlap observations assimilated by ERA5; related model "
            "systems also prevent fully independent confirmation."
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    known = [args.output / name for name in (
        "summary.json", "source_variant_metrics.csv", "family_centred_metrics.csv",
        "cohort_slopes.csv", "leave_one_family_out.csv", "bootstrap_slopes.npz",
        "geographic_sensitivity.csv", "geographic_family_metrics.csv",
        "outputs_manifest.json",
        "family_centred_validation.png", "family_centred_validation.pdf",
    )]
    if any(path.exists() for path in known) and not args.force:
        raise FileExistsError(f"outputs exist in {args.output}; use --force")

    seal = verify_seal()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    errors_manifest = (
        args.errors_manifest.resolve()
        if args.errors_manifest is not None
        else (args.errors.parent / "diagnostics.json").resolve()
    )
    roughness_manifest = (
        args.roughness_manifest.resolve()
        if args.roughness_manifest is not None
        else (args.roughness.parent / "manifest.json").resolve()
    )
    sites_manifest = (
        args.sites_manifest.resolve()
        if args.sites_manifest is not None
        else (args.sites.parent / "diagnostics.json").resolve()
    )
    error_provenance = verify_input_hash(
        args.errors,
        errors_manifest,
        ("outputs", "cell_day_errors.parquet"),
    )
    roughness_provenance = verify_input_hash(
        args.roughness,
        roughness_manifest,
        ("output",),
    )
    sites_provenance = verify_input_hash(
        args.sites,
        sites_manifest,
        ("outputs", "sites.csv"),
    )

    connection = duckdb.connect()
    metadata = connection.execute(
        f"""SELECT model, any_value(model_family) AS model_family,
                   any_value(cohort) AS cohort
            FROM read_parquet('{args.errors.as_posix()}')
            GROUP BY model ORDER BY model"""
    ).fetchdf()
    models = metadata.model.tolist()
    cells = connection.execute(
        f"SELECT DISTINCT cell_id FROM read_parquet('{args.errors.as_posix()}') ORDER BY cell_id"
    ).fetchdf().cell_id.tolist()
    days = connection.execute(
        f"SELECT DISTINCT valid_day FROM read_parquet('{args.errors.as_posix()}') ORDER BY valid_day"
    ).fetchdf().valid_day.tolist()
    model_index = {value: index for index, value in enumerate(models)}
    cell_index = {value: index for index, value in enumerate(cells)}
    day_index = {value: index for index, value in enumerate(days)}
    shape = (len(models), len(cells), len(days))
    era5_cube = np.full(shape, np.nan)
    observed_cube = np.full(shape, np.nan)
    roughness_cubes = {scale: np.full(shape, np.nan) for scale in SCALES}

    joined = connection.execute(
        f"""SELECT e.model, e.cell_id, e.valid_day, e.era5_mae_c, e.observed_mae_c,
                   r.roughness_25km_k, r.roughness_50km_k, r.roughness_100km_k
            FROM read_parquet('{args.errors.as_posix()}') e
            JOIN read_parquet('{args.roughness.as_posix()}') r
            USING (model, cell_id, valid_day)"""
    ).fetchdf()
    connection.close()
    mi = joined.model.map(model_index).to_numpy()
    ci = joined.cell_id.map(cell_index).to_numpy()
    di = joined.valid_day.map(day_index).to_numpy()
    era5_cube[mi, ci, di] = joined.era5_mae_c.to_numpy()
    observed_cube[mi, ci, di] = joined.observed_mae_c.to_numpy()
    for scale in SCALES:
        roughness_cubes[scale][mi, ci, di] = joined[f"roughness_{scale}km_k"].to_numpy()
    del joined

    common = np.all(np.isfinite(era5_cube) & np.isfinite(observed_cube), axis=0)
    for scale in SCALES:
        common &= np.all(np.isfinite(roughness_cubes[scale]), axis=0)
    if not common.any():
        raise SystemExit("no common model-cell-day support")
    for cube in (era5_cube, observed_cube, *roughness_cubes.values()):
        cube[:, ~common] = np.nan

    era5_point = aggregate(era5_cube)
    observed_point = aggregate(observed_cube)
    roughness_point = {scale: aggregate(roughness_cubes[scale]) for scale in SCALES}
    variants, families, cohorts = point_tables(
        metadata, era5_point, observed_point, roughness_point)
    variants.to_csv(args.output / "source_variant_metrics.csv", index=False)
    families.to_csv(args.output / "family_centred_metrics.csv", index=False)
    cohorts.to_csv(args.output / "cohort_slopes.csv", index=False)

    point_family = {}
    for scale in SCALES:
        frame = families[families.scale_km == scale]
        point_family[scale] = {
            "n_families": int(len(frame)),
            "slope_c_per_k": slope(frame.roughness_k.to_numpy(), frame.advantage_c.to_numpy()),
            "spearman": spearman(frame.roughness_k.to_numpy(), frame.advantage_c.to_numpy()),
        }

    primary = families[families.scale_km == 50].reset_index(drop=True)
    leave_rows = []
    for family in primary.model_family:
        frame = primary[primary.model_family != family]
        leave_rows.append({
            "excluded_family": family,
            "n_remaining": len(frame),
            "slope_c_per_k": slope(frame.roughness_k.to_numpy(), frame.advantage_c.to_numpy()),
            "spearman": spearman(frame.roughness_k.to_numpy(), frame.advantage_c.to_numpy()),
        })
    leave = pd.DataFrame(leave_rows)
    leave.to_csv(args.output / "leave_one_family_out.csv", index=False)

    sites = pd.read_csv(args.sites)
    geographic, geographic_families = geographic_tables(
        metadata, cells, sites, era5_cube, observed_cube, roughness_cubes, common)
    geographic.to_csv(args.output / "geographic_sensitivity.csv", index=False)
    geographic_families.to_csv(
        args.output / "geographic_family_metrics.csv", index=False)

    bootstrap_cfg = cfg["bootstrap"]
    widths = cfg["controlled_smoothing"]["bootstrap_block_days"]
    n_resamples = int(bootstrap_cfg["n_resamples"])
    cell_weights, day_weights, width_slices = bootstrap_weights(
        len(cells), len(days), [int(value) for value in widths], n_resamples,
        int(bootstrap_cfg["seed"]),
    )
    valid_counts = common.astype(float) @ day_weights
    family_names, family_advantage_cube = to_family_cube(
        metadata, era5_cube - observed_cube)
    if family_names != primary.model_family.sort_values().tolist():
        raise SystemExit("point and cell-day family reductions disagree")
    family_advantage_draws = weighted_draws(
        family_advantage_cube, cell_weights, day_weights, valid_counts)
    draws: dict[str, np.ndarray] = {}
    for scale in SCALES:
        roughness_families, family_roughness_cube = to_family_cube(
            metadata, roughness_cubes[scale])
        if roughness_families != family_names:
            raise SystemExit(f"family mismatch at {scale} km")
        family_roughness_draws = weighted_draws(
            family_roughness_cube, cell_weights, day_weights, valid_counts)
        all_slopes = vector_slopes(family_roughness_draws, family_advantage_draws)
        for width in widths:
            draws[f"scale_{scale}km_block_{width}d"] = all_slopes[width_slices[int(width)]]
        print(f"bootstrap scale={scale}km complete", flush=True)

    np.savez_compressed(args.output / "bootstrap_slopes.npz", **draws)
    intervals = {
        str(scale): {
            str(width): interval(draws[f"scale_{scale}km_block_{width}d"], float(bootstrap_cfg["ci"]))
            for width in widths
        }
        for scale in SCALES
    }
    leave_signs = np.sign(leave.slope_c_per_k.to_numpy())
    leave_stable = bool(np.isfinite(leave_signs).all() and (leave_signs > 0).all())
    widest = str(max(widths))
    primary_scale = 50
    all_scales_positive = all(point_family[scale]["slope_c_per_k"] > 0 for scale in SCALES)
    all_widest_intervals_positive = all(intervals[str(scale)][widest]["ci_low"] > 0 for scale in SCALES)
    primary_interval_positive = intervals[str(primary_scale)][widest]["ci_low"] > 0
    cohort_primary = cohorts[cohorts.scale_km == primary_scale]
    cohort_signs_consistent = bool(
        len(cohort_primary) > 0
        and (np.sign(cohort_primary.slope_c_per_k).nunique() == 1)
    )
    geographic_primary = geographic[geographic.scale_km == primary_scale]
    geographic_primary_all_positive = bool(
        (geographic_primary.slope_c_per_k > 0).all())
    association_replication = bool(
        primary_interval_positive and all_scales_positive and leave_stable)
    plot_family(families, args.output / "family_centred_validation", args.analysis_label)

    report = {
        "analysis": args.analysis_label,
        "role": args.analysis_role,
        "seal": seal,
        "inputs": {
            "errors": error_provenance,
            "roughness": roughness_provenance,
            "sites": sites_provenance,
        },
        "estimand": "B = equal-cell MAE against ERA5 minus equal-cell MAE against observed station T2m",
        "unit_policy": {
            "source_variants": len(models),
            "source_variant_regression_role": "descriptive only; variants are related",
            "family_reduction": "centre B and R within multi-model cohort, then average duplicate named families",
            "singleton_cohorts": "excluded from family regression because they provide no within-cohort contrast",
            "family_units": sorted(primary.model_family.tolist()),
        },
        "support": {
            "cells": len(cells), "days": len(days),
            "possible_cell_days": int(len(cells) * len(days)),
            "common_cell_days": int(common.sum()),
            "common_fraction": float(common.mean()),
        },
        "source_variant_metrics": variants.to_dict(orient="records"),
        "family_point_results": {str(key): value for key, value in point_family.items()},
        "cohort_results": cohorts.to_dict(orient="records"),
        "bootstrap_family_slopes": intervals,
        "leave_one_family_out": leave.to_dict(orient="records"),
        "leave_one_family_out_all_positive": leave_stable,
        "geographic_sensitivity_role": (
            "descriptive post-dry-run audit; not used to change the frozen global verdict"
        ),
        "geographic_sensitivity": geographic.to_dict(orient="records"),
        "validation_pattern": {
            "all_source_variants_have_lower_mae_against_era5": bool((variants.advantage_c < 0).all()),
            "primary_scale_km": primary_scale,
            "primary_95ci_above_zero_for_7d_blocks": bool(primary_interval_positive),
            "family_slope_positive_at_all_scales": bool(all_scales_positive),
            "family_95ci_above_zero_at_all_scales_for_7d_blocks": bool(all_widest_intervals_positive),
            "leave_one_family_out_all_positive_at_primary_scale": leave_stable,
            "comparable_cohort_slopes_have_same_sign_at_primary_scale": cohort_signs_consistent,
            "latitude_belt_slopes_all_positive_at_primary_scale": geographic_primary_all_positive,
            "association_replication_supported": association_replication,
            "rmse_requirement_evaluated": False,
            "global_controlled_smoothing_s1_s2_evaluated": False,
            "full_preregistered_confirmation_supported": False,
        },
        "confirmation": (
            "not supported by this sensitivity analysis: association diagnostics do not resolve "
            "observational independence, related model-system dependence, or the remaining "
            "pre-registered confirmation criteria"
        ),
        "interpretation_limit": args.observation_limit,
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output_names = [
        "summary.json", "source_variant_metrics.csv", "family_centred_metrics.csv",
        "cohort_slopes.csv", "leave_one_family_out.csv", "bootstrap_slopes.npz",
        "geographic_sensitivity.csv", "geographic_family_metrics.csv",
        "family_centred_validation.png", "family_centred_validation.pdf",
    ]
    output_manifest = {
        "analysis_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "outputs": {
            name: {
                "bytes": (args.output / name).stat().st_size,
                "sha256": sha256(args.output / name),
            }
            for name in output_names
        },
    }
    (args.output / "outputs_manifest.json").write_text(
        json.dumps(output_manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "family_point_results": report["family_point_results"],
        "validation_pattern": report["validation_pattern"],
        "support": report["support"],
    }, indent=2))


if __name__ == "__main__":
    main()
