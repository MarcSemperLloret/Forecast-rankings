#!/usr/bin/env python3
"""Run the national reference-support ladders (S2) with frozen density gates."""
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

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "national_causal_extension_2020.yaml"
EARTH_RADIUS_KM = 6371.0088


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    tail = (1.0 - confidence) / 2.0
    return tuple(np.quantile(values, [tail, 1.0 - tail]).tolist())


def block_indices(rng: np.random.Generator, days: int, draws: int, width: int) -> np.ndarray:
    starts = rng.integers(0, days, size=(draws, int(np.ceil(days / width))))
    return ((starts[:, :, None] + np.arange(width)) % days).reshape(draws, -1)[:, :days]


def haversine_matrix(stations: pd.DataFrame) -> np.ndarray:
    latitude = np.deg2rad(stations.latitude.to_numpy(float))[:, None]
    longitude = np.deg2rad(stations.longitude.to_numpy(float))[:, None]
    haversine = (
        np.sin((latitude - latitude.T) / 2.0) ** 2
        + np.cos(latitude) * np.cos(latitude.T) * np.sin((longitude - longitude.T) / 2.0) ** 2
    )
    return EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))


def weight_matrix(distance: np.ndarray, centres: np.ndarray, radius: float, weighting: str) -> np.ndarray:
    if radius == 0:
        return np.eye(len(distance), dtype=float)[centres]
    inside = distance[centres] <= radius
    if weighting == "uniform":
        return inside.astype(float)
    if weighting == "gaussian":
        return np.where(
            inside,
            np.exp(-0.5 * np.square(distance[centres] / (radius / 2.0))),
            0.0,
        )
    raise ValueError(f"unknown weighting: {weighting}")


def aggregate(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    present = np.isfinite(values)
    totals = np.where(present, values, 0.0) @ weights.T
    counts = present.astype(float) @ weights.T
    return np.where(counts > 0, totals / np.maximum(counts, 1e-12), np.nan)


def daily_loss(error: np.ndarray, valid_times: pd.DatetimeIndex, metric: str) -> pd.DataFrame:
    if metric == "mae":
        loss = np.nanmean(np.abs(error), axis=1)
    elif metric == "rmse":
        loss = np.nanmean(np.square(error), axis=1)
    else:
        raise ValueError(metric)
    frame = pd.DataFrame({"valid_day": valid_times.normalize(), "loss": loss})
    return frame.groupby("valid_day", as_index=False).loss.mean().sort_values("valid_day").reset_index(drop=True)


def metric_score(daily: pd.DataFrame, metric: str, indices: np.ndarray | None = None) -> np.ndarray | float:
    if indices is None:
        value: np.ndarray | float = float(daily.loss.mean())
    else:
        value = daily.loss.to_numpy()[indices].mean(axis=1)
    return np.sqrt(value) if metric == "rmse" else value


def optimum(losses: dict[float, pd.DataFrame], sigmas: list[float], metric: str) -> float:
    values = np.array([metric_score(losses[sigma], metric) for sigma in sigmas])
    return float(np.asarray(sigmas)[np.flatnonzero(values == values.min())[0]])


def rank_correlation(x: pd.Series, y: pd.Series) -> float:
    value = x.corr(y, method="spearman")
    return 0.0 if not np.isfinite(value) else float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = args.input.resolve() if args.input else resolve(cfg["outputs"]["blurred_panel"])
    output = args.output.resolve() if args.output else resolve(cfg["outputs"]["results_root"])
    output.mkdir(parents=True, exist_ok=True)
    targets = [
        output / "s2_support_curves.csv",
        output / "s2_support_optima.csv",
        output / "s2_support_thinning.csv",
        output / "s2_density_gates.csv",
        output / "s2_summary.json",
        output / "s2_manifest.json",
    ]
    if any(path.exists() for path in targets) and not args.force:
        raise FileExistsError(f"S2 output exists in {output}; use --force")

    family = cfg["reference_support_s2"]
    bootstrap = cfg["uncertainty"]
    sigmas = [float(value) for value in cfg["controlled_smoothing_s1"]["sigma_km"]]
    sigma_columns = [f"sigma_{sigma:g}km" for sigma in sigmas]
    ladders = {
        "exact_preregistered": [float(value) for value in family["exact_preregistered_radii_km"]],
        "prospective_common": [float(value) for value in family["prospective_common_radii_km"]],
    }
    models = family["primary_models"]
    metrics = family["metrics"]
    connection = duckdb.connect()
    networks = [row[0] for row in connection.execute(
        f"SELECT DISTINCT network FROM read_parquet('{source.as_posix()}') ORDER BY network"
    ).fetchall()]
    curves: list[dict] = []
    optima_rows: list[dict] = []
    thinning_rows: list[dict] = []
    density_rows: list[dict] = []

    for network in networks:
        stations = connection.execute(
            f"""
            SELECT station_id, any_value(latitude) AS latitude, any_value(longitude) AS longitude
            FROM read_parquet('{source.as_posix()}') WHERE network = ?
            GROUP BY station_id ORDER BY station_id
            """,
            [network],
        ).fetchdf()
        station_ids = stations.station_id.astype(str).tolist()
        distance = haversine_matrix(stations)
        observation_long = connection.execute(
            f"""
            SELECT station_id, valid_time, observed_t2m_c
            FROM read_parquet('{source.as_posix()}')
            WHERE network = ? AND model = ? ORDER BY valid_time, station_id
            """,
            [network, models[0]],
        ).fetchdf()
        observation = observation_long.pivot(
            index="valid_time", columns="station_id", values="observed_t2m_c"
        ).reindex(columns=station_ids)
        valid_times = pd.DatetimeIndex(observation.index)
        observed_values = observation.to_numpy(float)
        forecasts: dict[str, dict[float, np.ndarray]] = {}
        for model in models:
            quoted = ", ".join(f'"{column}"' for column in sigma_columns)
            forecast_long = connection.execute(
                f"""
                SELECT station_id, valid_time, {quoted}
                FROM read_parquet('{source.as_posix()}')
                WHERE network = ? AND model = ? ORDER BY valid_time, station_id
                """,
                [network, model],
            ).fetchdf()
            forecasts[model] = {}
            for sigma, column in zip(sigmas, sigma_columns, strict=True):
                wide = forecast_long.pivot(index="valid_time", columns="station_id", values=column)
                wide = wide.reindex(index=valid_times, columns=station_ids)
                forecasts[model][sigma] = wide.to_numpy(float)

        for ladder_name, radii in ladders.items():
            smallest_positive = min(radius for radius in radii if radius > 0)
            neighbour_counts = (distance <= smallest_positive).sum(axis=1)
            centres = np.flatnonzero(
                neighbour_counts >= int(family["minimum_stations_per_positive_radius"])
            )
            density_pass = len(centres) >= int(family["minimum_centres_for_inference"])
            density_rows.append(
                {
                    "network": network,
                    "ladder": ladder_name,
                    "smallest_positive_radius_km": smallest_positive,
                    "stations": int(len(stations)),
                    "eligible_centres": int(len(centres)),
                    "minimum_centres": int(family["minimum_centres_for_inference"]),
                    "density_gate_passed": density_pass,
                }
            )
            if not density_pass:
                print(
                    f"S2 density gate failed: {network}/{ladder_name}: {len(centres)} centres",
                    flush=True,
                )
                continue

            targets_by_weighting: dict[tuple[str, float], np.ndarray] = {}
            for weighting in family["weightings"]:
                for radius in radii:
                    weights = weight_matrix(distance, centres, radius, weighting)
                    targets_by_weighting[(weighting, radius)] = aggregate(observed_values, weights)

            thinning_targets: dict[tuple[int, float], np.ndarray] = {}
            for draw in range(int(family["thinning_draws"])):
                for radius in radii:
                    if radius == 0:
                        weights = np.eye(len(stations), dtype=float)[centres]
                    else:
                        weights = np.zeros((len(centres), len(stations)), dtype=float)
                        rng = np.random.default_rng(
                            int(bootstrap["seed"])
                            + draw * 97
                            + int(radius * 10)
                            + sum(map(ord, f"{network}{ladder_name}"))
                        )
                        for position, centre in enumerate(centres):
                            candidates = np.flatnonzero(distance[centre] <= radius)
                            chosen = rng.choice(
                                candidates,
                                size=int(family["thinning_stations"]),
                                replace=False,
                            )
                            weights[position, chosen] = 1.0
                    thinning_targets[(draw, radius)] = aggregate(observed_values, weights)

            for model in models:
                centre_forecasts = {sigma: forecasts[model][sigma][:, centres] for sigma in sigmas}
                for weighting in family["weightings"]:
                    for radius in radii:
                        target = targets_by_weighting[(weighting, radius)]
                        for metric in metrics:
                            losses = {
                                sigma: daily_loss(centre_forecasts[sigma] - target, valid_times, metric)
                                for sigma in sigmas
                            }
                            for sigma in sigmas:
                                curves.append(
                                    {
                                        "network": network,
                                        "ladder": ladder_name,
                                        "model": model,
                                        "weighting": weighting,
                                        "support_km": radius,
                                        "metric": metric,
                                        "sigma_km": sigma,
                                        "score": float(metric_score(losses[sigma], metric)),
                                        "n_centres": int(len(centres)),
                                        "n_days": int(len(losses[sigma])),
                                    }
                                )
                            point_optimum = optimum(losses, sigmas, metric)
                            for width in bootstrap["block_days"]:
                                rng = np.random.default_rng(
                                    int(bootstrap["seed"])
                                    + width
                                    + int(radius * 10)
                                    + sum(map(ord, f"{network}{ladder_name}{model}{weighting}{metric}"))
                                )
                                indices = block_indices(
                                    rng,
                                    len(losses[sigmas[0]]),
                                    int(bootstrap["n_resamples"]),
                                    width,
                                )
                                draws = np.stack(
                                    [metric_score(losses[sigma], metric, indices) for sigma in sigmas],
                                    axis=1,
                                )
                                best = np.asarray(sigmas)[draws.argmin(axis=1)]
                                low, high = interval(best, float(bootstrap["confidence"]))
                                optima_rows.append(
                                    {
                                        "network": network,
                                        "ladder": ladder_name,
                                        "model": model,
                                        "weighting": weighting,
                                        "support_km": radius,
                                        "metric": metric,
                                        "block_days": width,
                                        "optimal_sigma_km_point": point_optimum,
                                        "optimal_sigma_km_bootstrap_mean": float(best.mean()),
                                        "optimal_sigma_ci_low_km": low,
                                        "optimal_sigma_ci_high_km": high,
                                        "p_optimal_sigma_gt_zero": float((best > 0).mean()),
                                    }
                                )

                for draw in range(int(family["thinning_draws"])):
                    for radius in radii:
                        target = thinning_targets[(draw, radius)]
                        for metric in metrics:
                            losses = {
                                sigma: daily_loss(centre_forecasts[sigma] - target, valid_times, metric)
                                for sigma in sigmas
                            }
                            thinning_rows.append(
                                {
                                    "network": network,
                                    "ladder": ladder_name,
                                    "model": model,
                                    "draw": draw,
                                    "support_km": radius,
                                    "metric": metric,
                                    "stations_kept": 1 if radius == 0 else int(family["thinning_stations"]),
                                    "optimal_sigma_km": optimum(losses, sigmas, metric),
                                    "n_centres": int(len(centres)),
                                }
                            )
                print(f"S2 scored {network}/{ladder_name}/{model}; centres={len(centres)}", flush=True)
    connection.close()

    curve = pd.DataFrame(curves)
    optima = pd.DataFrame(optima_rows)
    thinning = pd.DataFrame(thinning_rows)
    density = pd.DataFrame(density_rows)
    if not curve.empty:
        curve = curve.sort_values(
            ["network", "ladder", "model", "weighting", "metric", "support_km", "sigma_km"]
        )
    if not optima.empty:
        optima = optima.sort_values(
            ["network", "ladder", "model", "weighting", "metric", "block_days", "support_km"]
        )
    if not thinning.empty:
        thinning = thinning.sort_values(
            ["network", "ladder", "model", "metric", "draw", "support_km"]
        )
    density = density.sort_values(["network", "ladder"])
    curve.to_csv(targets[0], index=False)
    optima.to_csv(targets[1], index=False)
    thinning.to_csv(targets[2], index=False)
    density.to_csv(targets[3], index=False)

    primary_block = int(bootstrap["primary_block_days"])
    decisions: dict[str, dict] = {}
    for row in density.itertuples(index=False):
        decisions.setdefault(row.ladder, {})[row.network] = {
            "eligible_centres": int(row.eligible_centres),
            "density_gate_passed": bool(row.density_gate_passed),
            "metrics": {},
        }
        if not row.density_gate_passed:
            decisions[row.ladder][row.network]["status"] = "geometry_failure_not_scientific_refutation"
            continue
        for metric in metrics:
            routes: dict[str, dict] = {}
            for weighting in family["weightings"]:
                subset = optima[
                    (optima.network == row.network)
                    & (optima.ladder == row.ladder)
                    & (optima.metric == metric)
                    & (optima.weighting == weighting)
                    & (optima.block_days == primary_block)
                ]
                correlations = {
                    model: rank_correlation(part.support_km, part.optimal_sigma_km_point)
                    for model, part in subset.groupby("model")
                }
                routes[weighting] = {
                    "model_spearman": correlations,
                    "mean_spearman": float(np.mean(list(correlations.values()))),
                    "models_positive": int(sum(value > 0 for value in correlations.values())),
                }
            thin_subset = thinning[
                (thinning.network == row.network)
                & (thinning.ladder == row.ladder)
                & (thinning.metric == metric)
            ]
            thin_means = thin_subset.groupby(["model", "support_km"], as_index=False).optimal_sigma_km.mean()
            correlations = {
                model: rank_correlation(part.support_km, part.optimal_sigma_km)
                for model, part in thin_means.groupby("model")
            }
            routes["thinning"] = {
                "model_spearman": correlations,
                "mean_spearman": float(np.mean(list(correlations.values()))),
                "models_positive": int(sum(value > 0 for value in correlations.values())),
            }
            for route in routes.values():
                route["passed"] = route["mean_spearman"] > 0 and route["models_positive"] >= 2
            metric_pass = all(route["passed"] for route in routes.values())
            decisions[row.ladder][row.network]["metrics"][metric] = {
                "routes": routes,
                "passed": metric_pass,
            }
        decisions[row.ladder][row.network]["passed"] = all(
            item["passed"] for item in decisions[row.ladder][row.network]["metrics"].values()
        )
        decisions[row.ladder][row.network]["status"] = (
            "passed" if decisions[row.ladder][row.network]["passed"] else "scientific_criterion_failed"
        )

    prospective = decisions.get("prospective_common", {})
    strong_success = bool(
        prospective
        and all(item.get("density_gate_passed") and item.get("passed") for item in prospective.values())
    )
    summary = {
        "analysis": "national reference-support ladders S2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decisions": decisions,
        "prospective_common_strong_success": strong_success,
        "exact_ladder_note": (
            "The exact preregistered ladder remains primary where its frozen density gate passes; "
            "a density failure is not counted as scientific refutation."
        ),
        "prospective_ladder_note": family["geometry_reason_for_extension"],
        "allowed_claim_if_s1_also_passes": cfg["interpretation"]["permitted_if_strong_success"],
        "forbidden_claim": cfg["interpretation"]["forbidden_without_documentary_independence"],
    }
    targets[4].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"path": config_path.relative_to(ROOT).as_posix(), "sha256": sha256(config_path)},
        "script_sha256": sha256(Path(__file__).resolve()),
        "input": {"path": source.relative_to(ROOT).as_posix(), "sha256": sha256(source)},
        "uncertainty": bootstrap,
        "outputs": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in targets[:5]},
    }
    targets[5].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
