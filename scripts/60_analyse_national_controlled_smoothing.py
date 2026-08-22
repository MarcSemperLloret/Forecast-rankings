#!/usr/bin/env python3
"""Score national controlled smoothing (S1) and MSE decomposition (S3)."""
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
REFERENCE_COLUMNS = {"era5": "era5_t2m_c", "station": "observed_t2m_c"}


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


def daily_components(frame: pd.DataFrame, prediction: str, reference: str) -> pd.DataFrame:
    error = frame[prediction].to_numpy(float) - frame[reference].to_numpy(float)
    daily = pd.DataFrame(
        {
            "valid_day": pd.to_datetime(frame.valid_time).dt.normalize(),
            "absolute": np.abs(error),
            "squared": np.square(error),
            "signed": error,
        }
    )
    return (
        daily.groupby("valid_day", as_index=False)
        .mean(numeric_only=True)
        .sort_values("valid_day")
        .reset_index(drop=True)
    )


def scores(daily: pd.DataFrame, indices: np.ndarray | None = None) -> dict[str, np.ndarray | float]:
    if indices is None:
        absolute = float(daily.absolute.mean())
        mse = float(daily.squared.mean())
        bias = float(daily.signed.mean())
    else:
        absolute = daily.absolute.to_numpy()[indices].mean(axis=1)
        mse = daily.squared.to_numpy()[indices].mean(axis=1)
        bias = daily.signed.to_numpy()[indices].mean(axis=1)
    centred_mse = np.maximum(mse - np.square(bias), 0.0)
    return {
        "mae": absolute,
        "rmse": np.sqrt(mse),
        "centred_rmse": np.sqrt(centred_mse),
        "mse": mse,
        "bias": bias,
        "bias_squared": np.square(bias),
        "centred_mse": centred_mse,
    }


def best_sigma(curve: pd.DataFrame, network: str, model: str, reference: str, metric: str) -> float:
    subset = curve[
        (curve.network == network)
        & (curve.model == model)
        & (curve.reference == reference)
        & (curve.metric == metric)
    ].sort_values(["score", "sigma_km"])
    if subset.empty:
        raise RuntimeError(f"missing curve for {network}/{model}/{reference}/{metric}")
    return float(subset.iloc[0].sigma_km)


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
        output / "s1_smoothing_curves.csv",
        output / "s1_smoothing_changes.csv",
        output / "s3_mse_decomposition.csv",
        output / "s1_s3_summary.json",
        output / "s1_s3_manifest.json",
    ]
    if any(path.exists() for path in targets) and not args.force:
        raise FileExistsError(f"S1/S3 output exists in {output}; use --force")

    sigmas = [float(value) for value in cfg["controlled_smoothing_s1"]["sigma_km"]]
    bootstrap = cfg["uncertainty"]
    connection = duckdb.connect()
    groups = connection.execute(
        f"SELECT DISTINCT network, model, model_family, cohort FROM read_parquet('{source.as_posix()}') "
        "ORDER BY network, model"
    ).fetchdf()
    curves: list[dict] = []
    changes: list[dict] = []
    decompositions: list[dict] = []
    daily_cache: dict[tuple[str, str, str, float], pd.DataFrame] = {}
    for group in groups.itertuples(index=False):
        frame = connection.execute(
            f"""
            SELECT valid_time, era5_t2m_c, observed_t2m_c,
                   {', '.join(f'"sigma_{sigma:g}km"' for sigma in sigmas)}
            FROM read_parquet('{source.as_posix()}')
            WHERE network = ? AND model = ? ORDER BY valid_time, station_id
            """,
            [group.network, group.model],
        ).fetchdf()
        for reference, reference_column in REFERENCE_COLUMNS.items():
            for sigma in sigmas:
                daily = daily_components(frame, f"sigma_{sigma:g}km", reference_column)
                daily_cache[(group.network, group.model, reference, sigma)] = daily
                point = scores(daily)
                for metric in cfg["controlled_smoothing_s1"]["metrics"]:
                    curves.append(
                        {
                            "network": group.network,
                            "model": group.model,
                            "model_family": group.model_family,
                            "cohort": group.cohort,
                            "reference": reference,
                            "sigma_km": sigma,
                            "metric": metric,
                            "score": float(point[metric]),
                            "n_days": int(len(daily)),
                        }
                    )
                decompositions.append(
                    {
                        "network": group.network,
                        "model": group.model,
                        "reference": reference,
                        "sigma_km": sigma,
                        "mse_c2": float(point["mse"]),
                        "bias_c": float(point["bias"]),
                        "bias_squared_c2": float(point["bias_squared"]),
                        "centred_mse_c2": float(point["centred_mse"]),
                    }
                )
        for width in bootstrap["block_days"]:
            first = daily_cache[(group.network, group.model, "era5", sigmas[0])]
            rng = np.random.default_rng(
                int(bootstrap["seed"]) + width + sum(map(ord, f"{group.network}{group.model}"))
            )
            indices = block_indices(rng, len(first), int(bootstrap["n_resamples"]), width)
            for reference in REFERENCE_COLUMNS:
                baseline = scores(daily_cache[(group.network, group.model, reference, sigmas[0])], indices)
                for sigma in sigmas[1:]:
                    current = scores(daily_cache[(group.network, group.model, reference, sigma)], indices)
                    for metric in cfg["controlled_smoothing_s1"]["metrics"]:
                        delta = np.asarray(current[metric]) - np.asarray(baseline[metric])
                        low, high = interval(delta, float(bootstrap["confidence"]))
                        changes.append(
                            {
                                "network": group.network,
                                "model": group.model,
                                "reference": reference,
                                "sigma_km": sigma,
                                "block_days": width,
                                "metric": metric,
                                "change": float(delta.mean()),
                                "ci_low": low,
                                "ci_high": high,
                                "p_smoothing_improves": float((delta < 0).mean()),
                            }
                        )
        print(f"S1 scored {group.network}/{group.model}", flush=True)
    connection.close()

    curve = pd.DataFrame(curves).sort_values(
        ["network", "model", "reference", "metric", "sigma_km"]
    )
    change = pd.DataFrame(changes).sort_values(
        ["network", "model", "reference", "metric", "block_days", "sigma_km"]
    )
    decomposition = pd.DataFrame(decompositions).sort_values(
        ["network", "model", "reference", "sigma_km"]
    )
    curve.to_csv(targets[0], index=False)
    change.to_csv(targets[1], index=False)
    decomposition.to_csv(targets[2], index=False)

    primary_models = cfg["scope"]["primary_models"]
    inferential_metrics = [
        cfg["controlled_smoothing_s1"]["primary_metric"],
        cfg["controlled_smoothing_s1"]["confirmatory_metric"],
    ]
    model_decisions: list[dict] = []
    network_decisions: dict[str, dict] = {}
    s3_rows: list[dict] = []
    for network in sorted(curve.network.unique()):
        network_decisions[network] = {"metrics": {}}
        for metric in inferential_metrics:
            passed = 0
            for model in primary_models:
                era5_optimum = best_sigma(curve, network, model, "era5", metric)
                station_optimum = best_sigma(curve, network, model, "station", metric)
                base = curve[
                    (curve.network == network)
                    & (curve.model == model)
                    & (curve.metric == metric)
                    & (curve.sigma_km == 0)
                ].set_index("reference").score
                positive = curve[
                    (curve.network == network)
                    & (curve.model == model)
                    & (curve.metric == metric)
                    & (curve.sigma_km > 0)
                ]
                by_reference = {
                    reference: part.sort_values("sigma_km").score.to_numpy() - float(base[reference])
                    for reference, part in positive.groupby("reference")
                }
                same_direction_everywhere = bool(
                    np.all(by_reference["era5"] * by_reference["station"] > 0)
                )
                model_pass = era5_optimum > 0 and station_optimum == 0
                passed += int(model_pass)
                model_decisions.append(
                    {
                        "network": network,
                        "model": model,
                        "metric": metric,
                        "era5_optimal_sigma_km": era5_optimum,
                        "station_optimal_sigma_km": station_optimum,
                        "opposite_optima_pass": model_pass,
                        "same_direction_at_every_positive_sigma": same_direction_everywhere,
                    }
                )
            network_decisions[network]["metrics"][metric] = {
                "primary_models_passing": passed,
                "required": 2,
                "passed": passed >= 2,
            }
        network_decisions[network]["passed"] = all(
            item["passed"] for item in network_decisions[network]["metrics"].values()
        )

        for model in primary_models:
            era5_optimum = best_sigma(curve, network, model, "era5", "rmse")
            base = decomposition[
                (decomposition.network == network)
                & (decomposition.model == model)
                & (decomposition.reference == "era5")
                & (decomposition.sigma_km == 0)
            ].iloc[0]
            optimum = decomposition[
                (decomposition.network == network)
                & (decomposition.model == model)
                & (decomposition.reference == "era5")
                & (decomposition.sigma_km == era5_optimum)
            ].iloc[0]
            bias_change = float(optimum.bias_squared_c2 - base.bias_squared_c2)
            centred_change = float(optimum.centred_mse_c2 - base.centred_mse_c2)
            s3_rows.append(
                {
                    "network": network,
                    "model": model,
                    "era5_optimal_sigma_km": era5_optimum,
                    "bias_squared_change_c2": bias_change,
                    "centred_mse_change_c2": centred_change,
                    "variance_component_dominates_improvement": centred_change < bias_change,
                }
            )

    decision_frame = pd.DataFrame(model_decisions)
    same_direction_violation = bool(
        decision_frame[decision_frame.metric.isin(inferential_metrics)]
        .same_direction_at_every_positive_sigma.any()
    )
    strong_success = bool(
        all(item["passed"] for item in network_decisions.values())
        and not same_direction_violation
    )
    s3_frame = pd.DataFrame(s3_rows)
    summary = {
        "analysis": "national controlled smoothing S1 and MSE decomposition S3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "networks": network_decisions,
        "model_decisions": model_decisions,
        "strong_s1_success": strong_success,
        "same_direction_violation_any_primary_model_metric": same_direction_violation,
        "s3_primary_models_variance_dominant": int(s3_frame.variance_component_dominates_improvement.sum()),
        "s3_primary_model_network_tests": int(len(s3_frame)),
        "s3_details": s3_rows,
        "allowed_claim_if_s2_also_passes": cfg["interpretation"]["permitted_if_strong_success"],
        "forbidden_claim": cfg["interpretation"]["forbidden_without_documentary_independence"],
    }
    targets[3].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"path": config_path.relative_to(ROOT).as_posix(), "sha256": sha256(config_path)},
        "script_sha256": sha256(Path(__file__).resolve()),
        "input": {"path": source.relative_to(ROOT).as_posix(), "sha256": sha256(source)},
        "uncertainty": bootstrap,
        "outputs": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in targets[:4]},
    }
    targets[4].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
