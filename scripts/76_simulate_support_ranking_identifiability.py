#!/usr/bin/env python3
"""Domain-free simulation of support-conditional forecast ranking.

This control was specified after the empirical weather result to test whether
the target-support mechanism is a general property of comparative forecast
evaluation rather than a meteorological accident. A latent field with a known
large-scale component and a known fine-scale component is generated over an
abstract periodic coordinate. Competing forecasts are constructed with a known
trade-off: some spend skill on large-scale accuracy, others inject fine-scale
structure. Only the spatial support of the verifying target is varied.

Two experiments are produced:

1. A three-model scorecard with the same paired circular moving-block bootstrap
   and studentized sup-t simultaneous bands used for the weather analysis,
   showing that expanding target support reverses pairwise orderings and the
   winner.

2. A phase diagram over the (large-scale skill gap, fine-scale roughness gap)
   plane recording the support radius at which the rougher model overtakes the
   smoother one. The analytic reversal boundary q = 1 + d^2 / var_fine is
   compared against the simulated reversals.

Nothing in this script is meteorological. It quantifies when support alone
changes a ranking, independently of any physical domain.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "synthetic_support_identifiability.yaml"
DEFAULT_OUTPUT = ROOT / "results" / "synthetic_support_identifiability"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail])
    return float(low), float(high)


def block_indices(rng: np.random.Generator, days: int, draws: int, width: int) -> np.ndarray:
    starts = rng.integers(0, days, size=(draws, int(np.ceil(days / width))))
    return ((starts[:, :, None] + np.arange(width)) % days).reshape(draws, -1)[:, :days]


def sup_t_band(
    point: np.ndarray, draws: np.ndarray, confidence: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Studentized simultaneous band over the last dimension."""
    standard_error = np.std(draws, axis=0, ddof=1)
    safe = np.where(standard_error > 0, standard_error, np.nan)
    standardized = np.abs((draws - point) / safe)
    max_statistic = np.nanmax(standardized, axis=1)
    critical = float(np.quantile(max_statistic, confidence))
    return point - critical * standard_error, point + critical * standard_error, critical


def scores(loss: np.ndarray, metric: str, indices: np.ndarray | None = None) -> np.ndarray:
    """Aggregate daily loss; output is support x model or draw x support x model."""
    if indices is None:
        value = np.nanmean(loss, axis=0)
    else:
        value = np.nanmean(loss[indices], axis=1)
    return np.sqrt(value) if metric == "rmse" else value


def rank_codes(values: np.ndarray, models: list[str]) -> tuple[str, ...]:
    order = np.argsort(values, axis=-1, kind="stable")
    if order.ndim == 1:
        return ("|".join(models[index] for index in order),)
    return tuple("|".join(models[index] for index in row) for row in order)


def spectral_component(
    rng: np.random.Generator, power: np.ndarray, target_variance: float, realisations: int, grid: int
) -> np.ndarray:
    """Generate real periodic fields with a prescribed power spectrum and variance."""
    n_freq = power.size
    real = rng.standard_normal((realisations, n_freq))
    imag = rng.standard_normal((realisations, n_freq))
    amplitude = np.sqrt(power / 2.0)
    coeffs = (real + 1j * imag) * amplitude
    coeffs[:, 0] = coeffs[:, 0].real * np.sqrt(2.0)  # DC term is real
    if grid % 2 == 0:
        coeffs[:, -1] = coeffs[:, -1].real * np.sqrt(2.0)  # Nyquist term is real
    field = np.fft.irfft(coeffs, n=grid, axis=1)
    empirical = field.std()
    if empirical > 0:
        field *= np.sqrt(target_variance) / empirical
    return field


def circular_kernel(grid: int, half_width: int, weighting: str) -> np.ndarray:
    """Unit-sum circular smoothing kernel; delta at half_width 0."""
    kernel = np.zeros(grid)
    if half_width <= 0:
        kernel[0] = 1.0
        return kernel
    offsets = np.arange(-half_width, half_width + 1)
    if weighting == "uniform":
        weights = np.ones_like(offsets, dtype=float)
    elif weighting == "gaussian":
        sigma = max(half_width / 2.0, 1e-9)
        weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    else:
        raise ValueError(weighting)
    kernel[offsets % grid] = weights
    return kernel / kernel.sum()


def area_average(field: np.ndarray, kernel_fft: np.ndarray) -> np.ndarray:
    return np.fft.irfft(np.fft.rfft(field, axis=1) * kernel_fft[None, :], n=field.shape[1], axis=1)


def daily_losses(forecast_pt: np.ndarray, target_pt: np.ndarray, metric: str) -> np.ndarray:
    """Per-realisation loss over stations for one model and one support."""
    error = forecast_pt - target_pt
    if metric == "mae":
        return np.mean(np.abs(error), axis=1)
    if metric == "rmse":
        return np.mean(np.square(error), axis=1)
    raise ValueError(metric)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    targets = [
        output / "support_profile_scores.csv",
        output / "pairwise_contrasts.csv",
        output / "support_shifts.csv",
        output / "rank_stability.csv",
        output / "forecast_smoothing_optima.csv",
        output / "flip_phase_diagram.csv",
        output / "summary.json",
        output / "outputs_manifest.json",
    ]
    if any(path.exists() for path in targets) and not args.force:
        raise FileExistsError(f"outputs exist in {output}; use --force")
    if args.force:
        for path in targets:
            if path.exists():
                if not path.is_file():
                    raise RuntimeError(f"refusing to replace non-file {path}")
                path.unlink()

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    field_cfg = cfg["field"]
    support_cfg = cfg["support"]
    unc = cfg["uncertainty"]

    grid = int(field_cfg["grid_points"])
    realisations = int(field_cfg["n_realisations"])
    n_stations = int(field_cfg["n_stations"])
    var_large = float(field_cfg["large_scale_variance"])
    var_fine = float(field_cfg["fine_scale_variance"])
    noise_std = float(field_cfg["measurement_noise_std"])

    wavenumbers = np.fft.rfftfreq(grid, d=1.0 / grid)
    large_power = np.exp(-0.5 * (wavenumbers / float(field_cfg["large_scale_wavenumber"])) ** 2)
    fine_power = np.exp(
        -0.5
        * ((wavenumbers - float(field_cfg["fine_scale_wavenumber"]))
           / float(field_cfg["fine_scale_bandwidth"])) ** 2
    )

    rng = np.random.default_rng(int(field_cfg["seed"]))
    station_index = np.sort(rng.choice(grid, size=n_stations, replace=False))

    large = spectral_component(rng, large_power, var_large, realisations, grid)
    fine = spectral_component(rng, fine_power, var_fine, realisations, grid)
    true_field = large + fine

    radii = [float(value) for value in support_cfg["radii_fraction"]]
    weightings = list(support_cfg["weightings"])
    metrics = list(support_cfg["metrics"])
    half_widths = {radius: int(round(radius * grid)) for radius in radii}

    # Precompute area-averaged targets at the stations for every support/weighting.
    targets_pt: dict[tuple[str, float], np.ndarray] = {}
    for weighting in weightings:
        for radius in radii:
            kernel_fft = np.fft.rfft(circular_kernel(grid, half_widths[radius], weighting))
            averaged = area_average(true_field, kernel_fft)
            noise = rng.standard_normal((realisations, n_stations)) * noise_std
            targets_pt[(weighting, radius)] = averaged[:, station_index] + noise

    large_pt = large[:, station_index]
    fine_pt = fine[:, station_index]

    def build_forecast(deficit: float, fine_gain: float) -> np.ndarray:
        """Raw forecast sampled at stations; correct fine detail scaled by fine_gain.

        The forecast is fixed across target supports: only the target changes.
        Correct fine detail is rewarded against a point target but becomes a pure
        penalty against a smoothed area target, from which fine scales are absent.
        """
        return (1.0 - deficit) * large_pt + fine_gain * fine_pt

    # ---- Experiment 1: three-model scorecard with bootstrap uncertainty. ----
    model_cfg = cfg["scorecard_models"]
    models = ["sharp", "mid", "broad"]
    model_pairs = list(itertools.combinations(range(len(models)), 2))
    forecasts_pt = {
        name: build_forecast(
            float(model_cfg[name]["large_scale_deficit"]),
            float(model_cfg[name]["fine_gain"]),
        )
        for name in models
    }

    block = int(unc["primary_block"])
    n_resamples = int(unc["n_resamples"])
    confidence = float(unc["confidence"])

    contrast_rows: list[dict] = []
    shift_rows: list[dict] = []
    winner_rows: list[dict] = []
    stability_rows: list[dict] = []
    band_critical_values: list[dict] = []

    for weighting in weightings:
        for metric in metrics:
            daily = np.stack(
                [
                    np.stack(
                        [daily_losses(forecasts_pt[name], targets_pt[(weighting, radius)], metric)
                         for name in models],
                        axis=1,
                    )
                    for radius in radii
                ],
                axis=1,
            )  # shape (realisations, n_support, n_model)

            seed = int(unc["seed"]) + sum(map(ord, f"{weighting}{metric}"))
            indices = block_indices(np.random.default_rng(seed), realisations, n_resamples, block)
            point_scores = scores(daily, metric)
            draw_scores = scores(daily, metric, indices)
            point_winners = np.argmin(point_scores, axis=1)
            draw_winners = np.argmin(draw_scores, axis=2)
            point_ranks = rank_codes(point_scores, models)

            contrasts = np.stack(
                [point_scores[:, a] - point_scores[:, b] for a, b in model_pairs], axis=1
            )
            contrast_draws = np.stack(
                [draw_scores[:, :, a] - draw_scores[:, :, b] for a, b in model_pairs], axis=2
            )
            simultaneous_low, simultaneous_high, contrast_critical = sup_t_band(
                contrasts.reshape(-1), contrast_draws.reshape(n_resamples, -1), confidence
            )
            band_critical_values.append(
                {
                    "weighting": weighting,
                    "metric": metric,
                    "family": "all_support_pairwise_contrasts",
                    "dimensions": int(contrasts.size),
                    "sup_t_critical": contrast_critical,
                }
            )

            shift = contrasts[1:] - contrasts[0]
            shift_draws = contrast_draws[:, 1:, :] - contrast_draws[:, [0], :]
            shift_low, shift_high, shift_critical = sup_t_band(
                shift.reshape(-1), shift_draws.reshape(n_resamples, -1), confidence
            )
            band_critical_values.append(
                {
                    "weighting": weighting,
                    "metric": metric,
                    "family": "positive_support_pairwise_shifts_from_point",
                    "dimensions": int(shift.size),
                    "sup_t_critical": shift_critical,
                }
            )

            for support_index, radius in enumerate(radii):
                for model_index, name in enumerate(models):
                    winner_rows.append(
                        {
                            "weighting": weighting,
                            "metric": metric,
                            "support_fraction": radius,
                            "model": name,
                            "point_score": float(point_scores[support_index, model_index]),
                            "point_winner": bool(point_winners[support_index] == model_index),
                            "bootstrap_winner_probability": float(
                                np.mean(draw_winners[:, support_index] == model_index)
                            ),
                            "n_stations": n_stations,
                            "n_realisations": realisations,
                            "block": block,
                        }
                    )

                changed_pair_point = [
                    np.sign(contrasts[support_index, pair]) != np.sign(contrasts[0, pair])
                    for pair, _ in enumerate(model_pairs)
                ]
                changed_pair_draws = np.stack(
                    [
                        np.sign(contrast_draws[:, support_index, pair])
                        != np.sign(contrast_draws[:, 0, pair])
                        for pair, _ in enumerate(model_pairs)
                    ],
                    axis=1,
                )
                stability_rows.append(
                    {
                        "weighting": weighting,
                        "metric": metric,
                        "support_fraction": radius,
                        "point_winner": models[point_winners[support_index]],
                        "point_ranking": point_ranks[support_index],
                        "point_winner_changed_vs_point": bool(
                            point_winners[support_index] != point_winners[0]
                        ),
                        "point_any_pair_order_changed_vs_point": bool(any(changed_pair_point)),
                        "point_pairwise_inversions_vs_point": int(sum(changed_pair_point)),
                        "bootstrap_probability_winner_changed_vs_point": float(
                            np.mean(draw_winners[:, support_index] != draw_winners[:, 0])
                        ),
                        "bootstrap_probability_any_pair_order_changed_vs_point": float(
                            np.mean(changed_pair_draws.any(axis=1))
                        ),
                        "n_stations": n_stations,
                        "n_realisations": realisations,
                        "block": block,
                    }
                )

                for pair_index, (a, b) in enumerate(model_pairs):
                    flat_index = support_index * len(model_pairs) + pair_index
                    pointwise_low, pointwise_high = interval(
                        contrast_draws[:, support_index, pair_index], confidence
                    )
                    contrast_rows.append(
                        {
                            "weighting": weighting,
                            "metric": metric,
                            "support_fraction": radius,
                            "model_a": models[a],
                            "model_b": models[b],
                            "contrast_a_minus_b": float(contrasts[support_index, pair_index]),
                            "pointwise_ci_low": pointwise_low,
                            "pointwise_ci_high": pointwise_high,
                            "simultaneous_ci_low": float(simultaneous_low[flat_index]),
                            "simultaneous_ci_high": float(simultaneous_high[flat_index]),
                            "bootstrap_probability_a_better": float(
                                np.mean(contrast_draws[:, support_index, pair_index] < 0)
                            ),
                        }
                    )

            for positive_index, radius in enumerate(radii[1:]):
                for pair_index, (a, b) in enumerate(model_pairs):
                    flat_index = positive_index * len(model_pairs) + pair_index
                    draws_for_shift = shift_draws[:, positive_index, pair_index]
                    pointwise_low, pointwise_high = interval(draws_for_shift, confidence)
                    shift_rows.append(
                        {
                            "weighting": weighting,
                            "metric": metric,
                            "support_fraction": radius,
                            "model_a": models[a],
                            "model_b": models[b],
                            "support_shift_in_contrast": float(shift[positive_index, pair_index]),
                            "pointwise_ci_low": pointwise_low,
                            "pointwise_ci_high": pointwise_high,
                            "simultaneous_ci_low": float(shift_low[flat_index]),
                            "simultaneous_ci_high": float(shift_high[flat_index]),
                            "point_order_reversed_vs_point": bool(
                                np.sign(contrasts[positive_index + 1, pair_index])
                                != np.sign(contrasts[0, pair_index])
                            ),
                        }
                    )
            print(f"scored synthetic scorecard: {weighting}/{metric}", flush=True)

    # ---- Experiment 1b: forecast-side smoothing optima follow target support. ----
    smoothing_widths = radii
    smoothing_rows: list[dict] = []
    for weighting in weightings:
        forecast_smoothed = {}
        for name in models:
            # Smooth the raw forecast field (rebuilt at full grid) over each width.
            deficit = float(model_cfg[name]["large_scale_deficit"])
            fine_gain = float(model_cfg[name]["fine_gain"])
            forecast_field = (1.0 - deficit) * large + fine_gain * fine
            per_width = {}
            for width in smoothing_widths:
                kernel_fft = np.fft.rfft(circular_kernel(grid, half_widths[width], weighting))
                per_width[width] = area_average(forecast_field, kernel_fft)[:, station_index]
            forecast_smoothed[name] = per_width
        for name in models:
            for radius in radii:
                target = targets_pt[(weighting, radius)]
                width_scores = {
                    width: float(np.sqrt(np.mean(daily_losses(
                        forecast_smoothed[name][width], target, "rmse"))))
                    for width in smoothing_widths
                }
                optimal_width = min(width_scores, key=width_scores.get)
                smoothing_rows.append(
                    {
                        "weighting": weighting,
                        "model": name,
                        "target_support_fraction": radius,
                        "optimal_forecast_smoothing_fraction": optimal_width,
                    }
                )

    # ---- Experiment 2: phase diagram of support-induced reversal. ----
    phase = cfg["phase_diagram"]
    metric_pd = phase["metric"]
    weighting_pd = phase["weighting"]
    deficits = [float(v) for v in phase["skill_gap_deficit"]]
    gains = [float(v) for v in phase["fine_content_gain"]]

    def rmse_scalar(forecast_pt: np.ndarray, target: np.ndarray) -> float:
        return float(np.sqrt(np.mean(daily_losses(forecast_pt, target, metric_pd))))

    # Fixed broad reference: best large scale, no fine detail.
    broad_pt = large_pt
    rmse_broad = {r: rmse_scalar(broad_pt, targets_pt[(weighting_pd, r)]) for r in radii}

    phase_rows: list[dict] = []
    positive_radii = [r for r in radii if r > 0]
    for deficit in deficits:
        for gain in gains:
            sharp_pt = (1.0 - deficit) * large_pt + gain * fine_pt
            rmse_sharp = {r: rmse_scalar(sharp_pt, targets_pt[(weighting_pd, r)]) for r in radii}
            sharp_wins_point = rmse_sharp[0.0] < rmse_broad[0.0]
            flip_support = np.nan
            if sharp_wins_point:
                for r in positive_radii:
                    if rmse_broad[r] < rmse_sharp[r]:
                        flip_support = r
                        break
            # Sharp wins point and broad wins at large support iff the large-scale
            # skill gap is smaller than the fine detail that averaging discards.
            analytic_flip_expected = bool(deficit ** 2 < gain * (2.0 - gain) * var_fine)
            phase_rows.append(
                {
                    "skill_gap_deficit": deficit,
                    "fine_content_gain": gain,
                    "sharp_wins_at_point": bool(sharp_wins_point),
                    "flip_support_fraction": float(flip_support) if flip_support == flip_support else None,
                    "reversal_observed": bool(sharp_wins_point and flip_support == flip_support),
                    "analytic_flip_expected": analytic_flip_expected,
                    "rmse_sharp_point": rmse_sharp[0.0],
                    "rmse_broad_point": rmse_broad[0.0],
                    "rmse_sharp_max_support": rmse_sharp[max(radii)],
                    "rmse_broad_max_support": rmse_broad[max(radii)],
                }
            )

    contrasts_frame = pd.DataFrame(contrast_rows)
    shifts_frame = pd.DataFrame(shift_rows)
    winners_frame = pd.DataFrame(winner_rows)
    stability_frame = pd.DataFrame(stability_rows)
    smoothing_frame = pd.DataFrame(smoothing_rows)
    phase_frame = pd.DataFrame(phase_rows)

    positive_stability = stability_frame[stability_frame.support_fraction > 0]
    agreement = float(
        np.mean(phase_frame.reversal_observed == phase_frame.analytic_flip_expected)
    )
    # Forecast smoothing should reward larger widths as target support grows.
    smoothing_monotone = float(
        smoothing_frame.groupby(["weighting", "model"], group_keys=False)
        .apply(lambda g: g.sort_values("target_support_fraction")
               .optimal_forecast_smoothing_fraction.is_monotonic_increasing)
        .mean()
    )

    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": (
            "Domain-free synthetic control characterising the generality of the "
            "target-support mechanism; not a meteorological confirmation."
        ),
        "estimand": (
            "Whether varying only the spatial support of the verifying target reverses a "
            "comparative forecast ranking, in a setting with a fully known ground truth."
        ),
        "design": {
            "grid_points": grid,
            "n_realisations": realisations,
            "n_stations": n_stations,
            "support_fraction": radii,
            "weightings": weightings,
            "metrics": metrics,
            "bootstrap": "paired circular moving-block bootstrap on realisations",
            "block": block,
            "n_resamples": n_resamples,
            "confidence": confidence,
            "simultaneous_band": "studentized sup-t within each weighting-metric family",
        },
        "scorecard": {
            "positive_support_scorecards": int(len(positive_stability)),
            "winner_changes": int(positive_stability.point_winner_changed_vs_point.sum()),
            "rank_changes": int(positive_stability.point_any_pair_order_changed_vs_point.sum()),
            "pairwise_support_shifts": int(len(shifts_frame)),
            "shifts_with_simultaneous_band_excluding_zero": int(
                ((shifts_frame.simultaneous_ci_low > 0) | (shifts_frame.simultaneous_ci_high < 0)).sum()
            ),
            "point_order_reversals": int(shifts_frame.point_order_reversed_vs_point.sum()),
        },
        "forecast_smoothing": {
            "fraction_of_model_curves_monotone_in_support": smoothing_monotone,
        },
        "phase_diagram": {
            "cells": int(len(phase_frame)),
            "reversals_observed": int(phase_frame.reversal_observed.sum()),
            "analytic_vs_simulation_agreement": agreement,
            "analytic_boundary": "reversal expected iff deficit^2 < gain * (2 - gain) * var_fine",
            "var_fine": var_fine,
        },
        "band_critical_values": band_critical_values,
        "interpretation_limit": cfg["interpretation"]["limit"],
    }

    winners_frame.to_csv(targets[0], index=False)
    contrasts_frame.to_csv(targets[1], index=False)
    shifts_frame.to_csv(targets[2], index=False)
    stability_frame.to_csv(targets[3], index=False)
    smoothing_frame.to_csv(targets[4], index=False)
    phase_frame.to_csv(targets[5], index=False)
    targets[6].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    targets[7].write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "config": {
                    "path": config_path.relative_to(ROOT).as_posix(),
                    "sha256": sha256(config_path),
                },
                "script_sha256": sha256(Path(__file__).resolve()),
                "outputs": {
                    path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                    for path in targets[:7]
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary["scorecard"] | summary["phase_diagram"], indent=2))


if __name__ == "__main__":
    main()
