#!/usr/bin/env python3
"""Plot the negative cross-family roughness boundary at native and coarse scales."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NATIVE = ROOT / "results" / "weather5k_model_family_native_analysis_2020"
DEFAULT_COARSE = ROOT / "results" / "weather5k_model_family_analysis_2020"
DEFAULT_OUTPUT = ROOT / "results" / "model_family_scale_boundary_2020"
MODEL_COLORS = {
    "ifs": "#D55E00",
    "graphcast": "#0072B2",
    "pangu": "#009E73",
    "fuxi": "#CC79A7",
    "gencast": "#777777",
}
MODEL_LABELS = {
    "ifs": "IFS",
    "graphcast": "GraphCast",
    "pangu": "Pangu",
    "fuxi": "FuXi",
    "gencast": "GenCast mean",
}
METRIC_COLORS = {"mae": "#0072B2", "rmse": "#D55E00"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scatter_panel(axis, data: pd.DataFrame, metric: str, panel: str) -> None:
    x = data["roughness_50km_k"].to_numpy(float)
    y = data[f"{metric}_advantage_c"].to_numpy(float)
    deterministic = data.model_family != "gencast"
    coefficient = np.polyfit(x[deterministic], y[deterministic], 1)
    grid = np.linspace(x.min() - 0.006, x.max() + 0.006, 100)
    axis.plot(grid, np.polyval(coefficient, grid), color="#333333", lw=1.1, zorder=1)
    offsets = {
        "ifs": (4, -11),
        "graphcast": (4, 5),
        "pangu": (4, 5),
        "fuxi": (4, 5),
        "gencast": (4, -11),
    }
    for row in data.itertuples(index=False):
        family = row.model_family
        x_value = row.roughness_50km_k
        y_value = getattr(row, f"{metric}_advantage_c")
        is_gencast = family == "gencast"
        axis.scatter(
            x_value,
            y_value,
            s=34,
            marker="D" if is_gencast else "o",
            facecolor="white" if is_gencast else MODEL_COLORS[family],
            edgecolor=MODEL_COLORS[family],
            linewidth=1.0,
            zorder=3,
        )
        dx, dy = offsets[family]
        axis.annotate(
            MODEL_LABELS[family],
            (x_value, y_value),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=6.5,
            color=MODEL_COLORS[family],
        )
    rho = data.loc[deterministic, ["roughness_50km_k", f"{metric}_advantage_c"]].corr(
        method="spearman"
    ).iloc[0, 1]
    axis.text(
        0.03,
        0.91,
        f"Deterministic families: Spearman ρ = {rho:.1f}",
        transform=axis.transAxes,
        fontsize=6.8,
    )
    padding = max(0.015, 0.22 * float(y.max() - y.min()))
    axis.set_ylim(float(y.min() - padding), float(y.max() + padding))
    axis.set_xlabel("50-km roughness (K)")
    axis.set_ylabel(f"B: ERA5 − station {metric.upper()} (°C)")
    axis.set_title(f"{panel}  Native {metric.upper()}", loc="left", fontweight="bold")


def forest_panel(axis, slopes: pd.DataFrame, resolution: str, panel: str) -> None:
    frame = slopes.copy().sort_values(["scale_km", "metric"])
    scales = sorted(frame.scale_km.unique())
    positions = {scale: index for index, scale in enumerate(scales)}
    for metric, offset, marker in (("mae", -0.10, "o"), ("rmse", 0.10, "s")):
        part = frame[frame.metric == metric]
        y = np.array([positions[value] + offset for value in part.scale_km])
        x = part.slope_c_per_k.to_numpy(float)
        low = part.ci_low.to_numpy(float)
        high = part.ci_high.to_numpy(float)
        axis.errorbar(
            x,
            y,
            xerr=np.vstack([x - low, high - x]),
            fmt=marker,
            color=METRIC_COLORS[metric],
            ecolor=METRIC_COLORS[metric],
            elinewidth=1.0,
            capsize=2.0,
            markersize=4.2,
            label=metric.upper(),
        )
    axis.axvline(0, color="#333333", lw=0.8)
    axis.set_yticks(range(len(scales)), [f"{int(value):,} km" for value in scales])
    axis.invert_yaxis()
    axis.set_xlabel("Roughness → B slope (°C/K)")
    axis.set_title(f"{panel}  {resolution}", loc="left", fontweight="bold")
    axis.legend(frameon=False, fontsize=6.8, ncol=2, loc="lower right")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, default=DEFAULT_NATIVE)
    parser.add_argument("--coarse", type=Path, default=DEFAULT_COARSE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    native = args.native.resolve()
    coarse = args.coarse.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(native / "model_metrics.csv")
    native_slopes = pd.read_csv(native / "slopes.csv")
    coarse_slopes = pd.read_csv(coarse / "slopes.csv")

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    scatter_panel(axes[0, 0], metrics, "mae", "a")
    scatter_panel(axes[0, 1], metrics, "rmse", "b")
    forest_panel(axes[1, 0], native_slopes, "Native 0.25°", "c")
    forest_panel(axes[1, 1], coarse_slopes, "Common 1.5° grid", "d")
    png = output / "model_family_scale_boundary.png"
    pdf = output / "model_family_scale_boundary.pdf"
    figure.savefig(png, dpi=600, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)

    source = pd.concat(
        [
            native_slopes.assign(resolution="native_0p25deg"),
            coarse_slopes.assign(resolution="common_1p5deg"),
        ],
        ignore_index=True,
    )
    source.to_csv(output / "model_family_scale_boundary_source_data.csv", index=False)
    (output / "model_family_scale_boundary_alt_text.md").write_text(
        "Four-panel figure showing that model-family roughness does not provide "
        "the predicted explanation of reference advantage. Panels a and b plot "
        "native 50-km roughness against ERA5-minus-station MAE and RMSE for IFS, "
        "GraphCast, Pangu and FuXi, with GenCast ensemble mean shown separately "
        "as a hollow diamond. The deterministic association is negative. Panel c "
        "shows negative bootstrap slopes with confidence intervals below zero at "
        "25, 50 and 100 km. Panel d shows the common 1.5-degree result changing "
        "from positive at 300 km to uncertain at 600 km and negative at 1,200 km. "
        "The result separates causal within-forecast support effects from "
        "cross-family associations.\n",
        encoding="utf-8",
    )
    inputs = [
        native / "model_metrics.csv",
        native / "slopes.csv",
        coarse / "slopes.csv",
    ]
    manifest = {
        "analysis_role": "transparent negative model-family boundary figure",
        "inputs": {path.relative_to(ROOT).as_posix(): sha256(path) for path in inputs},
        "outputs": {path.name: sha256(path) for path in [png, pdf, output / "model_family_scale_boundary_source_data.csv"]},
    }
    (output / "outputs_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"png": str(png), "pdf": str(pdf), "passed": True}, indent=2))


if __name__ == "__main__":
    main()
