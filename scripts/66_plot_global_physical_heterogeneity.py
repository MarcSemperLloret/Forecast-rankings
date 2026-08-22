#!/usr/bin/env python3
"""Publication figure for the pre-specified global physical mechanism test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "weather5k_physical_heterogeneity_2020"
DEFAULT_COAST = ROOT / "data" / "raw" / "geography" / "ne_10m_coastline_v5.1.2.geojson"
LABELS = {
    "absolute_elevation_mismatch_m": "Elevation mismatch",
    "era5_orography_sd_m": "Subgrid orography",
    "era5_landsea_heterogeneity_100km": "Land–sea heterogeneity",
}
COLORS = {"mae": "#0072B2", "rmse": "#D55E00"}


def coast_segments(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    for feature in payload["features"]:
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates", [])
        if geometry.get("type") == "LineString":
            yield np.asarray(coordinates)
        elif geometry.get("type") == "MultiLineString":
            for line in coordinates:
                yield np.asarray(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--coastline", type=Path, default=DEFAULT_COAST)
    parser.add_argument("--stem", default="global_physical_heterogeneity")
    args = parser.parse_args()
    root = args.input.resolve()
    cells = pd.read_csv(root / "cell_effects.csv")
    associations = pd.read_csv(root / "associations.csv")
    quartiles = pd.read_csv(root / "quartile_contrasts.csv")
    associations = associations[associations.covariate.isin(LABELS)]
    quartiles = quartiles[quartiles.covariate.isin(LABELS)]

    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 7.5,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure = plt.figure(figsize=(7.2, 4.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=[1.35, 1.0])
    map_axis = figure.add_subplot(grid[0, :])
    corr_axis = figure.add_subplot(grid[1, 0])
    contrast_axis = figure.add_subplot(grid[1, 1])

    for segment in coast_segments(args.coastline.resolve()):
        map_axis.plot(segment[:, 0], segment[:, 1], color="#777777", linewidth=0.28, zorder=0)
    limit = float(np.nanquantile(np.abs(cells.mae_advantage_c), 0.98))
    points = map_axis.scatter(
        cells.mean_longitude,
        cells.mean_latitude,
        c=cells.mae_advantage_c,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        s=4.2,
        linewidth=0,
        alpha=0.85,
        rasterized=True,
        zorder=1,
    )
    map_axis.set(xlim=(-180, 180), ylim=(-90, 90), xlabel="Longitude", ylabel="Latitude")
    map_axis.set_xticks(np.arange(-180, 181, 60))
    map_axis.set_yticks(np.arange(-60, 61, 30))
    map_axis.grid(color="#DDDDDD", linewidth=0.35, zorder=-1)
    colorbar = figure.colorbar(points, ax=map_axis, location="right", shrink=0.84, pad=0.015)
    colorbar.set_label("ERA5 − station MAE advantage (°C)")
    map_axis.set_title("a   The gridded-reference advantage is geographically structured", loc="left", fontweight="bold")

    positions = np.arange(len(LABELS))
    offsets = {"mae": -0.11, "rmse": 0.11}
    for metric in ("mae", "rmse"):
        frame = associations[associations.metric == metric].set_index("covariate").loc[list(LABELS)]
        y = positions + offsets[metric]
        corr_axis.errorbar(
            frame.spearman,
            y,
            xerr=[frame.spearman - frame.ci_low, frame.ci_high - frame.spearman],
            fmt="o",
            markersize=4,
            capsize=2,
            linewidth=1,
            color=COLORS[metric],
        )
    corr_axis.axvline(0, color="#555555", linewidth=0.7)
    corr_axis.set_yticks(positions, [LABELS[name] for name in LABELS])
    corr_axis.invert_yaxis()
    corr_axis.set_xlabel("Spearman ρ with reference advantage")
    corr_axis.set_title("b   Pre-specified associations", loc="left", fontweight="bold")
    corr_axis.grid(axis="x", color="#E5E5E5", linewidth=0.4)

    for metric in ("mae", "rmse"):
        frame = quartiles[quartiles.metric == metric].set_index("covariate").loc[list(LABELS)]
        y = positions + offsets[metric]
        contrast_axis.errorbar(
            frame.q4_minus_q1_advantage_c,
            y,
            xerr=[
                frame.q4_minus_q1_advantage_c - frame.ci_low,
                frame.ci_high - frame.q4_minus_q1_advantage_c,
            ],
            fmt="o",
            markersize=4,
            capsize=2,
            linewidth=1,
            color=COLORS[metric],
            label=metric.upper(),
        )
    contrast_axis.axvline(0, color="#555555", linewidth=0.7)
    contrast_axis.set_yticks(positions, [])
    contrast_axis.invert_yaxis()
    contrast_axis.set_xlabel("Q4 − Q1 reference advantage (°C)")
    contrast_axis.set_title("c   Extreme-quartile contrasts", loc="left", fontweight="bold")
    contrast_axis.grid(axis="x", color="#E5E5E5", linewidth=0.4)
    contrast_axis.legend(frameon=False, ncol=2, loc="upper right", handletextpad=0.4)

    for axis in (corr_axis, contrast_axis):
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Physical heterogeneity strengthens the mismatch between point and gridded verification",
        fontsize=10,
        fontweight="bold",
    )
    png, pdf = root / f"{args.stem}.png", root / f"{args.stem}.pdf"
    figure.savefig(png, dpi=600, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps({"png": str(png), "pdf": str(pdf)}, indent=2))


if __name__ == "__main__":
    main()
