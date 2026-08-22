#!/usr/bin/env python3
"""Build main Figure 2 from the frozen national controlled-smoothing outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "national_causal_support_2020"
OUTPUT = ROOT / "results" / "main_figure2_controlled_smoothing_2020"
MODELS = ("graphcast_hres_init", "ifs_hres", "pangu_hres_init")
MODEL_LABEL = {
    "graphcast_hres_init": "GraphCast",
    "ifs_hres": "IFS",
    "pangu_hres_init": "Pangu",
}
MODEL_COLOR = {
    "graphcast_hres_init": "#0072B2",
    "ifs_hres": "#777777",
    "pangu_hres_init": "#D55E00",
}
NETWORK_LABEL = {"inmet_hourly": "Brazil", "midas_open_uk": "United Kingdom"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plot_changes(axis: plt.Axes, changes: pd.DataFrame, network: str, panel: str) -> None:
    axis.set_title(f"{panel}  {NETWORK_LABEL[network]}", loc="left", fontweight="bold")
    frame = changes[
        (changes.network == network)
        & (changes.metric == "rmse")
        & (changes.block_days == 7)
        & changes.model.isin(MODELS)
    ]
    for model in MODELS:
        for reference, linestyle, marker, filled in (
            ("era5", "-", "o", True),
            ("station", "--", "s", False),
        ):
            subset = frame[(frame.model == model) & (frame.reference == reference)].sort_values("sigma_km")
            baseline = pd.DataFrame({
                "sigma_km": [0.0], "change": [0.0], "ci_low": [0.0], "ci_high": [0.0]
            })
            subset = pd.concat([baseline, subset[["sigma_km", "change", "ci_low", "ci_high"]]], ignore_index=True)
            color = MODEL_COLOR[model]
            face = color if filled else "white"
            axis.errorbar(
                subset.sigma_km,
                subset.change,
                yerr=np.vstack((subset.change - subset.ci_low, subset.ci_high - subset.change)),
                color=color,
                linestyle=linestyle,
                marker=marker,
                markerfacecolor=face,
                markeredgecolor=color,
                markersize=3.6,
                linewidth=1.1,
                elinewidth=0.65,
                capsize=1.8,
            )
    axis.axhline(0, color="#555555", linestyle=":", linewidth=0.8)
    axis.set_xlim(-4, 104)
    axis.set_xticks([0, 12.5, 25, 50, 100], ["0", "12.5", "25", "50", "100"])
    axis.set_xlabel("Forecast smoothing σ (km)")
    axis.set_ylabel("Change in RMSE (°C)")
    axis.spines[["top", "right"]].set_visible(False)


def plot_optima(axis: plt.Axes, decisions: pd.DataFrame, network: str, panel: str) -> None:
    axis.set_title(f"{panel}  {NETWORK_LABEL[network]}", loc="left", fontweight="bold")
    frame = decisions[decisions.network == network].copy()
    rows: list[tuple[str, str]] = []
    for model in MODELS:
        for metric in ("mae", "rmse"):
            rows.append((model, metric))
    y = np.arange(len(rows))
    for pos, (model, metric) in zip(y, rows):
        row = frame[(frame.model == model) & (frame.metric == metric)].iloc[0]
        station = float(row.station_optimal_sigma_km)
        era5 = float(row.era5_optimal_sigma_km)
        color = MODEL_COLOR[model]
        axis.plot([station, era5], [pos, pos], color=color, linewidth=1.4, zorder=1)
        axis.scatter(station, pos, s=31, facecolor="white", edgecolor=color, linewidth=1.2, zorder=2)
        axis.scatter(era5, pos, s=31, facecolor=color, edgecolor=color, linewidth=1.0, zorder=3)
    labels = [f"{MODEL_LABEL[model]} · {metric.upper()}" for model, metric in rows]
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(-2, 29)
    axis.set_xticks([0, 12.5, 25], ["0", "12.5", "25"])
    axis.set_xlabel("Error-minimizing smoothing σ* (km)")
    axis.grid(axis="x", color="#e6e6e6", linewidth=0.7)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)


def main() -> None:
    changes_path = INPUT / "s1_smoothing_changes.csv"
    summary_path = INPUT / "s1_s3_summary.json"
    changes = pd.read_csv(changes_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    decisions = pd.DataFrame(summary["model_decisions"])
    OUTPUT.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 7.5, "axes.titlesize": 8.5,
        "axes.labelsize": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
        "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.35), constrained_layout=True)
    plot_changes(axes[0, 0], changes, "inmet_hourly", "a")
    plot_changes(axes[0, 1], changes, "midas_open_uk", "b")
    plot_optima(axes[1, 0], decisions, "inmet_hourly", "c")
    plot_optima(axes[1, 1], decisions, "midas_open_uk", "d")

    model_handles = [
        Line2D([0], [0], color=MODEL_COLOR[model], marker="o", lw=1.2, label=MODEL_LABEL[model])
        for model in MODELS
    ]
    reference_handles = [
        Line2D([0], [0], color="#333333", marker="o", lw=1.1, label="ERA5"),
        Line2D([0], [0], color="#333333", marker="s", markerfacecolor="white", lw=1.1, ls="--", label="Stations"),
    ]
    axes[0, 0].legend(handles=model_handles, frameon=False, fontsize=6.3, ncol=3, loc="upper left")
    axes[0, 1].legend(handles=reference_handles, frameon=False, fontsize=6.3, loc="upper left")
    optima_handles = [
        Line2D([0], [0], color="#333333", marker="o", markerfacecolor="#333333", lw=0, label="ERA5 optimum"),
        Line2D([0], [0], color="#333333", marker="o", markerfacecolor="white", lw=0, label="Station optimum"),
    ]
    axes[1, 1].legend(handles=optima_handles, frameon=False, fontsize=6.3, loc="lower right")

    png = OUTPUT / "main_figure2_controlled_smoothing.png"
    pdf = OUTPUT / "main_figure2_controlled_smoothing.pdf"
    figure.savefig(png, dpi=600, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    changes[
        (changes.metric == "rmse") & (changes.block_days == 7) & changes.model.isin(MODELS)
    ].to_csv(OUTPUT / "panels_ab_source_data.csv", index=False)
    decisions.to_csv(OUTPUT / "panels_cd_source_data.csv", index=False)
    (OUTPUT / "main_figure2_controlled_smoothing_alt_text.md").write_text(
        "Four-panel figure. Panels a and b show changes in RMSE after smoothing unchanged forecasts in Brazil and the United Kingdom. ERA5 curves initially decline while station curves are minimized at zero smoothing. Panels c and d connect the station optimum at zero to ERA5 optima of 12.5 or 25 km for every model and both MAE and RMSE.\n",
        encoding="utf-8",
    )
    outputs = [png, pdf, OUTPUT / "panels_ab_source_data.csv", OUTPUT / "panels_cd_source_data.csv"]
    manifest = {
        "analysis_role": "main Figure 2; frozen national controlled-smoothing intervention",
        "inputs": {path.relative_to(ROOT).as_posix(): sha256(path) for path in (changes_path, summary_path)},
        "outputs": {path.name: sha256(path) for path in outputs},
    }
    (OUTPUT / "outputs_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"png": str(png), "pdf": str(pdf), "passed": True}, indent=2))


if __name__ == "__main__":
    main()
