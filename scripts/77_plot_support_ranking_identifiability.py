#!/usr/bin/env python3
"""Plot the domain-free support-ranking identifiability simulation.

Panel A: point-estimate RMSE of three synthetic forecasts as the target support
grows, showing the ranking crossing. Panel B: the sharp-minus-broad pairwise
contrast with studentized sup-t simultaneous bands, changing sign with support.
Panel C: the reversal phase diagram over the skill-gap and fine-content plane,
with the analytic reversal boundary overlaid.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "synthetic_support_identifiability"
DEFAULT_CONFIG = ROOT / "config" / "synthetic_support_identifiability.yaml"

MODEL_STYLE = {
    "sharp": {"color": "#0072B2", "label": "Sharp (correct fine detail)"},
    "mid": {"color": "#666666", "label": "Intermediate"},
    "broad": {"color": "#D55E00", "label": "Broad (best large scale)"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = args.input.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    png = output / "support_ranking_identifiability.png"
    pdf = output / "support_ranking_identifiability.pdf"
    alt = output / "support_ranking_identifiability_alt_text.md"
    manifest = output / "figure_manifest.json"
    targets = [png, pdf, alt, manifest]
    if any(path.exists() for path in targets) and not args.force:
        raise FileExistsError(f"figure outputs exist in {output}; use --force")
    if args.force:
        for path in targets:
            if path.exists():
                path.unlink()

    cfg = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    var_fine = float(cfg["field"]["fine_scale_variance"])

    profile = pd.read_csv(source / "support_profile_scores.csv")
    contrasts = pd.read_csv(source / "pairwise_contrasts.csv")
    phase = pd.read_csv(source / "flip_phase_diagram.csv")

    profile = profile[(profile.weighting == "uniform") & (profile.metric == "rmse")]
    contrasts = contrasts[
        (contrasts.weighting == "uniform")
        & (contrasts.metric == "rmse")
        & (contrasts.model_a == "sharp")
        & (contrasts.model_b == "broad")
    ].sort_values("support_fraction")

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    figure.subplots_adjust(left=0.07, right=0.985, bottom=0.19, top=0.83, wspace=0.42)

    # Panel A: ranking crossing.
    axis = axes[0]
    for model, style in MODEL_STYLE.items():
        part = profile[profile.model == model].sort_values("support_fraction")
        axis.plot(
            part.support_fraction * 100,
            part.point_score,
            color=style["color"],
            marker="o",
            markersize=3.5,
            label=style["label"],
        )
    axis.set_xlabel("Target support radius (% of domain)")
    axis.set_ylabel("RMSE (field units)")
    axis.set_title("A · Ranking crosses with support", loc="left", fontweight="bold")
    axis.legend(frameon=False, loc="upper center", handlelength=1.4)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    # Panel B: sharp-minus-broad contrast with simultaneous bands.
    axis = axes[1]
    x = contrasts.support_fraction.to_numpy(float) * 100
    y = contrasts.contrast_a_minus_b.to_numpy(float)
    low = contrasts.simultaneous_ci_low.to_numpy(float)
    high = contrasts.simultaneous_ci_high.to_numpy(float)
    axis.axhline(0, color="#222222", linewidth=0.7, zorder=0)
    axis.errorbar(
        x, y, yerr=np.vstack([y - low, high - y]),
        color="#0072B2", marker="o", markersize=3.5, capsize=2.0, elinewidth=0.9,
    )
    axis.set_xlabel("Target support radius (% of domain)")
    axis.set_ylabel("RMSE(sharp) − RMSE(broad)")
    axis.set_title("B · Contrast changes sign", loc="left", fontweight="bold")
    axis.text(
        0.97, 0.06, "sharp better ↓\nbroad better ↑", transform=axis.transAxes,
        ha="right", va="bottom", fontsize=6.2, color="#444444",
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    # Panel C: reversal phase diagram with analytic boundary.
    axis = axes[2]
    grid = phase.pivot(index="skill_gap_deficit", columns="fine_content_gain",
                       values="flip_support_fraction")
    gains = grid.columns.to_numpy(float)
    deficits = grid.index.to_numpy(float)
    mesh = axis.pcolormesh(
        gains, deficits, grid.to_numpy(float) * 100,
        cmap="viridis", shading="nearest",
    )
    no_flip = phase[~phase.reversal_observed]
    axis.scatter(
        no_flip.fine_content_gain, no_flip.skill_gap_deficit,
        marker="x", s=16, color="#B00020", linewidths=0.9, label="no reversal",
    )
    boundary_gain = np.linspace(gains.min(), min(gains.max(), 1.0), 100)
    axis.plot(
        boundary_gain, np.sqrt(boundary_gain * (2.0 - boundary_gain) * var_fine),
        color="white", linewidth=1.4, linestyle="--", label="analytic boundary",
    )
    axis.set_xlabel("Fine-scale content of sharp model, g")
    axis.set_ylabel("Large-scale skill gap, d")
    axis.set_title("C · Reversal phase diagram", loc="left", fontweight="bold")
    axis.set_ylim(deficits.min(), deficits.max())
    axis.legend(frameon=False, loc="upper left", fontsize=6.0)
    colorbar = figure.colorbar(mesh, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Reversal support (% of domain)", fontsize=6.5)
    colorbar.ax.tick_params(labelsize=6)

    figure.savefig(png, dpi=600, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)

    alt.write_text(
        "Three panels from a domain-free simulation with a known ground truth. Panel A shows "
        "the RMSE of three synthetic forecasts as the verifying target is averaged over larger "
        "support: the sharp forecast with correct fine detail wins at a point target, the broad "
        "forecast with the best large-scale skill wins at large support, and the two curves "
        "cross. Panel B shows the sharp-minus-broad RMSE contrast with studentized sup-t "
        "simultaneous 95% bands; the contrast is negative at a point target and positive at "
        "area support, changing sign with bands that exclude zero at both ends. Panel C is a "
        "phase diagram over the large-scale skill gap and the fine-scale content of the sharp "
        "model, coloured by the support radius at which the ranking reverses; a white dashed "
        "analytic boundary d = sqrt(g(2-g)*var_fine) separates reversal from no-reversal cells, "
        "and simulated no-reversal cells (red crosses) fall exactly on the predicted side.\n",
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "script_sha256": sha256(Path(__file__).resolve()),
                "inputs": {
                    name: sha256(source / name)
                    for name in [
                        "support_profile_scores.csv",
                        "pairwise_contrasts.csv",
                        "flip_phase_diagram.csv",
                    ]
                },
                "outputs": {
                    path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                    for path in [png, pdf, alt]
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"png": png.as_posix(), "pdf": pdf.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
