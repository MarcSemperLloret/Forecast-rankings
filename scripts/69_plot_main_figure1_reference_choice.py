#!/usr/bin/env python3
"""Build main Figure 1: support concept and the held-out AVAMET rank reversal."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Rectangle

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results"
DEFAULT_OUTPUT = ROOT / "results" / "main_figure1_reference_choice_2020"
MODEL_COLORS = {
    "graphcast_hres_init": "#0072B2",
    "ifs_hres": "#777777",
    "pangu_hres_init": "#D55E00",
}
MODEL_LABELS = {
    "graphcast_hres_init": "GraphCast",
    "ifs_hres": "IFS",
    "pangu_hres_init": "Pangu",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def concept_panel(axis) -> None:
    axis.set_title("a  Verification targets", loc="left", fontweight="bold")
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 6)
    axis.axis("off")
    for x0, label in ((0.5, "Point observation"), (5.5, "Gridded analysis")):
        axis.add_patch(Rectangle((x0, 1.2), 3.6, 3.6, facecolor="#F2F2F2", edgecolor="#555555", lw=0.9))
        axis.text(x0 + 1.8, 0.65, label, ha="center", va="center", fontsize=7.5, fontweight="bold")
    axis.add_patch(Circle((2.6, 3.35), 0.72, facecolor="#F0E442", edgecolor="none", alpha=0.8))
    axis.add_patch(Circle((1.45, 2.1), 0.52, facecolor="#56B4E9", edgecolor="none", alpha=0.75))
    axis.plot(2.6, 3.35, marker="*", ms=10, color="#111111")
    axis.annotate("station samples\nits local environment", xy=(2.6, 3.35), xytext=(0.55, 5.35),
                  arrowprops={"arrowstyle": "->", "lw": 0.7}, fontsize=6.8, ha="left")
    axis.add_patch(Rectangle((5.5, 1.2), 3.6, 3.6, facecolor="#A9D5EC", edgecolor="#555555", lw=0.9, alpha=0.95))
    axis.plot(7.6, 3.35, marker="*", ms=8, color="#111111")
    axis.annotate("one area-supported\ngrid-box state", xy=(7.25, 3.0), xytext=(5.6, 5.35),
                  arrowprops={"arrowstyle": "->", "lw": 0.7}, fontsize=6.8, ha="left")
    axis.annotate("same location, different support", xy=(5.0, 0.15), ha="center", fontsize=6.8, color="#444444")


def ranking_panel(axis, scores: pd.DataFrame) -> None:
    axis.set_title("b  Ranking changes with reference", loc="left", fontweight="bold")
    references = ["avamet", "era5_station"]
    x = np.arange(2)
    for model in MODEL_LABELS:
        part = scores[scores.model == model].set_index("reference").loc[references]
        values = part.mae_c_day_weighted.to_numpy(float)
        low = part.mae_ci_low_c.to_numpy(float)
        high = part.mae_ci_high_c.to_numpy(float)
        axis.plot(x, values, color=MODEL_COLORS[model], lw=1.25, zorder=1)
        axis.errorbar(x, values, yerr=np.vstack([values - low, high - values]), fmt="o",
                      color=MODEL_COLORS[model], capsize=2, ms=4.5, lw=0.9,
                      label=MODEL_LABELS[model], zorder=2)
    axis.set_xticks(x, ["Stations", "ERA5"])
    axis.set_ylabel("MAE (°C)")
    axis.legend(frameon=False, fontsize=6.7, loc="upper right")
    axis.text(0, 1.84, "GraphCast wins", ha="center", fontsize=6.7, color=MODEL_COLORS["graphcast_hres_init"])
    axis.text(1, 1.25, "Pangu wins", ha="center", fontsize=6.7, color=MODEL_COLORS["pangu_hres_init"])
    axis.set_xlim(-0.22, 1.22)
    axis.spines[["top", "right"]].set_visible(False)


def contrast_panel(axis, pairwise: pd.DataFrame, draws: pd.DataFrame) -> pd.DataFrame:
    axis.set_title("c  Paired model contrast", loc="left", fontweight="bold")
    pair = pairwise[(pairwise.model_a == "graphcast_hres_init") & (pairwise.model_b == "pangu_hres_init")].copy()
    point = dict(zip(pair.reference, pair.delta_mae_a_minus_b_c))
    intervals = {row.reference: (row.delta_ci_low_c, row.delta_ci_high_c) for row in pair.itertuples(index=False)}
    pivot = draws.pivot_table(index="draw", columns=["reference", "model"], values="mae_c_day_weighted")
    avamet_draw = pivot[("avamet", "graphcast_hres_init")] - pivot[("avamet", "pangu_hres_init")]
    era5_draw = pivot[("era5_station", "graphcast_hres_init")] - pivot[("era5_station", "pangu_hres_init")]
    effect_draw = avamet_draw - era5_draw
    effect_point = point["avamet"] - point["era5_station"]
    effect_interval = tuple(np.quantile(effect_draw.dropna(), [0.025, 0.975]))
    rows = [
        ("Stations", point["avamet"], *intervals["avamet"], "#0072B2"),
        ("ERA5", point["era5_station"], *intervals["era5_station"], "#D55E00"),
        ("Reference effect", effect_point, *effect_interval, "#222222"),
    ]
    for y, (label, value, low, high, color) in enumerate(rows):
        axis.errorbar(value, y, xerr=np.array([[value - low], [high - value]]), fmt="o",
                      color=color, capsize=2.5, ms=4.8, lw=1.0)
    axis.axvline(0, color="#555555", lw=0.8)
    axis.set_yticks(range(3), [row[0] for row in rows])
    axis.invert_yaxis()
    axis.set_xlabel("GraphCast − Pangu MAE (°C)")
    axis.text(-0.19, 2.55, "GraphCast favoured", fontsize=6.5, ha="left", color="#555555")
    axis.text(0.09, 2.55, "Pangu favoured", fontsize=6.5, ha="left", color="#555555")
    axis.spines[["top", "right", "left"]].set_visible(False)
    return pd.DataFrame(rows, columns=["comparison", "estimate_c", "ci_low_c", "ci_high_c", "color"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.results.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    score_path = root / "regional_t2m_2020_scores.csv"
    pair_path = root / "regional_t2m_2020_pairwise.csv"
    draw_path = root / "regional_t2m_2020_bootstrap_draws.csv"
    scores = pd.read_csv(score_path)
    pairwise = pd.read_csv(pair_path)
    draws = pd.read_csv(draw_path)

    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 7.5, "axes.labelsize": 7.5,
        "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), constrained_layout=True,
                                gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})
    concept_panel(axes[0])
    ranking_panel(axes[1], scores)
    contrast_source = contrast_panel(axes[2], pairwise, draws)
    png = output / "main_figure1_reference_choice.png"
    pdf = output / "main_figure1_reference_choice.pdf"
    figure.savefig(png, dpi=600, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)

    score_source = scores[["reference", "model", "n_common_cases", "n_daily_blocks", "mae_c_day_weighted", "mae_ci_low_c", "mae_ci_high_c", "mae_rank"]].copy()
    score_source["source_panel"] = "b"
    contrast_source["source_panel"] = "c"
    score_source.to_csv(output / "panel_b_source_data.csv", index=False)
    contrast_source.to_csv(output / "panel_c_source_data.csv", index=False)
    alt = (
        "Three-panel figure. Panel a contrasts a station sampling a local environment with an area-supported grid-box state at the same nominal location. "
        "Panel b shows mean absolute error and 95% temporal bootstrap intervals for GraphCast, IFS and Pangu against AVAMET stations and ERA5. GraphCast has the lowest station error, while Pangu has the lowest ERA5 error. "
        "Panel c shows the GraphCast-minus-Pangu contrast: negative against stations, positive against ERA5, and a negative paired reference effect whose interval excludes zero.\n"
    )
    (output / "main_figure1_reference_choice_alt_text.md").write_text(alt, encoding="utf-8")
    outputs = [png, pdf, output / "panel_b_source_data.csv", output / "panel_c_source_data.csv"]
    manifest = {
        "analysis_role": "main Figure 1 from frozen AVAMET confirmatory outputs",
        "inputs": {path.relative_to(ROOT).as_posix(): sha256(path) for path in (score_path, pair_path, draw_path)},
        "outputs": {path.name: sha256(path) for path in outputs},
    }
    (output / "outputs_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"png": str(png), "pdf": str(pdf), "passed": True}, indent=2))


if __name__ == "__main__":
    main()
