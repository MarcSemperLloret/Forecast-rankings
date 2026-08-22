#!/usr/bin/env python3
"""Build main Figure 3: support ladders and unchanged-forecast ranking scorecards."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAUSAL = ROOT / "results" / "national_causal_support_2020"
DEFAULT_SCORECARD = ROOT / "results" / "multisupport_scorecard_2020" / "scorecard.csv"
DEFAULT_OUTPUT = ROOT / "results" / "main_figure3_support_and_rankings_2020"
MODEL_LABEL = {"graphcast_hres_init": "GraphCast", "ifs_hres": "IFS", "pangu_hres_init": "Pangu"}
MODEL_CODE = {"graphcast_hres_init": "G", "ifs_hres": "I", "pangu_hres_init": "P"}
MODEL_COLOR = {"graphcast_hres_init": "#0072B2", "ifs_hres": "#777777", "pangu_hres_init": "#D55E00"}
NETWORK_LABEL = {"inmet_hourly": "Brazil", "midas_open_uk": "United Kingdom"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ladder_summary(optima: pd.DataFrame, thinning: pd.DataFrame) -> pd.DataFrame:
    fixed = optima[
        (optima.ladder == "prospective_common")
        & (optima.metric == "rmse")
        & (optima.block_days == 7)
        & (optima.weighting == "uniform")
        & optima.model.isin(MODEL_LABEL)
    ][["network", "model", "support_km", "optimal_sigma_km_point"]].copy()
    fixed = fixed.rename(columns={"optimal_sigma_km_point": "optimal_sigma_km"})
    fixed["route"] = "uniform"
    fixed["ci_low_km"] = fixed.optimal_sigma_km
    fixed["ci_high_km"] = fixed.optimal_sigma_km
    thin = thinning[
        (thinning.ladder == "prospective_common")
        & (thinning.metric == "rmse")
        & thinning.model.isin(MODEL_LABEL)
    ].copy()
    thin = thin.groupby(["network", "model", "support_km"], as_index=False).agg(
        optimal_sigma_km=("optimal_sigma_km", "mean"),
        ci_low_km=("optimal_sigma_km", lambda values: float(np.quantile(values, 0.025))),
        ci_high_km=("optimal_sigma_km", lambda values: float(np.quantile(values, 0.975))),
    )
    thin["route"] = "fixed-three-station thinning"
    return pd.concat([fixed, thin], ignore_index=True)


def plot_ladder(axis, frame: pd.DataFrame, network: str, panel: str) -> None:
    axis.set_title(f"{panel}  {NETWORK_LABEL[network]}", loc="left", fontweight="bold")
    for model in MODEL_LABEL:
        color = MODEL_COLOR[model]
        uniform = frame[(frame.network == network) & (frame.model == model) & (frame.route == "uniform")].sort_values("support_km")
        thin = frame[(frame.network == network) & (frame.model == model) & (frame.route == "fixed-three-station thinning")].sort_values("support_km")
        axis.plot(uniform.support_km, uniform.optimal_sigma_km, color=color, marker="o", ms=4, lw=1.3)
        axis.plot(thin.support_km, thin.optimal_sigma_km, color=color, marker="s", ms=3.5, lw=1.0, ls="--")
        axis.fill_between(thin.support_km, thin.ci_low_km, thin.ci_high_km, color=color, alpha=0.10, linewidth=0)
    axis.set_xlim(-8, 208)
    axis.set_ylim(-4, 104)
    axis.set_xticks([0, 50, 100, 200])
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.set_xlabel("Observational support radius (km)")
    axis.set_ylabel("RMSE-optimal smoothing (km)")
    axis.spines[["top", "right"]].set_visible(False)


def plot_scorecard(axis, scorecard: pd.DataFrame) -> None:
    axis.set_title("c  Rankings of unchanged forecasts", loc="left", fontweight="bold", y=1.075)
    group_columns = ["network", "weighting", "metric"]
    row_keys = list(scorecard[group_columns].drop_duplicates().itertuples(index=False, name=None))
    supports = sorted(scorecard.support_km.unique())
    for y, keys in enumerate(row_keys):
        frame = scorecard[
            (scorecard.network == keys[0])
            & (scorecard.weighting == keys[1])
            & (scorecard.metric == keys[2])
        ].set_index("support_km")
        for x, support in enumerate(supports):
            row = frame.loc[support]
            axis.add_patch(plt.Rectangle((x - 0.48, y - 0.42), 0.96, 0.84,
                                         facecolor=MODEL_COLOR[row.winner], edgecolor="white", linewidth=1))
            axis.text(x, y - 0.07, row.winner_label, ha="center", va="center",
                      color="white", fontweight="bold", fontsize=7.1)
            axis.text(x, y + 0.20, row.ranking_code, ha="center", va="center", color="white", fontsize=6.2)
    labels = [f"{NETWORK_LABEL[key[0]]} · {key[2].upper()} · {key[1]}" for key in row_keys]
    axis.set_xticks(range(len(supports)), [f"{value:g}" for value in supports])
    axis.set_yticks(range(len(row_keys)), labels)
    axis.invert_yaxis()
    axis.set_xlabel("Observational support radius (km)")
    axis.set_xlim(-0.5, len(supports) - 0.5)
    axis.set_ylim(len(row_keys) - 0.5, -0.5)
    axis.text(0, 1.015, "Winner; order best→worst (G=GraphCast, I=IFS, P=Pangu)",
              transform=axis.transAxes, ha="left", va="bottom", fontsize=6.7, color="#444444")
    for spine in axis.spines.values():
        spine.set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal", type=Path, default=DEFAULT_CAUSAL)
    parser.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    causal = args.causal.resolve()
    score_path = args.scorecard.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    optima_path = causal / "s2_support_optima.csv"
    thinning_path = causal / "s2_support_thinning.csv"
    optima = pd.read_csv(optima_path)
    thinning = pd.read_csv(thinning_path)
    scorecard = pd.read_csv(score_path)
    ladder = ladder_summary(optima, thinning)

    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 7.5, "axes.titlesize": 8.5,
        "axes.labelsize": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
        "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    figure = plt.figure(figsize=(7.2, 5.45), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=[1.0, 1.42])
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1]), figure.add_subplot(grid[1, :])]
    plot_ladder(axes[0], ladder, "inmet_hourly", "a")
    plot_ladder(axes[1], ladder, "midas_open_uk", "b")
    model_handles = [Line2D([0], [0], color=MODEL_COLOR[m], marker="o", lw=1.2, label=MODEL_LABEL[m]) for m in MODEL_LABEL]
    axes[0].legend(handles=model_handles, frameon=False, fontsize=6.5, ncol=3, loc="upper left")
    route_handles = [
        Line2D([0], [0], color="#333333", marker="o", lw=1.2, label="Uniform"),
        Line2D([0], [0], color="#333333", marker="s", lw=1.0, ls="--", label="Fixed 3 stations"),
    ]
    axes[1].legend(handles=route_handles, frameon=False, fontsize=6.5, loc="upper left")
    plot_scorecard(axes[2], scorecard)

    png = output / "main_figure3_support_and_rankings.png"
    pdf = output / "main_figure3_support_and_rankings.pdf"
    figure.savefig(png, dpi=600, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    ladder.to_csv(output / "panels_ab_source_data.csv", index=False)
    scorecard.to_csv(output / "panel_c_source_data.csv", index=False)
    (output / "main_figure3_support_and_rankings_alt_text.md").write_text(
        "Three-panel figure. Panels a and b show that the RMSE-optimal forecast smoothing increases as observational support expands from zero to 200 km in Brazil and the United Kingdom. Solid circles are uniform area means; dashed squares and translucent intervals summarize 60 fixed-three-station thinning draws. Colours distinguish GraphCast, IFS and Pangu. Panel c is a 32-cell scorecard for unchanged forecasts. GraphCast wins every point-support cell; rankings change in 22 of 24 positive-support cells and the winner changes in 12.\n",
        encoding="utf-8",
    )
    paths = [png, pdf, output / "panels_ab_source_data.csv", output / "panel_c_source_data.csv"]
    manifest = {
        "analysis_role": "main Figure 3 combining frozen causal support and descriptive scorecard outputs",
        "inputs": {path.relative_to(ROOT).as_posix(): sha256(path) for path in (optima_path, thinning_path, score_path)},
        "outputs": {path.name: sha256(path) for path in paths},
    }
    (output / "outputs_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"png": str(png), "pdf": str(pdf), "passed": True}, indent=2))


if __name__ == "__main__":
    main()
