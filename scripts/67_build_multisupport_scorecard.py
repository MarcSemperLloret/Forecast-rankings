#!/usr/bin/env python3
"""Build the reusable NG-5 scorecard from the national support ladders."""
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
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CURVES = ROOT / "results" / "national_causal_support_2020" / "s2_support_curves.csv"
DEFAULT_OUTPUT = ROOT / "results" / "multisupport_scorecard_2020"
MODEL_LABEL = {
    "graphcast_hres_init": "GraphCast",
    "ifs_hres": "IFS",
    "pangu_hres_init": "Pangu",
}
MODEL_CODE = {"graphcast_hres_init": "G", "ifs_hres": "I", "pangu_hres_init": "P"}
MODEL_COLOR = {"graphcast_hres_init": "#0072B2", "ifs_hres": "#999999", "pangu_hres_init": "#D55E00"}


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def inversions(reference_order: list[str], new_order: list[str]) -> int:
    position = {name: index for index, name in enumerate(reference_order)}
    values = [position[name] for name in new_order]
    return sum(values[i] > values[j] for i in range(len(values)) for j in range(i + 1, len(values)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", type=Path, default=DEFAULT_CURVES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    curves_path, output = args.curves.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    known = [
        output / "scorecard.csv", output / "summary.json",
        output / "multisupport_scorecard.png", output / "multisupport_scorecard.pdf",
        output / "outputs_manifest.json",
    ]
    if any(path.exists() for path in known) and not args.force:
        raise FileExistsError(f"outputs exist in {output}; use --force")
    if args.force:
        for path in known:
            if path.exists():
                if not path.is_file():
                    raise RuntimeError(f"refusing to replace non-file {path}")
                path.unlink()

    curves = pd.read_csv(curves_path)
    raw = curves[
        (curves.ladder == "prospective_common")
        & (curves.sigma_km == 0)
        & curves.model.isin(MODEL_LABEL)
    ].copy()
    group_columns = ["network", "weighting", "metric"]
    rows = []
    for keys, group in raw.groupby(group_columns):
        baseline = group[group.support_km == 0].sort_values("score")
        baseline_order = baseline.model.tolist()
        baseline_scores = baseline.set_index("model").score
        for support, frame in group.groupby("support_km"):
            ordered = frame.sort_values("score")
            order = ordered.model.tolist()
            aligned = ordered.set_index("model").loc[baseline_order]
            rho = float(spearmanr(baseline_scores.loc[baseline_order], aligned.score).statistic)
            rows.append({
                "network": keys[0],
                "weighting": keys[1],
                "metric": keys[2],
                "support_km": float(support),
                "ranking_best_to_worst": "|".join(order),
                "ranking_code": "–".join(MODEL_CODE[name] for name in order),
                "winner": order[0],
                "winner_label": MODEL_LABEL[order[0]],
                "winner_score_c": float(ordered.iloc[0].score),
                "runner_up_score_c": float(ordered.iloc[1].score),
                "winner_margin_c": float(ordered.iloc[1].score - ordered.iloc[0].score),
                "spearman_vs_point_ranking": rho,
                "pairwise_inversions_vs_point": inversions(baseline_order, order),
                "possible_pairs": 3,
                "winner_changed_vs_point": order[0] != baseline_order[0],
                "n_centres": int(ordered.iloc[0].n_centres),
                "n_days": int(ordered.iloc[0].n_days),
            })
    scorecard = pd.DataFrame(rows).sort_values(group_columns + ["support_km"])
    scorecard.to_csv(output / "scorecard.csv", index=False)

    at_200 = scorecard[scorecard.support_km == 200]
    positive = scorecard[scorecard.support_km > 0]
    network_gate = (
        positive.groupby("network").winner_changed_vs_point.any().to_dict()
    )
    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "descriptive reusable multi-support benchmark consequence (NG-5)",
        "definition": (
            "Rank unchanged sigma=0 forecasts against observational references whose spatial "
            "support is 0, 50, 100 or 200 km; no model is retuned between supports."
        ),
        "support": {
            "networks": int(scorecard.network.nunique()),
            "metrics": sorted(scorecard.metric.unique()),
            "weightings": sorted(scorecard.weighting.unique()),
            "support_km": sorted(scorecard.support_km.unique().tolist()),
            "models": list(MODEL_LABEL),
        },
        "gates": {
            "at_least_one_winner_change_in_each_network": bool(all(network_gate.values())),
            "pangu_wins_all_200km_scorecards": bool((at_200.winner == "pangu_hres_init").all()),
            "point_winner_differs_from_200km_winner_all_scorecards": bool(
                at_200.winner_changed_vs_point.all()
            ),
            "all_positive_support_rankings_differ_from_point": bool(
                (positive.pairwise_inversions_vs_point > 0).all()
            ),
        },
        "counts": {
            "scorecards": int(len(scorecard)),
            "positive_support_scorecards": int(len(positive)),
            "winner_changes_positive_support": int(positive.winner_changed_vs_point.sum()),
            "rank_changes_positive_support": int((positive.pairwise_inversions_vs_point > 0).sum()),
        },
        "interpretation_limit": (
            "Point estimates demonstrate the benchmark consequence. Rank uncertainty is not "
            "bootstrapped here; causal support and temporal uncertainty remain in the frozen S1/S2 analyses."
        ),
        "input": {"path": curves_path.relative_to(ROOT).as_posix(), "sha256": sha256(curves_path)},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    mpl.rcParams.update({
        "font.family": "Arial", "font.size": 7.5, "axes.titlesize": 8.5,
        "axes.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    row_keys = list(scorecard[group_columns].drop_duplicates().itertuples(index=False, name=None))
    supports = sorted(scorecard.support_km.unique())
    figure, axis = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    for y, keys in enumerate(row_keys):
        frame = scorecard[
            (scorecard.network == keys[0])
            & (scorecard.weighting == keys[1])
            & (scorecard.metric == keys[2])
        ].set_index("support_km")
        for x, support in enumerate(supports):
            row = frame.loc[support]
            axis.add_patch(plt.Rectangle(
                (x - 0.48, y - 0.42), 0.96, 0.84,
                facecolor=MODEL_COLOR[row.winner], edgecolor="white", linewidth=1,
            ))
            axis.text(x, y - 0.07, row.winner_label, ha="center", va="center",
                      color="white", fontweight="bold", fontsize=7.2)
            axis.text(x, y + 0.20, row.ranking_code, ha="center", va="center",
                      color="white", fontsize=6.3)
    labels = [
        f"{'Brazil' if key[0] == 'inmet_hourly' else 'United Kingdom'} · "
        f"{key[2].upper()} · {key[1]}"
        for key in row_keys
    ]
    axis.set_xticks(range(len(supports)), [f"{value:g}" for value in supports])
    axis.set_yticks(range(len(row_keys)), labels)
    axis.invert_yaxis()
    axis.set_xlabel("Observational reference support radius (km)")
    axis.set_xlim(-0.5, len(supports) - 0.5)
    axis.set_ylim(len(row_keys) - 0.5, -0.5)
    axis.set_title(
        "Forecast rankings depend on observational support",
        loc="left", fontweight="bold", fontsize=9, pad=24,
    )
    axis.text(
        0, 1.015,
        "Cell text: winner (top) and full order best→worst (G=GraphCast, I=IFS, P=Pangu)",
        transform=axis.transAxes, ha="left", va="bottom", fontsize=6.8, color="#444444",
    )
    for spine in axis.spines.values():
        spine.set_visible(False)
    png, pdf = output / "multisupport_scorecard.png", output / "multisupport_scorecard.pdf"
    figure.savefig(png, dpi=600, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)

    (output / "outputs_manifest.json").write_text(json.dumps({
        "script_sha256": sha256(Path(__file__).resolve()),
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in known[:-1]
        },
    }, indent=2), encoding="utf-8")
    print(json.dumps({"gates": summary["gates"], "counts": summary["counts"]}, indent=2))


if __name__ == "__main__":
    main()
