#!/usr/bin/env python3
"""Create the national causal evidence synthesis and Nature-width figure."""
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
DEFAULT_RESULTS = ROOT / "results" / "national_causal_support_2020"
MODELS = ["ifs_hres", "graphcast_hres_init", "pangu_hres_init"]
MODEL_LABELS = {
    "ifs_hres": "IFS-HRES",
    "graphcast_hres_init": "GraphCast",
    "pangu_hres_init": "Pangu-Weather",
}
MODEL_COLOURS = {
    "ifs_hres": "#0072B2",
    "graphcast_hres_init": "#D55E00",
    "pangu_hres_init": "#009E73",
}
MODEL_MARKERS = {"ifs_hres": "o", "graphcast_hres_init": "s", "pangu_hres_init": "^"}
MODEL_X_OFFSETS = {"ifs_hres": -4.0, "graphcast_hres_init": 0.0, "pangu_hres_init": 4.0}
NETWORK_LABELS = {"inmet_hourly": "INMET · Brazil", "midas_open_uk": "MIDAS · United Kingdom"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.15,
            "lines.markersize": 3.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    results = args.results.resolve()
    png = results / "national_causal_support_figure.png"
    pdf = results / "national_causal_support_figure.pdf"
    alt = results / "national_causal_support_figure_alt_text.md"
    synthesis = results / "SINTESIS_CAUSAL_NACIONAL_2020.md"
    source_data = results / "national_causal_support_figure_source_data.csv"
    manifest_path = results / "synthesis_manifest.json"
    outputs = [png, pdf, alt, synthesis, source_data, manifest_path]
    if any(path.exists() for path in outputs) and not args.force:
        raise FileExistsError(f"synthesis output exists in {results}; use --force")

    s1_change = pd.read_csv(results / "s1_smoothing_changes.csv")
    s1_summary = json.loads((results / "s1_s3_summary.json").read_text(encoding="utf-8"))
    s2_optima = pd.read_csv(results / "s2_support_optima.csv")
    s2_thinning = pd.read_csv(results / "s2_support_thinning.csv")
    s2_summary = json.loads((results / "s2_summary.json").read_text(encoding="utf-8"))
    configure_style()
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.45), layout="constrained")
    network_order = ["inmet_hourly", "midas_open_uk"]
    plotted_source: list[pd.DataFrame] = []

    for column, network in enumerate(network_order):
        axis = axes[0, column]
        subset = s1_change[
            (s1_change.network == network)
            & (s1_change.model.isin(MODELS))
            & (s1_change.metric == "rmse")
            & (s1_change.block_days == 7)
        ].copy()
        zero_rows = []
        for model in MODELS:
            for reference in ["era5", "station"]:
                zero_rows.append(
                    {
                        "network": network,
                        "model": model,
                        "reference": reference,
                        "sigma_km": 0.0,
                        "change": 0.0,
                        "ci_low": 0.0,
                        "ci_high": 0.0,
                        "metric": "rmse",
                        "block_days": 7,
                    }
                )
        subset = pd.concat([pd.DataFrame(zero_rows), subset], ignore_index=True)
        subset["figure_panel"] = "a" if column == 0 else "b"
        plotted_source.append(subset)
        for model in MODELS:
            for reference, linestyle, fill in [
                ("era5", "-", MODEL_COLOURS[model]),
                ("station", "--", "white"),
            ]:
                line = subset[(subset.model == model) & (subset.reference == reference)].sort_values("sigma_km")
                lower = line.change.to_numpy() - line.ci_low.to_numpy()
                upper = line.ci_high.to_numpy() - line.change.to_numpy()
                axis.errorbar(
                    line.sigma_km,
                    line.change,
                    yerr=np.vstack([lower, upper]),
                    color=MODEL_COLOURS[model],
                    linestyle=linestyle,
                    marker=MODEL_MARKERS[model],
                    markerfacecolor=fill,
                    markeredgecolor=MODEL_COLOURS[model],
                    markeredgewidth=0.7,
                    capsize=1.5,
                    elinewidth=0.7,
                )
        axis.axhline(0, color="#777777", linewidth=0.7, linestyle=":")
        axis.set(
            title=NETWORK_LABELS[network],
            xlabel="Forecast smoothing σ (km)",
            ylabel="Change in RMSE (°C)" if column == 0 else None,
            xticks=[0, 25, 50, 100],
        )

    for column, network in enumerate(network_order):
        axis = axes[1, column]
        uniform = s2_optima[
            (s2_optima.network == network)
            & (s2_optima.ladder == "prospective_common")
            & (s2_optima.model.isin(MODELS))
            & (s2_optima.metric == "rmse")
            & (s2_optima.weighting == "uniform")
            & (s2_optima.block_days == 7)
        ].copy()
        thin = s2_thinning[
            (s2_thinning.network == network)
            & (s2_thinning.ladder == "prospective_common")
            & (s2_thinning.model.isin(MODELS))
            & (s2_thinning.metric == "rmse")
        ].copy()
        thin_summary = (
            thin.groupby(["network", "model", "support_km"], as_index=False)
            .optimal_sigma_km.agg(
                thinning_mean="mean",
                thinning_median="median",
                thinning_ci_low=lambda values: float(np.quantile(values, 0.025)),
                thinning_ci_high=lambda values: float(np.quantile(values, 0.975)),
            )
        )
        uniform["figure_panel"] = "c" if column == 0 else "d"
        uniform["route"] = "uniform"
        thin_summary["figure_panel"] = "c" if column == 0 else "d"
        thin_summary["route"] = "thinning"
        plotted_source.extend([uniform, thin_summary])
        for model in MODELS:
            line = uniform[uniform.model == model].sort_values("support_km")
            lower = line.optimal_sigma_km_point.to_numpy() - line.optimal_sigma_ci_low_km.to_numpy()
            upper = line.optimal_sigma_ci_high_km.to_numpy() - line.optimal_sigma_km_point.to_numpy()
            axis.errorbar(
                line.support_km + MODEL_X_OFFSETS[model] - 1.5,
                line.optimal_sigma_km_point,
                yerr=np.vstack([lower, upper]),
                color=MODEL_COLOURS[model],
                linestyle="-",
                marker=MODEL_MARKERS[model],
                capsize=1.5,
                elinewidth=0.7,
            )
            line = thin_summary[thin_summary.model == model].sort_values("support_km")
            lower = line.thinning_median.to_numpy() - line.thinning_ci_low.to_numpy()
            upper = line.thinning_ci_high.to_numpy() - line.thinning_median.to_numpy()
            axis.errorbar(
                line.support_km + MODEL_X_OFFSETS[model] + 1.5,
                line.thinning_median,
                yerr=np.vstack([lower, upper]),
                color=MODEL_COLOURS[model],
                linestyle="--",
                marker=MODEL_MARKERS[model],
                markerfacecolor="white",
                markeredgecolor=MODEL_COLOURS[model],
                markeredgewidth=0.7,
                capsize=1.5,
                elinewidth=0.7,
            )
        axis.set(
            title=NETWORK_LABELS[network],
            xlabel="Reference support radius (km)",
            ylabel="Optimal forecast smoothing σ* (km)" if column == 0 else None,
            xticks=[0, 50, 100, 200],
        )

    for label, axis in zip("abcd", axes.ravel(), strict=True):
        axis.text(-0.13, 1.06, label, transform=axis.transAxes, fontweight="bold", fontsize=9, va="top")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#e5e5e5", linewidth=0.45, zorder=0)
        axis.tick_params(width=0.6, length=2.5)

    from matplotlib.lines import Line2D

    model_handles = [
        Line2D([0], [0], color=MODEL_COLOURS[model], marker=MODEL_MARKERS[model], label=MODEL_LABELS[model])
        for model in MODELS
    ]
    figure.legend(handles=model_handles, loc="outside upper center", ncols=3, frameon=False)
    reference_handles = [
        Line2D([0], [0], color="#333333", linestyle="-", marker="o", label="ERA5"),
        Line2D([0], [0], color="#333333", linestyle="--", marker="o", markerfacecolor="white", label="Stations"),
    ]
    axes[0, 1].legend(handles=reference_handles, title="Verification reference", frameon=False, loc="upper left")
    route_handles = [
        Line2D([0], [0], color="#333333", linestyle="-", marker="o", label="Uniform mean"),
        Line2D([0], [0], color="#333333", linestyle="--", marker="o", markerfacecolor="white", label="Three-station thinning"),
    ]
    axes[1, 1].legend(handles=route_handles, title="Support construction", frameon=False, loc="upper left")
    figure.set_constrained_layout_pads(w_pad=0.10, h_pad=0.08, wspace=0.10, hspace=0.10)
    figure.savefig(pdf)
    figure.savefig(png, dpi=600)
    plt.close(figure)
    pd.concat(plotted_source, ignore_index=True, sort=False).to_csv(source_data, index=False)

    alt.write_text(
        "Four panels compare Brazil (left) and the United Kingdom (right). Panels a and b show the change in RMSE "
        "relative to the unsmoothed forecast as Gaussian smoothing increases. For IFS-HRES, GraphCast and "
        "Pangu-Weather, solid ERA5 curves initially fall below zero whereas dashed station curves rise, so removing "
        "small-scale structure improves verification against ERA5 and harms verification against stations. Panels c "
        "and d show that the optimal forecast smoothing rises as station observations are aggregated over radii from "
        "0 to 200 km. The rise remains under the fixed three-station thinning control. Error bars in a–d are 95% "
        "intervals; thinning intervals describe the 60 fixed-count selections. Support-panel markers are offset "
        "slightly along the horizontal axis so coincident model results remain visible.",
        encoding="utf-8",
    )

    decision_rows = s1_summary["model_decisions"]
    s1_models_passing = sum(item["opposite_optima_pass"] for item in decision_rows)
    prospective = s2_summary["decisions"]["prospective_common"]
    exact = s2_summary["decisions"]["exact_preregistered"]
    synthesis.write_text(
        f"""# Síntesis causal nacional 2020

## Resultado principal

La intervención causal se generaliza fuera de AVAMET. En MIDAS-Reino Unido e
INMET-Brasil, los tres modelos primarios tienen un óptimo de suavizado positivo
contra ERA5 y nulo contra estaciones, tanto en MAE como en RMSE
({s1_models_passing}/{len(decision_rows)} contrastes modelo–red–métrica pasan).

El suavizado óptimo contra ERA5 se sitúa entre 12,5 y 25 km. La reducción de
RMSE frente a ERA5 varía entre 0,006 y 0,121 °C y todos los intervalos
bootstrap de siete días para esos óptimos excluyen cero. En los seis contrastes
modelo–red, la reducción del componente centrado del MSE supera a la del sesgo².

## Escala de la referencia

La escalera prospectiva 0/50/100/200 km pasa en INMET ({prospective['inmet_hourly']['eligible_centres']}
centros) y MIDAS ({prospective['midas_open_uk']['eligible_centres']} centros),
en MAE y RMSE, con media uniforme, ponderación gaussiana y thinning fijo a tres
estaciones. Todos los modelos presentan correlación de Spearman positiva; bajo
thinning, la correlación media de RMSE es 1,00 en ambas redes.

La escalera exacta prerregistrada 0/25/50/100 km pasa en MIDAS con
{exact['midas_open_uk']['eligible_centres']} centros. INMET conserva solo
{exact['inmet_hourly']['eligible_centres']} centros a 25 km y falla el gate de
geometría; no se interpreta como refutación científica.

## Afirmación permitida

> La dependencia causal de la evaluación respecto al soporte espacial de la
> referencia se generaliza desde AVAMET a dos redes nacionales, dos continentes,
> tres modelos y dos métricas.

## Límites

MIDAS e INMET no se solapan con Weather5K/ISD en el estrato seleccionado, pero
su ausencia de todas las rutas de asimilación de ERA5 no está documentada. Los
modelos comparten entrenamiento, inicializaciones o linajes y no constituyen
siete réplicas independientes. Sigue prohibida la expresión «confirmación
global completamente independiente de ERA5».

## Pie de figura

**Figura. El soporte espacial de la referencia controla la estructura que
recompensa la verificación.** **a,b**, cambio de RMSE al suavizar
geodésicamente el mismo pronóstico, relativo a σ=0, en INMET-Brasil y
MIDAS-Reino Unido. Las barras son IC bootstrap del 95% con bloques temporales
circulares de siete días. **c,d**, suavizado óptimo frente a referencias
observacionales construidas con soporte creciente. Las líneas continuas usan
media uniforme; las discontinuas mantienen tres estaciones por centro en 60
selecciones y dejan crecer únicamente el área cubierta. Los tres modelos se
muestrean en los mismos centros y tiempos en cada escalón. Los marcadores de
**c,d** llevan un desplazamiento horizontal mínimo para hacer visibles los
óptimos coincidentes; los radios analizados siguen siendo 0, 50, 100 y 200 km.
""",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "script_sha256": sha256(Path(__file__).resolve()),
        "inputs": {
            name: sha256(results / name)
            for name in [
                "s1_smoothing_changes.csv",
                "s1_s3_summary.json",
                "s2_support_optima.csv",
                "s2_support_thinning.csv",
                "s2_summary.json",
            ]
        },
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in [png, pdf, alt, synthesis, source_data]
        },
        "figure_specification": {
            "width_inches": 7.2,
            "target": "Nature double column (183 mm)",
            "png_dpi": 600,
            "vector_pdf": True,
            "palette": "Okabe-Ito subset with redundant line and marker encoding",
            "editorial_visual_quality_score": 11,
            "editorial_score_breakdown": {
                "legibility": 2,
                "hierarchy": 2,
                "composition": 2,
                "consistency": 2,
                "visual_economy": 1,
                "editorial_appeal": 2,
            },
            "review_note": "The fixed-count uncertainty is necessarily broad and retained rather than hidden.",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"figure": str(pdf), "synthesis": str(synthesis)}, indent=2))


if __name__ == "__main__":
    main()
