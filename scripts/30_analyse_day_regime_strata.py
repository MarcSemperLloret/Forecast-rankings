#!/usr/bin/env python3
"""Does the reference alter the ranking more on days when it diverges more?

The mechanism family answered this across stations. This answers it across
days: every valid day gets its own reference effect and its own regime
descriptors, and the two are related both as quantile strata, which read
easily, and as a rank correlation, which uses the whole range.

Within a stratum the days are no longer contiguous in the calendar, so the
moving blocks run along the stratum's own date-ordered sequence. The
stratification already breaks most of the serial dependence the blocks exist to
absorb; the four widths are reported so that the choice is visible.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "regional_t2m_2020.yaml"
MODELS = ("ifs_hres", "graphcast_hres_init", "pangu_hres_init")
REFERENCES = {"era5": "era5_t2m_c", "avamet": "avamet_t2m_qc_c"}


def interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    tail = (1 - confidence) / 2
    return tuple(np.quantile(values, [tail, 1 - tail]).tolist())


def block_indices(rng: np.random.Generator, days: int, draws: int, width: int) -> np.ndarray:
    starts = rng.integers(0, days, size=(draws, int(np.ceil(days / width))))
    return ((starts[:, :, None] + np.arange(width)) % days).reshape(draws, -1)[:, :days]


def daily_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per valid day: model errors under both references, plus regimes."""
    values = frame.copy()
    values["valid_day"] = pd.to_datetime(values.valid_time).dt.date
    observed, reanalysis = values.avamet_t2m_qc_c, values.era5_t2m_c
    columns = {f"{reference}__{model}": (values[f"{model}_t2m_c"] - values[column]).abs()
               for reference, column in REFERENCES.items() for model in MODELS}
    columns["avamet_network_mean_t2m_c"] = observed
    columns["era5_minus_avamet_mean_c"] = reanalysis - observed
    columns["era5_minus_avamet_squared"] = (reanalysis - observed) ** 2
    daily = pd.DataFrame({"valid_day": values.valid_day, **columns}).groupby("valid_day", as_index=False).mean(numeric_only=True)
    daily["era5_minus_avamet_rmse_c"] = np.sqrt(daily.pop("era5_minus_avamet_squared"))
    # ``observed`` is a groupby keyword, so the station column carries its own name.
    spatial = pd.DataFrame({"valid_day": values.valid_day, "station_id": values.station_id, "station_mean_t2m_c": observed})
    spatial = spatial.groupby(["valid_day", "station_id"], as_index=False)["station_mean_t2m_c"].mean()
    spatial = spatial.groupby("valid_day", as_index=False)["station_mean_t2m_c"].std()
    daily = daily.merge(spatial.rename(columns={"station_mean_t2m_c": "avamet_spatial_sd_c"}), on="valid_day", validate="one_to_one")
    return daily.sort_values("valid_day").reset_index(drop=True)


def effect_series(daily: pd.DataFrame, pair: tuple[str, str]) -> np.ndarray:
    first, second = pair
    return ((daily[f"avamet__{first}"] - daily[f"avamet__{second}"])
            - (daily[f"era5__{first}"] - daily[f"era5__{second}"])).to_numpy()


def score_days(daily: pd.DataFrame, pair: tuple[str, str], cfg: dict, seed_offset: int) -> list[dict]:
    """Bootstrap one set of days and contrast the two references on it."""
    bootstrap, family = cfg["bootstrap"], cfg["day_regime_strata"]
    first, second = (MODELS.index(model) for model in pair)
    errors = {reference: daily[[f"{reference}__{model}" for model in MODELS]].to_numpy() for reference in REFERENCES}
    rows = []
    for width in family["bootstrap_block_days"]:
        rng = np.random.default_rng(bootstrap["seed"] + seed_offset + width)
        indices = block_indices(rng, len(daily), bootstrap["n_resamples"], width)
        draws = {reference: values[indices].mean(axis=1) for reference, values in errors.items()}
        winners = {reference: np.argmin(values, axis=1) for reference, values in draws.items()}
        era = draws["era5"][:, first] - draws["era5"][:, second]
        avamet = draws["avamet"][:, first] - draws["avamet"][:, second]
        switch = avamet - era
        low, high = interval(switch, bootstrap["ci"])
        point = {reference: values.mean(axis=0) for reference, values in errors.items()}
        rows.append({"block_days": width, "n_days": len(daily),
                     "avamet_winner": MODELS[int(np.argmin(point["avamet"]))], "era5_winner": MODELS[int(np.argmin(point["era5"]))],
                     "p_winners_differ": float((winners["avamet"] != winners["era5"]).mean()),
                     "era5_delta_mae_c": point["era5"][first] - point["era5"][second],
                     "avamet_delta_mae_c": point["avamet"][first] - point["avamet"][second],
                     "primary_effect_c": switch.mean(), "primary_effect_ci_low_c": low, "primary_effect_ci_high_c": high,
                     "primary_pair_reversal_probability": float((era * avamet < 0).mean())})
    return rows


def spearman(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    ranked_first, ranked_second = rankdata(first, axis=0), rankdata(second, axis=0)
    centred_first = ranked_first - ranked_first.mean(axis=0)
    centred_second = ranked_second - ranked_second.mean(axis=0)
    return ((centred_first * centred_second).sum(axis=0)
            / np.sqrt((centred_first**2).sum(axis=0) * (centred_second**2).sum(axis=0)))


def plot_regimes(daily: pd.DataFrame, effects: np.ndarray, names: list[str], labels: dict[str, str],
                 strata: pd.DataFrame, lead: int, output: Path) -> None:
    figure, axes = plt.subplots(2, len(names), figsize=(3.4 * len(names), 6.6), layout="constrained")
    for column, name in enumerate(names):
        upper = axes[0, column]
        upper.scatter(daily[name], effects, s=14, alpha=0.6, color="#0072B2", edgecolors="none")
        rho = pd.Series(daily[name]).corr(pd.Series(effects), method="spearman")
        upper.axhline(0, color="black", linewidth=0.8, linestyle="--")
        upper.set(title=f"ρ = {rho:.2f}", xlabel=labels[name])
        upper.grid(color="#dddddd", linewidth=0.5)

        lower = axes[1, column]
        values = strata[strata.regime == name].sort_values("quantile")
        positions = np.arange(len(values))
        lower.errorbar(positions, values.primary_effect_c, fmt="o", markersize=5, color="#0072B2", ecolor="#0072B2",
                       elinewidth=1.0, capsize=2.5,
                       yerr=[values.primary_effect_c - values.primary_effect_ci_low_c,
                             values.primary_effect_ci_high_c - values.primary_effect_c])
        lower.axhline(0, color="black", linewidth=0.8, linestyle="--")
        lower.set_xticks(positions, values["quantile"])
        lower.set(xlabel=f"Cuartil · {labels[name].split(' (')[0]}")
        lower.grid(axis="y", color="#dddddd", linewidth=0.5)
    axes[0, 0].set_ylabel("Efecto del día (°C)\nGraphCast − Pangu, AVAMET menos ERA5")
    axes[1, 0].set_ylabel("Efecto del cuartil (°C)")
    figure.suptitle(f"El efecto de la referencia según el régimen del día (+{lead} h)", fontsize=11)
    figure.savefig(output.with_suffix(".png"), dpi=300, facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    family, bootstrap, paths = cfg["day_regime_strata"], cfg["bootstrap"], cfg["paths"]
    output = ROOT / paths["day_regime_results_directory"]
    output.mkdir(parents=True, exist_ok=True)
    pair, names = tuple(family["primary_pair"]), list(family["regimes"])
    daily_rows, stratum_rows, correlation_rows = [], [], []
    tables = {}
    for lead in family["leads_hours"]:
        source = ROOT / paths["spatial_directory"] / f"lead={lead:03d}" / "batches" / "*.parquet"
        frame = duckdb.sql(f"SELECT * FROM read_parquet('{source.as_posix()}')").fetchdf()
        frame = frame.dropna(subset=[*REFERENCES.values(), *(f"{model}_t2m_c" for model in MODELS)])
        daily = daily_table(frame)
        effects = effect_series(daily, pair)
        daily = daily.assign(lead_h=lead, day_effect_c=effects)
        tables[lead] = daily

        for row in score_days(daily, pair, cfg, seed_offset=lead * 1000):
            stratum_rows.append({"lead_h": lead, "regime": "all_days", "quantile": "all", "quantile_low": np.nan,
                                 "quantile_high": np.nan, **row})
        for name in names:
            labels = pd.qcut(daily[name], family["n_quantiles"], labels=[f"Q{index + 1}" for index in range(family["n_quantiles"])])
            edges = pd.qcut(daily[name], family["n_quantiles"]).cat.categories
            for label, category in zip(labels.cat.categories, edges, strict=True):
                subset = daily[labels == label].reset_index(drop=True)
                for row in score_days(subset, pair, cfg, seed_offset=lead * 1000 + sum(map(ord, f"{name}{label}"))):
                    stratum_rows.append({"lead_h": lead, "regime": name, "quantile": label,
                                         "quantile_low": float(category.left), "quantile_high": float(category.right), **row})
            values = daily[name].to_numpy(float)
            rho = float(spearman(effects[:, None], values[:, None])[0])
            for width in family["bootstrap_block_days"]:
                rng = np.random.default_rng(bootstrap["seed"] + lead * 100 + width + sum(map(ord, name)))
                indices = block_indices(rng, len(daily), bootstrap["n_resamples"], width)
                draws = spearman(effects[indices].T, values[indices].T)
                low, high = interval(draws, bootstrap["ci"])
                correlation_rows.append({"lead_h": lead, "regime": name, "label": family["regimes"][name], "block_days": width,
                                         "n_days": len(daily), "spearman_rho": rho, "spearman_rho_bootstrap_mean": float(draws.mean()),
                                         "ci_low": low, "ci_high": high,
                                         "p_same_sign": float((np.sign(draws) == np.sign(rho)).mean())})
        daily_rows.append(daily)
        print(f"analysed day regimes lead={lead}; days={len(daily)}", flush=True)

    pd.concat(daily_rows, ignore_index=True).to_csv(output / "daily_effects_and_regimes.csv", index=False)
    strata = pd.DataFrame(stratum_rows).sort_values(["lead_h", "regime", "quantile", "block_days"])
    strata.to_csv(output / "regime_strata.csv", index=False)
    correlations = pd.DataFrame(correlation_rows).sort_values(["lead_h", "regime", "block_days"])
    correlations.to_csv(output / "regime_correlations.csv", index=False)

    lead, width = family["primary_lead_hours"], family["bootstrap_block_days"][0]
    plot_regimes(tables[lead], tables[lead].day_effect_c.to_numpy(), names, family["regimes"],
                 strata[(strata.lead_h == lead) & (strata.block_days == width)], lead, output / "day_regime_effects")
    (output / "day_regime_effects_alt_text.md").write_text(
        f"Cuatro columnas, una por régimen: diferencia media ERA5 − AVAMET del día, RMSE ERA5 − AVAMET del día, temperatura "
        f"media de la red y variabilidad espacial de AVAMET. La fila superior dispersa el efecto de referencia GraphCast "
        f"menos Pangu de cada día frente al régimen de ese día, con su ρ de Spearman en el título; la inferior da el efecto "
        f"por cuartil del régimen con intervalos de confianza del 95 %. Todo a +{lead} h y bloque de {width} día; la línea "
        "discontinua marca el efecto nulo.", encoding="utf-8")
    (output / "day_regime_summary.json").write_text(json.dumps(
        {"pilot_name": cfg["pilot_name"], "analysis": "day-regime stratification of the reference effect",
         "leads_hours": family["leads_hours"], "regimes": names, "n_quantiles": family["n_quantiles"],
         "primary_pair": list(pair),
         "bootstrap": {"method": "circular moving-block bootstrap along the date-ordered days of each stratum",
                       "caveat": "within a stratum the days are not contiguous in the calendar",
                       "n_resamples": bootstrap["n_resamples"], "block_days": family["bootstrap_block_days"],
                       "ci": bootstrap["ci"], "seed": bootstrap["seed"]},
         "status": "completed"}, indent=2), encoding="utf-8")

    view = strata[(strata.lead_h == lead) & (strata.block_days == width) & (strata.regime != "all_days")]
    print(view[["regime", "quantile", "quantile_low", "quantile_high", "n_days", "primary_effect_c",
                "primary_effect_ci_low_c", "primary_effect_ci_high_c", "avamet_winner", "era5_winner",
                "p_winners_differ"]].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(correlations[(correlations.lead_h == lead) & (correlations.block_days == width)]
          .to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
