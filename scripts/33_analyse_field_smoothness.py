#!/usr/bin/env python3
"""Does the gridded reference favour the smoother forecast?

Three reversals point the same way — Pangu over GraphCast, the ensemble mean
over the deterministic run, and ERA5 over everything — which suggests a single
rule rather than three coincidences. The rule is testable: measure how much
small-scale structure each field carries over the domain the stations occupy,
and check it against the reference effects already measured.

Two objections shape how it is measured.

The first is that a window defined in grid cells is not a physical scale. Here
every source is verified to sit on the same 0.25 degree grid before anything is
computed, so cells do convert to kilometres identically; but three cells is
still 83 km north-south and 65 km east-west at this latitude, which is not one
scale either. Roughness is therefore defined by an isotropic Gaussian filter
whose width is given in kilometres, and reported at three widths. An ordering
that holds only at one width is not an ordering.

The second is that pairs are not independent units: each model appears in
several of them. The model-level statistic is reported alongside, and it is the
one the global phase should scale up.

Structure is read from the fields alone. No station enters this script.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from math import comb
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from public_zarr import PublicZarr

CONFIG = ROOT / "config" / "regional_t2m_2020.yaml"
CACHE = ROOT / "data" / "raw" / "weatherbench2_cache"
METRE_PER_DEGREE = 111_320.0
COHORTS = ("hres_initialised", "era5_initialised", "operational_ifs_pair")


def index_of(values: np.ndarray | pd.DatetimeIndex, wanted: object, name: str) -> int:
    found = np.flatnonzero(np.asarray(values) == wanted)
    if len(found) != 1:
        raise RuntimeError(f"{name}={wanted} is missing or non-unique")
    return int(found[0])


def source_specs(cfg: dict) -> dict[str, dict]:
    specs = dict(cfg["weatherbench2"]["models"])
    specs.update(dict(cfg["model_extensions"]))
    specs["era5"] = cfg["weatherbench2"]["era5"]
    return {name: specs[name] for name in cfg["field_smoothness"]["sources"]}


def verify_common_grid(specs: dict[str, dict]) -> float:
    """Abort unless every source holds the same cells, whatever their order.

    Comparing structure across products only means something on one grid: a
    window in cells is a different distance on a coarser one. The check is the
    premise of everything below, so it runs first and raises.
    """
    reference: tuple[np.ndarray, np.ndarray] | None = None
    for name, spec in specs.items():
        store = PublicZarr(spec["path"], CACHE)
        latitudes = np.sort(store.coordinate(spec["latitude_name"]))
        longitudes = np.sort(np.mod(store.coordinate(spec["longitude_name"]), 360))
        if reference is None:
            reference = (latitudes, longitudes)
            continue
        if latitudes.shape != reference[0].shape or not np.allclose(latitudes, reference[0]):
            raise RuntimeError(f"{name} does not share the common latitude grid")
        if longitudes.shape != reference[1].shape or not np.allclose(longitudes, reference[1]):
            raise RuntimeError(f"{name} does not share the common longitude grid")
    assert reference is not None
    return float(np.diff(reference[0]).mean())


def domain_window(store: PublicZarr, spec: dict, bounds: dict, margin_cells: int) -> tuple[slice, slice, int]:
    latitudes = store.coordinate(spec["latitude_name"])
    longitudes = np.mod(store.coordinate(spec["longitude_name"]) + 180, 360) - 180
    inside_lat = np.flatnonzero((latitudes >= bounds["south"]) & (latitudes <= bounds["north"]))
    inside_lon = np.flatnonzero((longitudes >= bounds["west"]) & (longitudes <= bounds["east"]))
    if len(inside_lat) < 5 or len(inside_lon) < 5:
        raise RuntimeError("the domain window is too small to measure structure")
    rows = slice(max(inside_lat.min() - margin_cells, 0), min(inside_lat.max() + 1 + margin_cells, len(latitudes)))
    columns = slice(max(inside_lon.min() - margin_cells, 0), min(inside_lon.max() + 1 + margin_cells, len(longitudes)))
    return rows, columns, margin_cells


def structure(field: np.ndarray, latitudes: np.ndarray, cell_degrees: float, scales_km: list[float],
              margin: int) -> dict[str, float]:
    """Small-scale structure of one patch, at scales given in kilometres.

    The filter is isotropic in kilometres, which on a latitude-longitude grid
    means a wider cell count along longitude than along latitude.
    """
    core = (slice(margin, field.shape[0] - margin), slice(margin, field.shape[1] - margin))
    cell_km_latitude = cell_degrees * METRE_PER_DEGREE / 1000
    cosine = float(np.cos(np.deg2rad(latitudes.mean())))
    cell_km_longitude = cell_km_latitude * cosine
    values = {"domain_sd_k": float(field[core].std())}
    north_south = np.abs(np.diff(field, axis=0))[core[0], core[1]] / cell_km_latitude * 100
    east_west = np.abs(np.diff(field, axis=1))[core[0], core[1]] / cell_km_longitude * 100
    values["mean_gradient_k_per_100km"] = float((north_south.mean() + east_west.mean()) / 2)
    for scale in scales_km:
        sigma = (scale / cell_km_latitude, scale / cell_km_longitude)
        residual = field - gaussian_filter(field, sigma=sigma, mode="nearest")
        values[f"roughness_{scale:g}km_k"] = float(residual[core].std())
    return values


def measure(name: str, spec: dict, times: pd.DatetimeIndex, lead: int | None, bounds: dict,
            cell_degrees: float, scales_km: list[float], margin: int) -> pd.DataFrame:
    store = PublicZarr(spec["path"], CACHE)
    rows_slice, columns_slice, used = domain_window(store, spec, bounds, margin)
    latitudes = store.coordinate(spec["latitude_name"])[rows_slice]
    store_times = store.times(spec["time_name"])
    lead_index = None if lead is None else index_of(store.timedeltas_hours(spec["lead_name"]), lead, "lead")

    def one(when: pd.Timestamp) -> dict:
        field = store.field2d(spec["t2m_name"], index_of(store_times, when, "time"), lead_index)
        return {"source": name, "time": when,
                **structure(field[rows_slice, columns_slice], latitudes, cell_degrees, scales_km, used)}

    with ThreadPoolExecutor(max_workers=4) as executor:
        rows = []
        for number, row in enumerate(executor.map(one, times), start=1):
            if number == 1 or number % 250 == 0 or number == len(times):
                print(f"smoothness {name}: {number}/{len(times)}", flush=True)
            rows.append(row)
    return pd.DataFrame(rows)


def measured_pairs() -> pd.DataFrame:
    frames = []
    for cohort in COHORTS:
        path = ROOT / "results" / f"regional_t2m_2020_{cohort}_pairwise.csv"
        if path.exists():
            frame = pd.read_csv(path).pivot(index=["model_a", "model_b"], columns="reference",
                                            values="delta_mae_a_minus_b_c").reset_index()
            frames.append(frame.assign(cohort=cohort))
    pairs = pd.concat(frames, ignore_index=True)
    pairs["reference_effect_c"] = pairs["avamet"] - pairs["era5_station"]
    return pairs.rename(columns={"avamet": "avamet_delta_mae_c", "era5_station": "era5_delta_mae_c"})


def measured_models() -> pd.DataFrame:
    """Per-model penalty for being judged by stations instead of the reanalysis.

    One row per model and cohort instead of one per pair, because a model that
    appears in several pairs contributes the same information to each of them.
    """
    frames = []
    for cohort in COHORTS:
        path = ROOT / "results" / f"regional_t2m_2020_{cohort}_scores.csv"
        if path.exists():
            frame = pd.read_csv(path).pivot(index="model", columns="reference", values="mae_c_day_weighted").reset_index()
            frames.append(frame.assign(cohort=cohort))
    models = pd.concat(frames, ignore_index=True)
    models["station_penalty_c"] = models["avamet"] - models["era5_station"]
    return models.rename(columns={"avamet": "mae_avamet_c", "era5_station": "mae_era5_c"})


def plot_rule(pairs: pd.DataFrame, models: pd.DataFrame, column: str, scale: float, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), layout="constrained")
    markers = {"hres_initialised": "o", "era5_initialised": "^", "operational_ifs_pair": "s"}
    for cohort, marker in markers.items():
        subset = pairs[pairs.cohort == cohort]
        axes[0].scatter(subset[f"roughness_a_minus_b_k"], subset.reference_effect_c, s=52, marker=marker,
                        color="#0072B2", edgecolors="black", linewidths=0.3, label=cohort.replace("_", " "))
    for _, row in pairs.iterrows():
        axes[0].annotate(f"{row.model_a.split('_')[0]}−{row.model_b.split('_')[0]}",
                         (row.roughness_a_minus_b_k, row.reference_effect_c), fontsize=7,
                         xytext=(4, 4), textcoords="offset points")
    rho_pairs = pairs.roughness_a_minus_b_k.corr(pairs.reference_effect_c, method="spearman")
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].axvline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set(xlabel="Diferencia de rugosidad A − B (K)", ylabel="Efecto de referencia A − B (°C)",
                title=f"Por pares (no independientes) · ρ = {rho_pairs:.2f}")
    axes[0].legend(fontsize=8, frameon=False, title="Cohorte", title_fontsize=8)

    for cohort, marker in markers.items():
        subset = models[models.cohort == cohort]
        axes[1].scatter(subset[column], subset.penalty_centred_c, s=52, marker=marker,
                        color="#D55E00", edgecolors="black", linewidths=0.3, label=cohort.replace("_", " "))
    for _, row in models.iterrows():
        axes[1].annotate(row.model.replace("_hres_init", "").replace("_", " "), (row[column], row.penalty_centred_c),
                         fontsize=7, xytext=(4, 4), textcoords="offset points")
    rho_models = models[column].corr(models.penalty_centred_c, method="spearman")
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set(xlabel=f"Rugosidad del modelo a {scale:g} km (K)",
                ylabel="Penalización por evaluar con estaciones (°C)\n[centrada en su cohorte]",
                title=f"Por modelo (unidad correcta) · ρ = {rho_models:.2f}")
    for axis in axes:
        axis.grid(color="#dddddd", linewidth=0.5)
    figure.suptitle(f"La referencia de rejilla favorece al campo más suave · filtro gaussiano de {scale:g} km", fontsize=11)
    figure.savefig(output.with_suffix(".png"), dpi=300, facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    family, paths = cfg["field_smoothness"], cfg["paths"]
    output = ROOT / paths["field_smoothness_results_directory"]
    output.mkdir(parents=True, exist_ok=True)
    specs = source_specs(cfg)
    cell_degrees = verify_common_grid(specs)
    scales = [float(scale) for scale in family["scales_km"]]
    margin = int(np.ceil(3 * max(scales) / (cell_degrees * METRE_PER_DEGREE / 1000)))

    stations = pd.read_csv(ROOT / paths["spatial_stations_csv"])
    padding = family["domain_padding_deg"]
    bounds = {"south": stations.latitude.min() - padding, "north": stations.latitude.max() + padding,
              "west": stations.longitude.min() - padding, "east": stations.longitude.max() + padding}
    panel = (ROOT / paths["annual_aligned_parquet"]).as_posix()
    initialisations = pd.DatetimeIndex(
        duckdb.sql(f"SELECT DISTINCT init_time FROM read_parquet('{panel}') ORDER BY init_time").fetchdf().init_time)
    lead = family["lead_hours"]

    frames = []
    for name, spec in specs.items():
        times = initialisations + pd.Timedelta(hours=lead) if name == "era5" else initialisations
        frames.append(measure(name, spec, times, None if name == "era5" else lead, bounds, cell_degrees, scales, margin))
    fields = pd.concat(frames, ignore_index=True)
    fields.to_csv(output / "field_structure_by_time.csv", index=False)

    columns = [f"roughness_{scale:g}km_k" for scale in scales]
    summary = fields.groupby("source", as_index=False)[["mean_gradient_k_per_100km", "domain_sd_k", *columns]].mean()
    counts = fields.groupby("source", as_index=False).time.size().rename(columns={"size": "n_fields"})
    summary = summary.merge(counts, on="source").sort_values(columns[0], ascending=False).reset_index(drop=True)
    summary.to_csv(output / "field_smoothness_summary.csv", index=False)

    pairs, models = measured_pairs(), measured_models()
    models = models[models.model.isin(summary.source)].merge(summary[["source", *columns]], left_on="model", right_on="source")
    models["penalty_centred_c"] = models.station_penalty_c - models.groupby("cohort").station_penalty_c.transform("mean")
    pairs = pairs[pairs.model_a.isin(summary.source) & pairs.model_b.isin(summary.source)].copy()

    rows = []
    for scale, column in zip(scales, columns, strict=True):
        roughness = summary.set_index("source")[column]
        pairs["roughness_a_minus_b_k"] = roughness.reindex(pairs.model_a).to_numpy() - roughness.reindex(pairs.model_b).to_numpy()
        pairs["favoured_by_avamet"] = np.where(pairs.reference_effect_c < 0, pairs.model_a, pairs.model_b)
        pairs["rougher_member"] = np.where(pairs.roughness_a_minus_b_k > 0, pairs.model_a, pairs.model_b)
        pairs["rule_holds"] = pairs.favoured_by_avamet == pairs.rougher_member
        agreeing, tested = int(pairs.rule_holds.sum()), len(pairs)
        rows.append({"scale_km": scale, "unit": "pair", "n": tested, "pairs_agreeing": agreeing,
                     "one_sided_sign_test_p": float(sum(comb(tested, k) for k in range(agreeing, tested + 1)) / 2**tested),
                     "spearman_rho": float(pairs.roughness_a_minus_b_k.corr(pairs.reference_effect_c, method="spearman")),
                     "roughness_order_rough_to_smooth": " > ".join(roughness.sort_values(ascending=False).index)})
        rows.append({"scale_km": scale, "unit": "model", "n": len(models), "pairs_agreeing": np.nan,
                     "one_sided_sign_test_p": np.nan,
                     "spearman_rho": float(models[column].corr(models.penalty_centred_c, method="spearman")),
                     "roughness_order_rough_to_smooth": " > ".join(roughness.sort_values(ascending=False).index)})
        pairs.rename(columns={"roughness_a_minus_b_k": f"roughness_diff_{scale:g}km_k"}).to_csv(
            output / f"pair_rule_check_{scale:g}km.csv", index=False)
    scale_table = pd.DataFrame(rows)
    scale_table.to_csv(output / "rule_by_scale.csv", index=False)
    models.drop(columns="source").to_csv(output / "model_level_penalty.csv", index=False)

    primary = float(family["primary_scale_km"])
    roughness = summary.set_index("source")[f"roughness_{primary:g}km_k"]
    pairs["roughness_a_minus_b_k"] = roughness.reindex(pairs.model_a).to_numpy() - roughness.reindex(pairs.model_b).to_numpy()
    plot_rule(pairs, models, f"roughness_{primary:g}km_k", primary, output / "reference_effect_vs_roughness")
    (output / "reference_effect_vs_roughness_alt_text.md").write_text(
        f"Dos paneles, ambos con filtro gaussiano de {primary:g} km. El izquierdo dispersa siete pares de modelos: en "
        "horizontal, cuánta estructura de pequeña escala tiene de más el modelo A que el B; en vertical, cuánto desplaza la "
        "referencia observacional su diferencia de MAE. El derecho usa el modelo como unidad: en horizontal su rugosidad, en "
        "vertical cuánto más error acumula al ser juzgado por estaciones en vez de por la reanálisis, centrado dentro de su "
        "cohorte. Los dos paneles descienden. Los marcadores distinguen las tres cohortes.", encoding="utf-8")

    verdict = {"pilot_name": cfg["pilot_name"], "analysis": "field smoothness versus the reference effect",
               "prediction": "the member favoured by the station reference is the rougher one",
               "lead_hours": lead, "domain": bounds, "scales_km": scales, "primary_scale_km": primary,
               "common_grid_degrees": cell_degrees,
               "common_grid_note": "every source is verified to hold the same cells before measuring, so a window in "
                                   "cells converts to kilometres identically; the filter is defined in kilometres anyway "
                                   "and is isotropic, which a window in cells is not at this latitude",
               "by_scale": scale_table.to_dict(orient="records"),
               "dependence_caveat": "pairs share models, so the pair-level sign test and correlation describe a coherent "
                                    "ordering rather than an independent sample; the model-level statistic is the unit "
                                    "the global phase should scale up, and here it rests on eight rows in three cohorts",
               "domain_caveat": "roughness is measured over the station domain only; a global ordering may differ",
               "causal_caveat": "this is an association between fields and scores; the controlled degradation of a single "
                                "forecast is what would make it causal",
               "status": "completed"}
    (output / "field_smoothness_summary.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(scale_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(models[["cohort", "model", "mae_era5_c", "mae_avamet_c", "station_penalty_c", "penalty_centred_c",
                  f"roughness_{primary:g}km_k"]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
