#!/usr/bin/env python3
"""Leave one geographic region out, and rank the models without it.

The spatial block bootstrap already says the effect survives spatial
resampling, but it cannot say which part of the network carries it. This does:
the stations are partitioned into geographic regions, each region is removed in
turn, and the whole comparison is repeated on what is left. Each region is also
scored on its own, so a region with an unusual local behaviour is visible
rather than inferred from its removal.

The partitions are geometric and fixed by seed. None is drawn around a result.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from pyproj import Transformer
from scipy.cluster.vq import kmeans2

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


def daily_errors(frame: pd.DataFrame, reference: str) -> pd.DataFrame:
    values = {"valid_day": pd.to_datetime(frame.valid_time).dt.date}
    for model in MODELS:
        values[model] = (frame[f"{model}_t2m_c"] - frame[REFERENCES[reference]]).abs()
    return pd.DataFrame(values).groupby("valid_day", as_index=False).mean(numeric_only=True).sort_values("valid_day").reset_index(drop=True)


def partition_stations(stations: pd.DataFrame, name: str, spec: dict, cfg: dict) -> pd.Series:
    """Assign each station to a region under one pre-specified partition."""
    family = cfg["leave_one_region_out"]
    to_metres = Transformer.from_crs("EPSG:4326", family["projection"], always_xy=True).transform
    eastings, northings = to_metres(stations.longitude.to_numpy(), stations.latitude.to_numpy())
    if spec["method"] == "kmeans":
        coordinates = np.column_stack([eastings, northings]) / 1000.0
        centroids, labels = kmeans2(coordinates, spec["clusters"], minit="++", seed=family["kmeans_seed"], missing="raise")
        order = np.argsort(centroids[:, 1])[::-1]
        renumbered = np.empty(len(order), dtype=int)
        renumbered[order] = np.arange(len(order))
        return pd.Series([f"{name}_r{renumbered[label]:02d}" for label in labels], index=stations.index)
    if spec["method"] == "grid_blocks":
        degrees = spec["degrees"]
        blocks = (np.floor(stations.latitude / degrees).astype(int).astype(str) + "_"
                  + np.floor(stations.longitude / degrees).astype(int).astype(str))
        return name + "_" + blocks
    raise ValueError(f"unknown partition method: {spec['method']}")


def analyse_fold(frame: pd.DataFrame, cfg: dict) -> list[dict]:
    """Score one subset of stations under both references."""
    daily = {reference: daily_errors(frame, reference) for reference in REFERENCES}
    if not daily["era5"].valid_day.equals(daily["avamet"].valid_day):
        raise RuntimeError("references do not share the same valid-day blocks")
    bootstrap, family = cfg["bootstrap"], cfg["leave_one_region_out"]
    first, second = (MODELS.index(model) for model in family["primary_pair"])
    rows = []
    for width in family["bootstrap_block_days"]:
        rng = np.random.default_rng(bootstrap["seed"] + width * 10 + len(frame) % 997)
        indices = block_indices(rng, len(daily["avamet"]), bootstrap["n_resamples"], width)
        draws, winners = {}, {}
        for reference, values in daily.items():
            sampled = values[list(MODELS)].to_numpy()[indices].mean(axis=1)
            draws[reference], winners[reference] = sampled, np.argmin(sampled, axis=1)
        era = draws["era5"][:, first] - draws["era5"][:, second]
        avamet = draws["avamet"][:, first] - draws["avamet"][:, second]
        switch = avamet - era
        low, high = interval(switch, bootstrap["ci"])
        point = {reference: values[list(MODELS)].mean().to_numpy() for reference, values in daily.items()}
        rows.append({"block_days": width, "n_common_cases": len(frame), "n_stations": frame.station_id.nunique(),
                     "n_daily_blocks": len(daily["avamet"]),
                     "avamet_winner": MODELS[int(np.argmin(point["avamet"]))], "era5_winner": MODELS[int(np.argmin(point["era5"]))],
                     "p_winners_differ": float((winners["avamet"] != winners["era5"]).mean()),
                     "era5_delta_mae_c": point["era5"][first] - point["era5"][second],
                     "avamet_delta_mae_c": point["avamet"][first] - point["avamet"][second],
                     "primary_effect_c": switch.mean(), "primary_effect_ci_low_c": low, "primary_effect_ci_high_c": high,
                     "p_effect_gt_zero": float((switch > 0).mean()), "primary_pair_reversal_probability": float((era * avamet < 0).mean())})
    return rows


def plot_regions(stations: pd.DataFrame, folds: pd.DataFrame, partition: str, lead: int, width: int, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), layout="constrained")
    regions = sorted(stations.region.unique())
    colours = plt.get_cmap("tab20")(np.linspace(0, 1, 20))[: len(regions)]
    for colour, region in zip(colours, regions, strict=True):
        subset = stations[stations.region == region]
        axes[0].scatter(subset.longitude, subset.latitude, s=26, color=colour, edgecolors="black", linewidths=0.25,
                        label=f"{region.split('_r')[-1]} (n={len(subset)})")
    axes[0].set(xlabel="Longitud (°)", ylabel="Latitud (°)", title=f"Regiones · {partition}")
    axes[0].legend(fontsize=6, frameon=False, ncols=2, title="Región", title_fontsize=7)
    axes[0].grid(color="#dddddd", linewidth=0.5)

    values = folds[(folds.fold_type == "leave_one_out") & (folds.block_days == width)].sort_values("primary_effect_c")
    full = folds[(folds.fold_type == "all_stations") & (folds.block_days == width)].primary_effect_c.iloc[0]
    positions = np.arange(len(values))
    axes[1].errorbar(values.primary_effect_c, positions, fmt="o", markersize=5, color="#0072B2", ecolor="#0072B2",
                     elinewidth=1.0, capsize=2.5,
                     xerr=[values.primary_effect_c - values.primary_effect_ci_low_c, values.primary_effect_ci_high_c - values.primary_effect_c])
    axes[1].axvline(full, color="black", linewidth=0.9, linestyle="--", label=f"todas las estaciones ({full:.3f} °C)")
    axes[1].axvline(0, color="#aa3377", linewidth=0.9, linestyle=":", label="efecto nulo")
    axes[1].set_yticks(positions, [region.split("_r")[-1] for region in values.region])
    axes[1].set(xlabel="Efecto de referencia GraphCast − Pangu (°C)", ylabel="Región excluida",
                title=f"Dejando fuera una región (+{lead} h, bloque {width} d)")
    axes[1].legend(fontsize=7, frameon=False, loc="best")
    axes[1].grid(axis="x", color="#dddddd", linewidth=0.5)
    figure.savefig(output.with_suffix(".png"), dpi=300, facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    family, paths = cfg["leave_one_region_out"], cfg["paths"]
    output = ROOT / paths["leave_one_region_out_results_directory"]
    output.mkdir(parents=True, exist_ok=True)
    stations = pd.read_csv(ROOT / paths["spatial_stations_csv"])[["station_id", "latitude", "longitude", "altitude_m", "coast_distance_km"]]
    memberships = {name: partition_stations(stations, name, spec, cfg) for name, spec in family["partitions"].items()}
    driver_path = ROOT / paths["station_effect_driver_results_directory"] / "station_effects_and_covariates.csv"
    drivers = pd.read_csv(driver_path) if driver_path.exists() else None

    definitions, fold_rows, region_rows = [], [], []
    for lead in family["leads_hours"]:
        source = ROOT / paths["spatial_directory"] / f"lead={lead:03d}" / "batches" / "*.parquet"
        frame = duckdb.sql(f"SELECT * FROM read_parquet('{source.as_posix()}')").fetchdf()
        frame = frame.dropna(subset=[*REFERENCES.values(), *(f"{model}_t2m_c" for model in MODELS)])
        for partition, membership in memberships.items():
            assignment = stations.assign(region=membership.to_numpy())
            panel = frame.merge(assignment[["station_id", "region"]], on="station_id", validate="many_to_one")
            baseline = analyse_fold(panel, cfg)
            for row in baseline:
                fold_rows.append({"lead_h": lead, "partition": partition, "fold_type": "all_stations", "region": "none", **row})
            reference_effect = {row["block_days"]: row["primary_effect_c"] for row in baseline}
            for region in sorted(assignment.region.unique()):
                retained = panel[panel.region != region]
                for row in analyse_fold(retained, cfg):
                    fold_rows.append({"lead_h": lead, "partition": partition, "fold_type": "leave_one_out", "region": region,
                                      "effect_shift_c": row["primary_effect_c"] - reference_effect[row["block_days"]], **row})
                only = panel[panel.region == region]
                if only.station_id.nunique() >= 2:
                    for row in analyse_fold(only, cfg):
                        fold_rows.append({"lead_h": lead, "partition": partition, "fold_type": "single_region", "region": region, **row})
        print(f"analysed leave-one-region-out lead={lead}; partitions={len(memberships)}", flush=True)

    for partition, membership in memberships.items():
        assignment = stations.assign(partition=partition, region=membership.to_numpy())
        definitions.append(assignment)
        summary = assignment.groupby(["partition", "region"], as_index=False).agg(
            n_stations=("station_id", "size"), mean_latitude=("latitude", "mean"), mean_longitude=("longitude", "mean"),
            mean_altitude_m=("altitude_m", "mean"), mean_coast_distance_km=("coast_distance_km", "mean"))
        if drivers is not None:
            local = drivers[drivers.lead_h == family["primary_lead_hours"]].merge(
                assignment[["station_id", "region"]], on="station_id", validate="one_to_one")
            summary = summary.merge(local.groupby("region", as_index=False).agg(
                mean_station_effect_c=("station_effect_c", "mean"), mean_delta_z_m=("delta_z_m", "mean"),
                mean_era5_minus_avamet_bias_c=("era5_minus_avamet_bias_c", "mean")), on="region", how="left")
        region_rows.append(summary)

    folds = pd.DataFrame(fold_rows).sort_values(["lead_h", "partition", "fold_type", "region", "block_days"])
    folds.to_csv(output / "leave_one_region_out_folds.csv", index=False)
    pd.concat(definitions, ignore_index=True).to_csv(output / "region_definitions.csv", index=False)
    regions = pd.concat(region_rows, ignore_index=True)
    regions.to_csv(output / "region_summary.csv", index=False)

    lead, partition, width = family["primary_lead_hours"], family["primary_partition"], family["bootstrap_block_days"][0]
    selection = folds[(folds.lead_h == lead) & (folds.partition == partition)]
    plot_regions(stations.assign(region=memberships[partition].to_numpy()), selection, partition, lead, width,
                 output / "leave_one_region_out_effects")
    (output / "leave_one_region_out_effects_alt_text.md").write_text(
        f"Dos paneles. El izquierdo sitúa las 146 estaciones AVAMET en longitud y latitud, coloreadas por la región "
        f"geométrica a la que pertenecen bajo la partición {partition}. El derecho muestra, para cada región excluida, el "
        f"efecto de referencia GraphCast menos Pangu recalculado sin ella, con su intervalo de confianza del 95 %, a +{lead} h "
        f"y bloque de {width} día. La línea discontinua marca el efecto con todas las estaciones y la punteada el efecto nulo.",
        encoding="utf-8")
    (output / "leave_one_region_out_summary.json").write_text(json.dumps(
        {"pilot_name": cfg["pilot_name"], "analysis": "leave-one-geographic-region-out replication",
         "leads_hours": family["leads_hours"], "partitions": family["partitions"], "projection": family["projection"],
         "primary_partition": partition, "primary_pair": family["primary_pair"],
         "fold_types": ["all_stations", "leave_one_out", "single_region"],
         "bootstrap": {"method": "circular moving-block bootstrap on ordered valid days", "n_resamples": cfg["bootstrap"]["n_resamples"],
                       "block_days": family["bootstrap_block_days"], "ci": cfg["bootstrap"]["ci"], "seed": cfg["bootstrap"]["seed"]},
         "status": "completed"}, indent=2), encoding="utf-8")

    columns = ["region", "n_stations", "primary_effect_c", "primary_effect_ci_low_c", "primary_effect_ci_high_c",
               "effect_shift_c", "avamet_winner", "era5_winner", "p_winners_differ"]
    view = selection[(selection.fold_type == "leave_one_out") & (selection.block_days == width)][columns]
    print(view.sort_values("primary_effect_c").to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    if drivers is not None:
        print(regions[regions.partition == partition].to_string(index=False, float_format=lambda value: f"{value:.2f}"))


if __name__ == "__main__":
    main()
