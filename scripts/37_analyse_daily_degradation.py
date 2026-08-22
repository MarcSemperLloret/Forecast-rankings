#!/usr/bin/env python3
"""Does the finding survive being degraded to daily maxima and minima?

The global station archives that hold most of the stations — GHCN-Daily, GSOD,
ECA&D — keep no instantaneous values, only daily extremes. Whether they can be
used is not a matter of opinion: it can be decided here, in the one domain where
the instantaneous answer is already known.

The forecast day is built from the 00 UTC run at +24, +30, +36 and +42 h, which
lands on 00, 06, 12 and 18 UTC of the same UTC day. AVAMET reports sub-hourly,
so the observed extremes are computed twice: from the full record, and from the
same four instants the forecast is sampled at. The gap between the two is the
sampling bias that a four-times-daily archive would carry, measured instead of
assumed.

A UTC day is not the local observation window a daily archive actually uses.
That mismatch is the next question, not this one, and it is recorded as a limit.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "regional_t2m_2020.yaml"
MODELS = ("ifs_hres", "graphcast_hres_init", "pangu_hres_init")


def interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    tail = (1 - confidence) / 2
    return tuple(np.quantile(values, [tail, 1 - tail]).tolist())


def block_indices(rng: np.random.Generator, days: int, draws: int, width: int) -> np.ndarray:
    starts = rng.integers(0, days, size=(draws, int(np.ceil(days / width))))
    return ((starts[:, :, None] + np.arange(width)) % days).reshape(draws, -1)[:, :days]


def observed_extremes(cfg: dict, stations: pd.Series) -> pd.DataFrame:
    """Daily maxima and minima from the full sub-hourly record, frozen QC applied."""
    archive = (ROOT / cfg["avamet"]["archive_glob"]).resolve().as_posix()
    connection = duckdb.connect()
    connection.execute("SET TimeZone='UTC'")
    connection.register("wanted", stations.to_frame("station_id"))
    values = connection.execute(
        f"""SELECT a.station_id, a.observed_utc, a.temperature_c
              FROM read_parquet('{archive}', hive_partitioning=true) AS a
              INNER JOIN wanted USING (station_id)
             WHERE a.temperature_c IS NOT NULL
             ORDER BY a.station_id, a.observed_utc"""
    ).fetchdf()
    low, high = cfg["avamet"]["t2m_plausible_c"]
    qc = cfg["avamet"]["qc"]
    physical = values.temperature_c.between(low, high)
    masked = values.temperature_c.where(physical)
    median = masked.groupby(values.station_id).transform(
        lambda item: item.rolling(qc["rolling_window_reports"], center=True, min_periods=3).median())
    values["clean"] = values.temperature_c.where(physical & ((masked - median).abs() <= qc["local_median_deviation_c"]))
    values["valid_day"] = pd.to_datetime(values.observed_utc, utc=True).dt.tz_localize(None).dt.date
    grouped = values.dropna(subset=["clean"]).groupby(["station_id", "valid_day"], as_index=False)
    return grouped.clean.agg(["max", "min", "mean", "size"]).rename(
        columns={"max": "avamet_true_tmax_c", "min": "avamet_true_tmin_c",
                 "mean": "avamet_true_tmean_c", "size": "reports_per_day"})


def daily_panel(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """One row per station and UTC day, with every quantity built from four instants."""
    family, paths = cfg["daily_resolution_degradation"], cfg["paths"]
    leads = family["window_leads_hours"]
    frames = []
    for lead in leads:
        source = (ROOT / paths["spatial_directory"] / f"lead={lead:03d}" / "batches" / "*.parquet").as_posix()
        columns = ", ".join([*(f"{model}_t2m_c" for model in MODELS), "era5_t2m_c", "avamet_t2m_qc_c"])
        frame = duckdb.sql(f"SELECT station_id, init_time, valid_time, {columns} FROM read_parquet('{source}')").fetchdf()
        frames.append(frame[pd.to_datetime(frame.init_time).dt.hour == family["init_cycle_utc"]].assign(lead_h=lead))
    stacked = pd.concat(frames, ignore_index=True)
    stacked["valid_day"] = pd.to_datetime(stacked.valid_time).dt.date
    stacked = stacked.dropna(subset=[*(f"{model}_t2m_c" for model in MODELS), "era5_t2m_c", "avamet_t2m_qc_c"])
    complete = stacked.groupby(["station_id", "valid_day"]).lead_h.transform("nunique") == len(leads)
    stacked = stacked[complete]
    sources = {f"{model}_t2m_c": model for model in MODELS} | {"era5_t2m_c": "era5", "avamet_t2m_qc_c": "avamet_sampled"}
    grouped = stacked.groupby(["station_id", "valid_day"], as_index=False)
    panel = grouped.agg(**{f"{name}_{quantity}_c": (column, quantity)
                           for column, name in sources.items() for quantity in ("max", "min", "mean")})
    panel = panel.rename(columns={column: column.replace("_max_c", "_tmax_c").replace("_min_c", "_tmin_c")
                                  .replace("_mean_c", "_tmean_c") for column in panel.columns})
    observed = observed_extremes(cfg, panel.station_id.drop_duplicates())
    panel = panel.merge(observed, on=["station_id", "valid_day"], how="inner", validate="one_to_one")
    # Many daily archives publish an average derived as the midpoint of the two
    # extremes rather than a true mean. That derivation is tested as its own
    # quantity, because it inherits the extremes' sampling bias.
    for name in [*MODELS, "era5", "avamet_sampled", "avamet_true"]:
        panel[f"{name}_tmidpoint_c"] = (panel[f"{name}_tmax_c"] + panel[f"{name}_tmin_c"]) / 2
    return panel, {"leads_hours": leads, "init_cycle_utc": family["init_cycle_utc"]}


def score(panel: pd.DataFrame, quantity: str, reference: str, cfg: dict) -> tuple[list[dict], dict]:
    family, bootstrap = cfg["daily_resolution_degradation"], cfg["bootstrap"]
    first, second = (MODELS.index(model) for model in family["primary_pair"])
    errors = pd.DataFrame({"valid_day": panel.valid_day,
                           **{model: (panel[f"{model}_{quantity}_c"] - panel[f"{reference}_{quantity}_c"]).abs()
                              for model in MODELS}})
    daily = errors.groupby("valid_day", as_index=False).mean(numeric_only=True).sort_values("valid_day").reset_index(drop=True)
    point = daily[list(MODELS)].mean().to_numpy()
    rows = []
    draws_by_width = {}
    for width in family["bootstrap_block_days"]:
        rng = np.random.default_rng(bootstrap["seed"] + width + sum(map(ord, f"{quantity}{reference}")))
        indices = block_indices(rng, len(daily), bootstrap["n_resamples"], width)
        sampled = daily[list(MODELS)].to_numpy()[indices].mean(axis=1)
        draws_by_width[width] = sampled
        for position, model in enumerate(MODELS):
            low, high = interval(sampled[:, position], bootstrap["ci"])
            rows.append({"quantity": quantity, "reference": reference, "block_days": width, "model": model,
                         "n_daily_blocks": len(daily), "n_cases": len(panel), "mae_c": point[position],
                         "mae_ci_low_c": low, "mae_ci_high_c": high,
                         "mae_rank": int(pd.Series(point).rank(method="min").iloc[position])})
    return rows, {"point": point, "draws": draws_by_width, "first": first, "second": second, "n_days": len(daily)}


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    family, bootstrap, paths = cfg["daily_resolution_degradation"], cfg["bootstrap"], cfg["paths"]
    output = ROOT / paths["daily_degradation_results_directory"]
    output.mkdir(parents=True, exist_ok=True)
    panel, provenance = daily_panel(cfg)
    if panel.empty:
        raise RuntimeError("no complete daily windows; check that leads 24, 30, 36 and 42 are all materialised")

    references = ["era5", "avamet_sampled", "avamet_true"]
    score_rows, effect_rows, states = [], [], {}
    for quantity in ("tmax", "tmin", "tmean", "tmidpoint"):
        for reference in references:
            rows, state = score(panel, quantity, reference, cfg)
            score_rows.extend(rows)
            states[(quantity, reference)] = state
        for observed in ("avamet_sampled", "avamet_true"):
            reanalysis, station = states[(quantity, "era5")], states[(quantity, observed)]
            for width in family["bootstrap_block_days"]:
                era = reanalysis["draws"][width][:, reanalysis["first"]] - reanalysis["draws"][width][:, reanalysis["second"]]
                avamet = station["draws"][width][:, station["first"]] - station["draws"][width][:, station["second"]]
                switch = avamet - era
                low, high = interval(switch, bootstrap["ci"])
                effect_rows.append({"quantity": quantity, "observed_reference": observed, "block_days": width,
                                    "era5_winner": MODELS[int(np.argmin(reanalysis["point"]))],
                                    "avamet_winner": MODELS[int(np.argmin(station["point"]))],
                                    "era5_delta_mae_c": reanalysis["point"][reanalysis["first"]] - reanalysis["point"][reanalysis["second"]],
                                    "avamet_delta_mae_c": station["point"][station["first"]] - station["point"][station["second"]],
                                    "reference_effect_c": float(switch.mean()), "effect_ci_low_c": low,
                                    "effect_ci_high_c": high,
                                    "p_effect_lt_zero": float((switch < 0).mean()),
                                    "p_pairwise_order_reverses": float((era * avamet < 0).mean())})

    scores = pd.DataFrame(score_rows).sort_values(["quantity", "reference", "block_days", "mae_rank"])
    scores.to_csv(output / "daily_scores.csv", index=False)
    effects = pd.DataFrame(effect_rows).sort_values(["quantity", "observed_reference", "block_days"])
    effects.to_csv(output / "daily_reference_effects.csv", index=False)

    quantities = ["tmax", "tmin", "tmean", "tmidpoint"]
    sampling = pd.DataFrame({
        "quantity": quantities,
        "sampled_minus_true_mean_c": [float((panel[f"avamet_sampled_{quantity}_c"] - panel[f"avamet_true_{quantity}_c"]).mean())
                                      for quantity in quantities],
        "sampled_minus_true_sd_c": [float((panel[f"avamet_sampled_{quantity}_c"] - panel[f"avamet_true_{quantity}_c"]).std())
                                    for quantity in quantities],
        "mean_reports_per_day": [float(panel.reports_per_day.mean())] * len(quantities)})
    sampling.to_csv(output / "sampling_bias.csv", index=False)

    # Model-level penalty, the unit the global phase will use.
    penalties = []
    for quantity in ("tmax", "tmin", "tmean", "tmidpoint"):
        for position, model in enumerate(MODELS):
            penalties.append({"quantity": quantity, "model": model,
                              "mae_era5_c": float(states[(quantity, "era5")]["point"][position]),
                              "mae_avamet_true_c": float(states[(quantity, "avamet_true")]["point"][position]),
                              "station_penalty_c": float(states[(quantity, "avamet_true")]["point"][position]
                                                         - states[(quantity, "era5")]["point"][position])})
    penalty = pd.DataFrame(penalties)
    roughness = pd.read_csv(ROOT / paths["field_smoothness_results_directory"] / "field_smoothness_summary.csv")
    penalty = penalty.merge(roughness[["source", "roughness_50km_k"]], left_on="model", right_on="source").drop(columns="source")
    penalty["penalty_centred_c"] = penalty.station_penalty_c - penalty.groupby("quantity").station_penalty_c.transform("mean")
    penalty.to_csv(output / "daily_model_penalty.csv", index=False)
    association = {quantity: float(group.roughness_50km_k.corr(group.penalty_centred_c, method="spearman"))
                   for quantity, group in penalty.groupby("quantity")}

    narrow = effects[(effects.block_days == family["bootstrap_block_days"][-1]) & (effects.observed_reference == "avamet_true")]
    verdict = {"pilot_name": cfg["pilot_name"], "analysis": "degradation to daily maxima and minima", **provenance,
               "n_station_days": int(len(panel)), "n_stations": int(panel.station_id.nunique()),
               "n_days": int(panel.valid_day.nunique()),
               "sign_of_reference_effect_preserved": bool((narrow.reference_effect_c < 0).all()),
               "winners_differ_by_reference": {row.quantity: bool(row.era5_winner != row.avamet_winner)
                                               for row in narrow.itertuples()},
               "roughness_penalty_spearman": association,
               "four_sample_sampling_bias_c": sampling.set_index("quantity").sampled_minus_true_mean_c.to_dict(),
               "limits": ["a UTC day is not the local observation window that daily archives use",
                          "extremes sampled four times a day understate the true range, and the size of that "
                          "understatement is reported rather than assumed",
                          "the controlled-smoothing check has not been repeated at daily resolution"],
               "status": "completed"}
    (output / "daily_degradation_summary.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(scores[scores.block_days == family["bootstrap_block_days"][-1]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(effects[effects.block_days == family["bootstrap_block_days"][-1]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(sampling.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(penalty.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(json.dumps({key: verdict[key] for key in ("sign_of_reference_effect_preserved", "winners_differ_by_reference",
                                                    "roughness_penalty_spearman", "four_sample_sampling_bias_c")}, indent=2))


if __name__ == "__main__":
    main()
