#!/usr/bin/env python3
"""Separate the initialisation cycle from the hour the forecast is valid at.

At any lead that is a multiple of 24 h the two are the same thing: the 00 UTC
run is always verified at 00 UTC. Splitting such a panel by cycle answers both
questions at once and neither separately, so this script reports that split as
what it is and then closes the design with lead 36 h, which swaps the pairing.

Lead 36 h sits between the two bracket leads already materialised. The pure
lead dependence is read off the bracket, and the departure of lead 36 h from it
is attributed to the valid hour. That attribution has a falsifiable signature:
the departure must change sign between the two cycles, because the two cycles
swap their valid hours at 36 h. A departure with the same sign in both cycles
would be a lead effect the bracket failed to capture, not a diurnal one.
"""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

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


def analyse(frame: pd.DataFrame, cfg: dict, seed_offset: int) -> tuple[list[dict], dict[int, np.ndarray]]:
    """Score one panel and return both the summary rows and the effect draws."""
    daily = {reference: daily_errors(frame, reference) for reference in REFERENCES}
    if not daily["era5"].valid_day.equals(daily["avamet"].valid_day):
        raise RuntimeError("references do not share the same valid-day blocks")
    bootstrap, family = cfg["bootstrap"], cfg["diurnal_cycle_extension"]
    first, second = (MODELS.index(model) for model in family["primary_pair"])
    rows, effect_draws = [], {}
    for width in family["bootstrap_block_days"]:
        rng = np.random.default_rng(bootstrap["seed"] + seed_offset + width)
        indices = block_indices(rng, len(daily["avamet"]), bootstrap["n_resamples"], width)
        draws = {reference: values[list(MODELS)].to_numpy()[indices].mean(axis=1) for reference, values in daily.items()}
        winners = {reference: np.argmin(values, axis=1) for reference, values in draws.items()}
        era = draws["era5"][:, first] - draws["era5"][:, second]
        avamet = draws["avamet"][:, first] - draws["avamet"][:, second]
        switch = avamet - era
        effect_draws[width] = switch
        low, high = interval(switch, bootstrap["ci"])
        point = {reference: values[list(MODELS)].mean().to_numpy() for reference, values in daily.items()}
        rows.append({"block_days": width, "n_common_cases": len(frame), "n_stations": frame.station_id.nunique(),
                     "n_daily_blocks": len(daily["avamet"]),
                     "avamet_winner": MODELS[int(np.argmin(point["avamet"]))], "era5_winner": MODELS[int(np.argmin(point["era5"]))],
                     "p_winners_differ": float((winners["avamet"] != winners["era5"]).mean()),
                     "primary_effect_c": switch.mean(), "primary_effect_ci_low_c": low, "primary_effect_ci_high_c": high,
                     "primary_pair_reversal_probability": float((era * avamet < 0).mean())})
    return rows, effect_draws


def daily_effect(frame: pd.DataFrame, pair: tuple[str, str]) -> pd.Series:
    """The effect of each valid day, as a series indexed by date."""
    first, second = pair
    observed, reanalysis = frame.avamet_t2m_qc_c, frame.era5_t2m_c
    gap_avamet = (frame[f"{first}_t2m_c"] - observed).abs() - (frame[f"{second}_t2m_c"] - observed).abs()
    gap_era5 = (frame[f"{first}_t2m_c"] - reanalysis).abs() - (frame[f"{second}_t2m_c"] - reanalysis).abs()
    values = pd.DataFrame({"valid_day": pd.to_datetime(frame.valid_time).dt.date, "effect": gap_avamet - gap_era5})
    return values.groupby("valid_day").effect.mean().sort_index()


def valid_hour_contrast(frame: pd.DataFrame, lead: int, cfg: dict) -> list[dict]:
    """Compare the two valid hours of one lead, paired on the calendar day.

    This is the discriminator that survives without assuming anything about how
    the effect varies with lead. At leads 24 and 48 the panel verified at 12 UTC
    is the 12 UTC run; at lead 36 it is the 00 UTC run. If the ordering between
    valid hours is the same at all three leads while the ordering between cycles
    flips, the valid hour is what matters and the cycle is not.
    """
    family, bootstrap = cfg["diurnal_cycle_extension"], cfg["bootstrap"]
    pair = tuple(family["primary_pair"])
    series = {}
    for cycle in family["cycles_utc"]:
        subset = frame[frame.init_cycle_utc == cycle]
        series[int((cycle + lead) % 24)] = daily_effect(subset, pair)
    if set(series) != {0, 12}:
        raise RuntimeError(f"lead={lead} does not produce both valid hours: {sorted(series)}")
    paired = pd.concat({hour: values for hour, values in series.items()}, axis=1, join="inner")
    difference = (paired[12] - paired[0]).to_numpy()
    rows = []
    for width in family["bootstrap_block_days"]:
        rng = np.random.default_rng(bootstrap["seed"] + lead * 7 + width)
        indices = block_indices(rng, len(difference), bootstrap["n_resamples"], width)
        draws = difference[indices].mean(axis=1)
        low, high = interval(draws, bootstrap["ci"])
        rows.append({"lead_h": lead, "block_days": width, "n_paired_days": len(difference),
                     "cycle_valid_at_00utc": int((0 - lead) % 24), "cycle_valid_at_12utc": int((12 - lead) % 24),
                     "effect_valid_00utc_c": float(paired[0].mean()), "effect_valid_12utc_c": float(paired[12].mean()),
                     "contrast_12_minus_00_c": float(draws.mean()), "contrast_ci_low_c": low, "contrast_ci_high_c": high,
                     "p_contrast_lt_zero": float((draws < 0).mean())})
    return rows


def load_panel(cfg: dict, lead: int) -> pd.DataFrame | None:
    source = ROOT / cfg["paths"]["spatial_directory"] / f"lead={lead:03d}" / "batches"
    if not list(source.glob("*.parquet")):
        return None
    frame = duckdb.sql(f"SELECT * FROM read_parquet('{(source / '*.parquet').as_posix()}')").fetchdf()
    frame = frame.dropna(subset=[*REFERENCES.values(), *(f"{model}_t2m_c" for model in MODELS)])
    frame["init_cycle_utc"] = pd.to_datetime(frame.init_time).dt.hour
    frame["valid_hour_utc"] = pd.to_datetime(frame.valid_time).dt.hour
    return frame


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    family, paths = cfg["diurnal_cycle_extension"], cfg["paths"]
    output = ROOT / paths["diurnal_cycle_results_directory"]
    output.mkdir(parents=True, exist_ok=True)
    crossed, bracket = family["crossed_lead_hours"], family["bracket_leads_hours"]
    leads = [*bracket, crossed]

    rows, draws, contrasts = [], {}, []
    for lead in sorted(leads):
        frame = load_panel(cfg, lead)
        if frame is None:
            print(f"lead={lead} is not materialised yet; skipping", flush=True)
            continue
        aliased = lead % 24 == 0
        contrasts.extend(valid_hour_contrast(frame, lead, cfg))
        for cycle in family["cycles_utc"]:
            subset = frame[frame.init_cycle_utc == cycle]
            valid_hours = sorted(subset.valid_hour_utc.unique())
            if len(valid_hours) != 1:
                raise RuntimeError(f"lead={lead} cycle={cycle} spans several valid hours: {valid_hours}")
            scored, effects = analyse(subset, cfg, seed_offset=lead * 100 + cycle)
            draws[(lead, cycle)] = effects
            for row in scored:
                rows.append({"lead_h": lead, "init_cycle_utc": cycle, "valid_hour_utc": int(valid_hours[0]),
                             "cycle_aliased_with_valid_hour": aliased, "panel": f"lead{lead:03d}_cycle{cycle:02d}", **row})
        scored, _ = analyse(frame, cfg, seed_offset=lead * 100 + 99)
        for row in scored:
            rows.append({"lead_h": lead, "init_cycle_utc": -1, "valid_hour_utc": -1,
                         "cycle_aliased_with_valid_hour": aliased, "panel": f"lead{lead:03d}_both_cycles", **row})
        print(f"analysed diurnal lead={lead}; cases={len(frame)}", flush=True)

    panels = pd.DataFrame(rows).sort_values(["lead_h", "init_cycle_utc", "block_days"])
    panels.to_csv(output / "diurnal_cycle_panels.csv", index=False)
    contrast = pd.DataFrame(contrasts).sort_values(["lead_h", "block_days"])
    contrast.to_csv(output / "valid_hour_contrasts.csv", index=False)
    narrow = contrast[contrast.block_days == family["bootstrap_block_days"][0]]
    same_ordering = bool((narrow.contrast_12_minus_00_c < 0).all() or (narrow.contrast_12_minus_00_c > 0).all())

    departures = []
    if all((crossed, cycle) in draws for cycle in family["cycles_utc"]) and all((lead, cycle) in draws for lead in bracket for cycle in family["cycles_utc"]):
        confidence = cfg["bootstrap"]["ci"]
        for cycle in family["cycles_utc"]:
            for width in family["bootstrap_block_days"]:
                expected = np.mean([draws[(lead, cycle)][width] for lead in bracket], axis=0)
                departure = draws[(crossed, cycle)][width] - expected
                low, high = interval(departure, confidence)
                departures.append({"init_cycle_utc": cycle, "valid_hour_utc": int((cycle + crossed) % 24), "block_days": width,
                                   "crossed_lead_h": crossed, "bracket_leads_h": "+".join(str(lead) for lead in bracket),
                                   "crossed_effect_c": float(draws[(crossed, cycle)][width].mean()),
                                   "bracket_expected_effect_c": float(expected.mean()),
                                   "departure_c": float(departure.mean()), "departure_ci_low_c": low, "departure_ci_high_c": high,
                                   "p_departure_gt_zero": float((departure > 0).mean())})
        frame = pd.DataFrame(departures)
        frame.to_csv(output / "diurnal_departures.csv", index=False)
        signs = frame.groupby("block_days").departure_c.apply(lambda values: float(np.sign(values).nunique() == 2))
        verdict = {"departures_change_sign_between_cycles": {int(width): bool(value) for width, value in signs.items()}}
    else:
        verdict = {"departures_change_sign_between_cycles": "lead 36 h is not materialised yet"}

    (output / "diurnal_cycle_summary.json").write_text(json.dumps(
        {"pilot_name": cfg["pilot_name"], "analysis": "initialisation cycle versus valid hour",
         "aliasing": "at leads that are multiples of 24 h the cycle and the valid hour are the same split",
         "crossed_lead_hours": crossed, "bracket_leads_hours": bracket, "primary_pair": family["primary_pair"],
         "attribution_test": "the lead-36 h departure from the bracket must change sign between cycles to be diurnal",
         "valid_hour_ordering_identical_at_every_lead": same_ordering,
         "valid_hour_note": "the model-free discriminator: the cycle behind each valid hour swaps at lead 36 h, so an ordering that holds at all three leads belongs to the valid hour",
         **verdict,
         "bootstrap": {"method": "circular moving-block bootstrap on ordered valid days",
                       "n_resamples": cfg["bootstrap"]["n_resamples"], "block_days": family["bootstrap_block_days"],
                       "ci": cfg["bootstrap"]["ci"], "seed": cfg["bootstrap"]["seed"]},
         "status": "completed" if departures else "partial; awaiting lead 36 h"}, indent=2), encoding="utf-8")

    view = panels[panels.block_days == family["bootstrap_block_days"][0]]
    print(view[["lead_h", "init_cycle_utc", "valid_hour_utc", "n_common_cases", "primary_effect_c",
                "primary_effect_ci_low_c", "primary_effect_ci_high_c", "avamet_winner", "era5_winner",
                "p_winners_differ"]].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(narrow.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    if departures:
        print(pd.DataFrame(departures).to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
