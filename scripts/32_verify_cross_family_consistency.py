#!/usr/bin/env python3
"""Check that the families agree on the effect they share.

Five families compute the reference effect on the same 146-station panel at the
same lead by different routes: whole-panel scoring, the representativeness
station panel, the leave-one-region-out baseline, the all-days regime baseline
and the both-cycles diurnal baseline. They must agree. A disagreement beyond
bootstrap noise means one of them filters or weights the panel differently, so
this runs as a check with a threshold rather than as a printed coincidence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "regional_t2m_2020.yaml"
TOLERANCE_C = 0.005


def read(path: Path, query: str, column: str) -> float | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path).query(query)
    if frame.empty:
        return None
    if len(frame) != 1:
        raise RuntimeError(f"{path.name}: expected one row, found {len(frame)} for {query!r}")
    return float(frame[column].iloc[0])


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    paths = cfg["paths"]
    lead, width = 24, 1
    sources = {
        "spatial_replication": (ROOT / paths["spatial_results_directory"] / "spatial_robustness.csv",
                                f"lead_h == {lead} and stratum_type == 'all_eligible' and block_days == {width}", "primary_effect_c"),
        "representativeness_station_bilinear": (ROOT / paths["representativeness_results_directory"] / "representativeness_robustness.csv",
                                                f"lead_h == {lead} and unit == 'station' and extraction == 'bilinear' and stratum == 'all_eligible' and block_days == {width}", "primary_effect_c"),
        "leave_one_region_out_baseline": (ROOT / paths["leave_one_region_out_results_directory"] / "leave_one_region_out_folds.csv",
                                          f"lead_h == {lead} and fold_type == 'all_stations' and partition == 'kmeans_10' and block_days == {width}", "primary_effect_c"),
        "day_regimes_all_days": (ROOT / paths["day_regime_results_directory"] / "regime_strata.csv",
                                 f"lead_h == {lead} and regime == 'all_days' and block_days == {width}", "primary_effect_c"),
        "diurnal_both_cycles": (ROOT / paths["diurnal_cycle_results_directory"] / "diurnal_cycle_panels.csv",
                                f"lead_h == {lead} and init_cycle_utc == -1 and block_days == {width}", "primary_effect_c"),
    }
    rows = []
    for family, (path, query, column) in sources.items():
        value = read(path, query, column)
        rows.append({"family": family, "lead_h": lead, "block_days": width, "primary_effect_c": value,
                     "source": str(path.relative_to(ROOT)) if path.exists() else "missing"})
    frame = pd.DataFrame(rows)
    found = frame.dropna(subset=["primary_effect_c"])
    if len(found) < 2:
        raise RuntimeError("fewer than two families produced the shared effect; nothing to verify")
    spread = float(found.primary_effect_c.max() - found.primary_effect_c.min())
    frame["deviation_from_median_c"] = frame.primary_effect_c - found.primary_effect_c.median()

    output = ROOT / "results" / "cross_family_consistency.csv"
    frame.to_csv(output, index=False)
    verdict = {"lead_hours": lead, "block_days": width, "families": len(found), "spread_c": spread,
               "tolerance_c": TOLERANCE_C, "consistent": bool(spread <= TOLERANCE_C),
               "note": "point estimate versus bootstrap mean explains a spread of this order",
               "missing": frame[frame.primary_effect_c.isna()].family.tolist()}
    (ROOT / "results" / "cross_family_consistency.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(json.dumps(verdict, indent=2))
    if not verdict["consistent"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
