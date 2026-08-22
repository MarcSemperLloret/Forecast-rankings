#!/usr/bin/env python3
"""Inventory regional reference-effect sensitivity specifications."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "regional_robustness_inventory_2020"
SOURCES = (
    ("lead time", ROOT / "results/lead_time_t2m_2020/hres_multilead_robustness.csv",
     "primary_effect_c", "primary_effect_ci_low_c", "primary_effect_ci_high_c"),
    ("spatial strata", ROOT / "results/spatial_t2m_2020/spatial_robustness.csv",
     "primary_effect_c", "primary_effect_ci_low_c", "primary_effect_ci_high_c"),
    ("representativeness", ROOT / "results/representativeness_2020/representativeness_robustness.csv",
     "primary_effect_c", "primary_effect_ci_low_c", "primary_effect_ci_high_c"),
    ("orography", ROOT / "results/orography_sensitivity_2020/orography_robustness.csv",
     "primary_effect_c", "primary_effect_ci_low_c", "primary_effect_ci_high_c"),
    ("metric and uncertainty", ROOT / "results/metric_and_spatial_uncertainty_2020/metric_robustness.csv",
     "primary_effect", "primary_effect_ci_low", "primary_effect_ci_high"),
)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    inventory: list[dict[str, object]] = []
    exceptions: list[pd.DataFrame] = []
    for design, path, effect_col, low_col, high_col in SOURCES:
        frame = pd.read_csv(path)
        comparable = frame[frame[effect_col].notna()].copy()
        comparable["effect_negative"] = comparable[effect_col] < 0
        comparable["ci_excludes_zero"] = (
            (comparable[low_col] > 0) | (comparable[high_col] < 0)
        )
        inventory.append({
            "design": design,
            "source": path.relative_to(ROOT).as_posix(),
            "n_specifications": int(len(comparable)),
            "n_effect_negative": int(comparable.effect_negative.sum()),
            "n_ci_excludes_zero": int(comparable.ci_excludes_zero.sum()),
        })
        failed = comparable[~comparable.ci_excludes_zero].copy()
        if not failed.empty:
            failed.insert(0, "design", design)
            failed.insert(1, "source", path.relative_to(ROOT).as_posix())
            exceptions.append(failed)

    inventory_frame = pd.DataFrame(inventory)
    totals = {
        "design": "TOTAL",
        "source": "five frozen regional sensitivity tables",
        "n_specifications": int(inventory_frame.n_specifications.sum()),
        "n_effect_negative": int(inventory_frame.n_effect_negative.sum()),
        "n_ci_excludes_zero": int(inventory_frame.n_ci_excludes_zero.sum()),
    }
    inventory_frame = pd.concat([inventory_frame, pd.DataFrame([totals])], ignore_index=True)
    inventory_frame.to_csv(OUTPUT / "configuration_inventory.csv", index=False)
    exception_frame = pd.concat(exceptions, ignore_index=True) if exceptions else pd.DataFrame()
    exception_frame.to_csv(OUTPUT / "ci_exceptions.csv", index=False)
    summary = {
        "definition": (
            "One specification is one non-missing comparative reference-effect row in the five "
            "frozen regional sensitivity tables; diagnostic signed-bias rows are excluded."
        ),
        "n_specifications": totals["n_specifications"],
        "n_effect_negative": totals["n_effect_negative"],
        "n_ci_excludes_zero": totals["n_ci_excludes_zero"],
        "n_ci_contains_zero": int(len(exception_frame)),
        "passed_sign_gate": totals["n_specifications"] == totals["n_effect_negative"],
        "all_intervals_exclude_zero": totals["n_specifications"] == totals["n_ci_excludes_zero"],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
