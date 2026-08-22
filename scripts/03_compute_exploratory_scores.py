#!/usr/bin/env python3
"""Score the common complete cases from the short pilot; never infer significance."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "interim" / "aligned_t2m_mvp.parquet"
RESULTS = ROOT / "results"
MODELS = ["ifs_hres", "graphcast_hres_init", "pangu_hres_init"]
REFERENCES = {"era5_station": "era5_t2m_c", "avamet": "avamet_t2m_qc_c"}


def main() -> None:
    frame = duckdb.sql(f"SELECT * FROM read_parquet('{INPUT.as_posix()}')").fetchdf()
    required = [*REFERENCES.values(), *(f"{model}_t2m_c" for model in MODELS)]
    common = frame.dropna(subset=required).copy()
    scores = []
    for reference, target_column in REFERENCES.items():
        for model in MODELS:
            error = common[f"{model}_t2m_c"] - common[target_column]
            scores.append(
                {
                    "reference": reference,
                    "model": model,
                    "n_common_cases": len(common),
                    "mae_c": error.abs().mean(),
                    "rmse_c": np.sqrt(np.mean(np.square(error))),
                    "bias_c": error.mean(),
                }
            )
    output = pd.DataFrame(scores)
    output["mae_rank"] = output.groupby("reference")["mae_c"].rank(method="min").astype(int)
    output = output.sort_values(["reference", "mae_rank", "model"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    output.to_csv(RESULTS / "exploratory_scores_t2m_mvp.csv", index=False)
    rankings = output[["reference", "model", "mae_rank", "mae_c"]].copy()
    rankings.to_csv(RESULTS / "exploratory_rankings_t2m_mvp.csv", index=False)
    summary = {
        "n_common_cases": len(common),
        "scope": "exploratory short pilot only; no confidence intervals or GO/NO-GO inference",
        "same_cases_for_both_references": True,
        "rankings_identical": bool(
            rankings.sort_values(["reference", "mae_rank", "model"])
            .groupby("reference")["model"].apply(tuple).nunique() == 1
        ),
    }
    (RESULTS / "exploratory_scores_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(output.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
