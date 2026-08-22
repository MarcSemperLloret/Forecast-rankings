#!/usr/bin/env python3
"""Pre-select the 2020 regional AVAMET station network without model data."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "regional_t2m_2020.yaml"


def select_maximin(candidates: pd.DataFrame, n: int) -> pd.DataFrame:
    """Prioritise coverage, then spread selected sites over lat/lon/altitude."""
    candidates = candidates.sort_values(["exact_utc_samples", "station_id"], ascending=[False, True]).reset_index(drop=True)
    features = candidates[["latitude", "longitude", "altitude_m"]].to_numpy(float)
    features = (features - features.mean(axis=0)) / features.std(axis=0, ddof=0)
    selected, remaining = [0], set(range(1, len(candidates)))
    while len(selected) < n:
        choices = sorted(remaining)
        distances = np.linalg.norm(features[choices, None, :] - features[selected][None, :, :], axis=2).min(axis=1)
        best = max(zip(distances, choices, strict=True), key=lambda value: (value[0], -value[1]))[1]
        selected.append(best)
        remaining.remove(best)
    return candidates.iloc[selected].copy()


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    archive = (ROOT / cfg["avamet"]["archive_glob"]).resolve().as_posix()
    start = cfg["period"]["init_start"]
    end = (pd.Timestamp(cfg["period"]["init_end"]) + pd.Timedelta(hours=24)).isoformat()
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    stations = con.execute(
        f"""SELECT station_id, first(latitude) AS latitude, first(longitude) AS longitude,
                    first(altitude_m) AS altitude_m,
                    count(DISTINCT observed_utc) FILTER (WHERE hour(observed_utc) IN (0, 12) AND minute(observed_utc) = 0) AS exact_utc_samples
             FROM read_parquet('{archive}', hive_partitioning=true)
             WHERE temperature_c IS NOT NULL
               AND observed_utc >= TIMESTAMPTZ '{start}+00:00'
               AND observed_utc <= TIMESTAMPTZ '{end}+00:00'
             GROUP BY station_id
             HAVING latitude IS NOT NULL AND longitude IS NOT NULL AND altitude_m IS NOT NULL"""
    ).fetchdf()
    eligible = stations.loc[stations.exact_utc_samples >= cfg["stations"]["min_exact_utc_samples"]].copy()
    n = cfg["stations"]["n"]
    if len(eligible) < n:
        raise RuntimeError(f"only {len(eligible)} eligible stations; need {n}")
    selected = select_maximin(eligible, n).sort_values("station_id").reset_index(drop=True)
    selected["selection_method"] = cfg["stations"]["selection_method"]
    selected["selection_seed"] = cfg["stations"]["selection_seed"]
    selected["selection_period"] = f"{start} to {end}"
    output = ROOT / cfg["paths"]["stations_csv"]
    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output, index=False)
    summary = {
        "pilot_name": cfg["pilot_name"], "eligible_stations": len(eligible), "selected_stations": selected.station_id.tolist(),
        "n_selected": len(selected), "minimum_exact_utc_samples": cfg["stations"]["min_exact_utc_samples"],
        "method": cfg["stations"]["selection_method"], "selection_precedes_forecast_download": True,
    }
    (ROOT / cfg["paths"]["station_selection_json"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(selected.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
