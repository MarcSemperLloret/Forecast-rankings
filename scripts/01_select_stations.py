#!/usr/bin/env python3
"""Select the 10-station MVP without looking at forecast or score data."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "mvp_t2m.yaml"


def read_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def select_maximin(candidates: pd.DataFrame, n: int) -> pd.DataFrame:
    """Deterministic maximin selection after prioritising exact-time coverage."""
    candidates = candidates.sort_values(
        ["exact_utc_samples", "station_id"], ascending=[False, True]
    ).reset_index(drop=True)
    columns = ["latitude", "longitude", "altitude_m"]
    features = candidates[columns].to_numpy(dtype=float)
    features = (features - features.mean(axis=0)) / features.std(axis=0, ddof=0)

    selected = [0]
    remaining = set(range(1, len(candidates)))
    while len(selected) < n:
        distances = np.linalg.norm(
            features[list(remaining), None, :] - features[selected][None, :, :], axis=2
        )
        min_distance = distances.min(axis=1)
        choices = list(remaining)
        # Stable lexical tie-breaking prevents hidden dependence on hash order.
        best = max(
            zip(min_distance, choices),
            key=lambda item: (item[0], -item[1]),
        )[1]
        selected.append(best)
        remaining.remove(best)
    return candidates.iloc[selected].copy()


def main() -> None:
    cfg = read_config()
    archive = (ROOT / cfg["avamet"]["archive_glob"]).resolve().as_posix()
    min_samples = cfg["stations"]["min_exact_utc_samples"]
    n = cfg["stations"]["n"]
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    stations = con.execute(
        f"""
        SELECT
            station_id,
            first(latitude) AS latitude,
            first(longitude) AS longitude,
            first(altitude_m) AS altitude_m,
            count(DISTINCT observed_utc) FILTER (
                WHERE hour(observed_utc) IN (0, 12) AND minute(observed_utc) = 0
            ) AS exact_utc_samples
        FROM read_parquet('{archive}', hive_partitioning=true)
        WHERE temperature_c IS NOT NULL
        GROUP BY station_id
        HAVING latitude IS NOT NULL AND longitude IS NOT NULL AND altitude_m IS NOT NULL
        """
    ).fetchdf()
    eligible = stations.loc[stations.exact_utc_samples >= min_samples].copy()
    if len(eligible) < n:
        raise RuntimeError(f"only {len(eligible)} stations pass coverage; need {n}")

    selected = select_maximin(eligible, n).sort_values("station_id").reset_index(drop=True)
    selected["selection_method"] = "coverage_then_maximin_lat_lon_alt"
    selected["selection_seed"] = cfg["stations"]["selection_seed"]
    output = ROOT / "data" / "processed"
    output.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output / "stations_mvp.csv", index=False)
    (output / "station_selection_summary.json").write_text(
        json.dumps(
            {
                "eligible_stations": len(eligible),
                "minimum_exact_utc_samples": min_samples,
                "selected_stations": selected.station_id.tolist(),
                "method": "coverage_then_maximin_lat_lon_alt",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
