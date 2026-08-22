#!/usr/bin/env python3
"""Select every eligible 2020 station and pre-specify spatial strata."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import requests
import yaml
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.ops import transform, unary_union

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "regional_t2m_2020.yaml"


def coastline_geometry(path: Path, cfg: dict) -> tuple[object, str]:
    """Fetch once, then retain the exact coast source and digest locally."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        response = requests.get(cfg["url"], timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Only retain the west-Mediterranean vicinity required for the station set.
    lines = []
    for feature in payload["features"]:
        geometry = feature["geometry"]
        coordinates = geometry["coordinates"]
        sequences = coordinates if geometry["type"] == "MultiLineString" else [coordinates]
        for sequence in sequences:
            local = [(x, y) for x, y in sequence if -3.0 <= x <= 3.0 and 36.0 <= y <= 42.0]
            if len(local) >= 2:
                lines.append(LineString(local))
    if not lines:
        raise RuntimeError("no usable Mediterranean coastline segments were found")
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:25830", always_xy=True).transform
    return transform(to_utm, unary_union(lines)), hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    spatial, paths = cfg["spatial_replication"], cfg["paths"]
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
    selected = stations.loc[stations.exact_utc_samples >= cfg["stations"]["min_exact_utc_samples"]].copy()
    if selected.empty:
        raise RuntimeError("no eligible stations")
    coast_path = ROOT / paths["coastline_geojson"]
    coast, digest = coastline_geometry(coast_path, spatial["coastline"])
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:25830", always_xy=True).transform
    selected["coast_distance_km"] = [
        transform(to_utm, Point(lon, lat)).distance(coast) / 1000
        for lat, lon in zip(selected.latitude, selected.longitude, strict=True)
    ]
    selected["coast_stratum"] = np.where(
        selected.coast_distance_km <= spatial["coast_threshold_km"], "coast_0_25km", "interior_gt25km"
    )
    # Equal-count altitude tertiles are defined before any forecast values are read.
    selected["altitude_stratum"] = pd.qcut(
        selected.altitude_m, q=spatial["altitude_strata"], labels=["altitude_low", "altitude_mid", "altitude_high"]
    ).astype(str)
    selected["selection_method"] = "all_coverage_eligible_pre_forecast"
    selected["selection_period"] = f"{start} to {end}"
    selected = selected.sort_values("station_id").reset_index(drop=True)
    output = ROOT / paths["spatial_stations_csv"]
    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output, index=False)
    summary = {
        "pilot_name": cfg["pilot_name"], "eligible_stations": len(selected), "selected_stations": len(selected),
        "minimum_exact_utc_samples": cfg["stations"]["min_exact_utc_samples"],
        "selection_precedes_forecast_download": True, "coastline": {"source": spatial["coastline"]["url"], "version": spatial["coastline"]["version"], "sha256": digest},
        "coast_threshold_km": spatial["coast_threshold_km"],
        "stratum_counts": {column: selected[column].value_counts().sort_index().to_dict() for column in ("coast_stratum", "altitude_stratum")},
        "altitude_m_breaks": [float(x) for x in np.quantile(selected.altitude_m, [0, 1 / 3, 2 / 3, 1])],
    }
    (ROOT / paths["spatial_station_selection_json"]).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(selected.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
