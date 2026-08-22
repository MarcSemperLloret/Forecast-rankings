#!/usr/bin/env python3
"""Collapse records that describe the same physical site into one site.

GHCN, GSOD and ECA&D overlap by construction, and national services feed all
three, so one site can appear four times. Left alone it would contribute four
times to a cell average and count as four stations towards a support radius.

Records are never deleted. They are grouped into sites, and the site is what
every later step counts. A site keeps the metadata of its highest-priority
member and the union of its networks, so that a site is flagged as potentially
assimilated when any of its members is.

Two records join the same site when they share an identifier, or when they are
within the merge radius **and** their elevations agree within the tolerance. A
pair closer than the radius but disagreeing in elevation is reported and left
apart: a valley floor and a slope 300 m away are different sites, and merging
them would destroy the subgrid structure this work is about.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from validation import haversine_km

CONFIG = ROOT / "config" / "regional_t2m_2020.yaml"


def connected_components(adjacency: np.ndarray) -> np.ndarray:
    """Site label for each record, by transitive closure of the join relation."""
    labels = np.full(len(adjacency), -1)
    site = 0
    for start in range(len(adjacency)):
        if labels[start] != -1:
            continue
        stack, members = [start], []
        while stack:
            current = stack.pop()
            if labels[current] != -1:
                continue
            labels[current] = site
            members.append(current)
            stack.extend(np.flatnonzero(adjacency[current] & (labels == -1)).tolist())
        site += 1
    return labels


def build_sites(stations: pd.DataFrame, policy: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    coordinates = stations[["latitude", "longitude"]].to_numpy(float)
    distance = haversine_km(coordinates, coordinates)
    np.fill_diagonal(distance, np.inf)
    close = distance <= policy["merge_radius_km"]

    if "altitude_m" in stations:
        elevation = stations.altitude_m.to_numpy(float)
        gap = np.abs(elevation[:, None] - elevation[None, :])
        # A missing elevation cannot contradict, so it does not block a merge.
        agrees = ~np.isfinite(gap) | (gap <= policy["merge_elevation_tolerance_m"])
    else:
        gap = np.full_like(distance, np.nan)
        agrees = np.ones_like(close)

    join = close & agrees
    if policy.get("identifier_match_wins") and "site_identifier" in stations:
        identifier = stations.site_identifier.fillna("")
        same = (identifier.to_numpy()[:, None] == identifier.to_numpy()[None, :]) & (identifier.to_numpy()[:, None] != "")
        np.fill_diagonal(same, False)
        join = join | same

    labels = connected_components(join)
    stations = stations.assign(site_id=[f"site_{label:06d}" for label in labels])

    priority = {name: rank for rank, name in enumerate(policy["network_priority"])}
    assimilated = set(policy["assimilated_networks"])
    networks = stations.network if "network" in stations else pd.Series(["unknown"] * len(stations), index=stations.index)
    stations = stations.assign(_rank=[priority.get(str(name), len(priority)) for name in networks],
                               _network=networks.to_numpy())
    ordered = stations.sort_values(["site_id", "_rank", "station_id"])
    representative = ordered.groupby("site_id", as_index=False).first()
    grouped = ordered.groupby("site_id")
    sites = representative.assign(
        n_members=grouped.station_id.size().to_numpy(),
        members="|".join if False else grouped.station_id.apply(lambda values: "|".join(values)).to_numpy(),
        networks=grouped._network.apply(lambda values: "|".join(sorted(set(values)))).to_numpy(),
        mean_latitude=grouped.latitude.mean().to_numpy(),
        mean_longitude=grouped.longitude.mean().to_numpy())
    sites["potentially_assimilated"] = [any(name in assimilated for name in value.split("|")) for value in sites.networks]
    sites = sites.drop(columns=["_rank", "_network"])

    rejected = []
    near = np.argwhere(close & ~agrees)
    for first, second in near:
        if first >= second:
            continue
        rejected.append({"station_a": stations.iloc[first].station_id, "station_b": stations.iloc[second].station_id,
                         "distance_km": float(distance[first, second]), "elevation_gap_m": float(gap[first, second]),
                         "decision": "kept apart: within the radius but the elevations disagree"})
    return sites, pd.DataFrame(rejected, columns=["station_a", "station_b", "distance_km", "elevation_gap_m", "decision"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    policy = cfg["station_deduplication"]
    header = pd.read_csv(args.stations, nrows=0)
    string_columns = {
        name: "string" for name in ("station_id", "site_identifier", "site_id")
        if name in header.columns
    }
    stations = pd.read_csv(args.stations, dtype=string_columns)
    sites, rejected = build_sites(stations, policy)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sites.to_csv(args.output, index=False)
    rejected.to_csv(args.output.with_name(args.output.stem + "_kept_apart.csv"), index=False)
    collapsed = sites[sites.n_members > 1]
    report = {"source": str(args.stations), "policy": policy, "records": int(len(stations)),
              "sites": int(len(sites)), "collapsed_sites": int(len(collapsed)),
              "records_absorbed": int(len(stations) - len(sites)),
              "largest_site_members": int(sites.n_members.max()),
              "pairs_kept_apart_on_elevation": int(len(rejected)),
              "potentially_assimilated_sites": int(sites.potentially_assimilated.sum())}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not collapsed.empty:
        print("\nsitios colapsados:")
        print(collapsed[["site_id", "station_id", "n_members", "members", "networks"]].to_string(index=False))
    if not rejected.empty:
        print("\npares cercanos mantenidos aparte por altitud:")
        print(rejected.to_string(index=False))


if __name__ == "__main__":
    main()
