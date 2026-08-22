#!/usr/bin/env python3
"""Assign sites to equal-area cells and report what the weighting will do.

The cells are the unit the global result is averaged over, so this runs before
any confirmatory analysis and its output is auditable on its own: how many cells
the network occupies, how unevenly the sites fall inside them, and how much the
equal-cell weighting differs from weighting every station the same.

That last number is the point. If the two weightings agree, the choice does not
matter and it can be said so. If they diverge, the divergence is the density of
the network speaking, and the pre-registered choice is the one that answers it.
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
from tessellation import EqualAreaGrid

CONFIG = ROOT / "config" / "regional_t2m_2020.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sites", type=Path, help="site table with station_id, latitude, longitude")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latitude-column", default="latitude")
    parser.add_argument("--longitude-column", default="longitude")
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    policy = cfg["spatial_weighting"]
    if policy["tessellation"] != "equal_area_latitude_bands":
        raise RuntimeError(f"unsupported tessellation: {policy['tessellation']}")

    grid = EqualAreaGrid(float(policy["target_cell_km"]))
    sites = pd.read_csv(args.sites)
    sites["cell_id"] = grid.assign(sites[args.latitude_column].to_numpy(), sites[args.longitude_column].to_numpy())
    sites["cell_area_km2"] = grid.area_of(sites.cell_id.to_numpy())
    occupancy = sites.groupby("cell_id", as_index=False).agg(
        n_sites=("station_id", "size"), cell_area_km2=("cell_area_km2", "first"),
        mean_latitude=(args.latitude_column, "mean"), mean_longitude=(args.longitude_column, "mean"))
    eligible = occupancy[occupancy.n_sites >= policy["min_sites_per_cell"]]

    # How far the two weightings can pull apart, measured on this network alone:
    # under equal-cell weighting a site in a crowded cell carries 1/n of a cell,
    # while under equal-station weighting it carries a full station.
    weight_per_site = 1.0 / sites.groupby("cell_id").station_id.transform("size")
    weight_per_site = weight_per_site / weight_per_site.sum()
    flat = np.full(len(sites), 1.0 / len(sites))
    divergence = float(np.abs(weight_per_site.to_numpy() - flat).sum() / 2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sites.to_csv(args.output, index=False)
    occupancy.to_csv(args.output.with_name(args.output.stem + "_cells.csv"), index=False)
    report = {"source": str(args.sites), "policy": policy, "grid": grid.summary(),
              "sites": int(len(sites)), "occupied_cells": int(len(occupancy)),
              "eligible_cells": int(len(eligible)),
              "sites_per_occupied_cell": {"mean": float(occupancy.n_sites.mean()),
                                          "max": int(occupancy.n_sites.max()),
                                          "cells_with_one_site": int((occupancy.n_sites == 1).sum())},
              "total_variation_between_weightings": divergence,
              "interpretation": "half the sum of absolute weight differences; 0 means the two weightings agree, "
                                "1 means they share no mass"}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("\nocupación por celda:")
    print(occupancy.sort_values("n_sites", ascending=False).head(12).to_string(index=False,
                                                                              float_format=lambda value: f"{value:.2f}"))


if __name__ == "__main__":
    main()
