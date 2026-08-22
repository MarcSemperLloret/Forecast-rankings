"""An approximately equal-area tessellation of the sphere, built from scratch.

The global result must not be dominated by wherever the network happens to be
dense, so stations are aggregated into cells first and the cells are weighted
equally. That only works if the cells cover comparable areas; a plain
latitude-longitude grid does not, since its cells shrink towards the poles by
cos(latitude).

The construction is the classic one and needs no dependency beyond numpy:

* the sphere is cut into latitude bands whose height is the target cell size;
* each band is cut into as many equal longitude slices as its own area allows.

Because the number of slices has to be an integer, cell areas are close to the
target rather than exactly equal, so ``summary`` reports the real dispersion
instead of leaving "approximately" as a claim.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_KM = 6371.0088
DEGREE_KM = 2 * np.pi * EARTH_RADIUS_KM / 360


@dataclass(frozen=True)
class EqualAreaGrid:
    """Cells of approximately ``target_cell_km`` on a side, from pole to pole."""

    target_cell_km: float

    @property
    def target_area_km2(self) -> float:
        return float(self.target_cell_km**2)

    @property
    def band_edges(self) -> np.ndarray:
        bands = max(int(round(180.0 / (self.target_cell_km / DEGREE_KM))), 2)
        return np.linspace(-90.0, 90.0, bands + 1)

    @property
    def slices_per_band(self) -> np.ndarray:
        edges = np.radians(self.band_edges)
        areas = 2 * np.pi * EARTH_RADIUS_KM**2 * np.abs(np.diff(np.sin(edges)))
        return np.maximum(np.round(areas / self.target_area_km2).astype(int), 1)

    @property
    def cell_areas_km2(self) -> np.ndarray:
        edges = np.radians(self.band_edges)
        areas = 2 * np.pi * EARTH_RADIUS_KM**2 * np.abs(np.diff(np.sin(edges)))
        return areas / self.slices_per_band

    def assign(self, latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
        """Cell identifier for each point, as ``band_slice`` strings."""
        latitude = np.asarray(latitude, dtype=float)
        longitude = np.mod(np.asarray(longitude, dtype=float), 360.0)
        edges = self.band_edges
        band = np.clip(np.searchsorted(edges, latitude, side="right") - 1, 0, len(edges) - 2)
        slices = self.slices_per_band[band]
        index = np.clip((longitude / 360.0 * slices).astype(int), 0, slices - 1)
        return np.array([f"b{one:03d}s{other:04d}" for one, other in zip(band, index)])

    def area_of(self, cell_ids: np.ndarray) -> np.ndarray:
        bands = np.array([int(cell[1:4]) for cell in np.atleast_1d(cell_ids)])
        return self.cell_areas_km2[bands]

    def summary(self) -> dict:
        areas = self.cell_areas_km2
        weights = self.slices_per_band.astype(float)
        mean = float(np.average(areas, weights=weights))
        spread = float(np.sqrt(np.average((areas - mean) ** 2, weights=weights)))
        return {"target_cell_km": self.target_cell_km, "target_area_km2": self.target_area_km2,
                "bands": int(len(areas)), "cells": int(self.slices_per_band.sum()),
                "cell_area_mean_km2": mean, "cell_area_sd_km2": spread,
                "cell_area_min_km2": float(areas.min()), "cell_area_max_km2": float(areas.max()),
                "relative_sd": spread / mean,
                "note": "areas differ from the target only because the slice count per band is an integer"}
