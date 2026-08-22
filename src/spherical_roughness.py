"""Sparse, geodesically isotropic Gaussian smoothing on a latitude-longitude grid."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class KernelDiagnostics:
    sigma_km: float
    truncation_sigma: float
    targets: int
    sources: int
    nonzero_weights: int
    minimum_neighbors: int
    median_neighbors: float
    maximum_neighbors: int
    maximum_row_sum_error: float
    omitted_radial_mass_upper_bound: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def unit_xyz(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    """Unit-sphere Cartesian coordinates for latitude/longitude degrees."""
    latitude = np.deg2rad(np.asarray(latitude, dtype=np.float64))
    longitude = np.deg2rad(np.asarray(longitude, dtype=np.float64))
    cosine = np.cos(latitude)
    return np.column_stack(
        [cosine * np.cos(longitude), cosine * np.sin(longitude), np.sin(latitude)]
    )


def coordinate_signature(latitudes: np.ndarray, longitudes: np.ndarray) -> str:
    """Order-sensitive digest: matrix columns must match the flattened field."""
    digest = hashlib.sha256()
    for values in (latitudes, longitudes):
        array = np.asarray(values, dtype="<f8")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def target_signature(
    cell_ids: np.ndarray, latitudes: np.ndarray, longitudes: np.ndarray
) -> str:
    digest = hashlib.sha256()
    for cell in np.asarray(cell_ids, dtype=str):
        digest.update(cell.encode("utf-8") + b"\0")
    digest.update(np.asarray(latitudes, dtype="<f8").tobytes())
    digest.update(np.asarray(longitudes, dtype="<f8").tobytes())
    return digest.hexdigest()


def equal_area_cell_centres(grid: object, cell_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Area centroids of cells emitted by :class:`tessellation.EqualAreaGrid`."""
    edges = np.asarray(grid.band_edges, dtype=np.float64)
    slices_per_band = np.asarray(grid.slices_per_band, dtype=np.int64)
    latitudes, longitudes = [], []
    for cell_id in np.asarray(cell_ids, dtype=str):
        band = int(cell_id[1:4])
        slice_index = int(cell_id[5:9])
        south, north = np.deg2rad(edges[band : band + 2])
        latitude = np.rad2deg(np.arcsin((np.sin(south) + np.sin(north)) / 2))
        longitude = (slice_index + 0.5) * 360.0 / slices_per_band[band]
        latitudes.append(latitude)
        longitudes.append((longitude + 180.0) % 360.0 - 180.0)
    return np.asarray(latitudes), np.asarray(longitudes)


def _source_geometry(
    latitudes: np.ndarray, longitudes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    latitude_grid, longitude_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
    xyz = unit_xyz(latitude_grid.ravel(), longitude_grid.ravel())
    # Regular latitude-longitude quadrature: dA is proportional to cos(latitude).
    area = np.maximum(np.cos(np.deg2rad(latitude_grid.ravel())), 0.0)
    return xyz, area


def build_geodesic_gaussian_matrix(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    target_latitudes: np.ndarray,
    target_longitudes: np.ndarray,
    sigma_km: float,
    truncation_sigma: float = 4.0,
    workers: int = -1,
) -> tuple[csr_matrix, KernelDiagnostics]:
    """Normalized Gaussian surface quadrature from a global grid to targets.

    Kernel weights depend only on great-circle distance and grid-cell area.
    Searching in unit-sphere Cartesian space handles the dateline and poles
    without special branches. The 2-D Gaussian radial mass outside 4 sigma is
    exp(-8), which is recorded rather than silently ignored.
    """
    if sigma_km <= 0 or truncation_sigma <= 0:
        raise ValueError("sigma_km and truncation_sigma must be positive")
    latitudes = np.asarray(latitudes, dtype=np.float64)
    longitudes = np.asarray(longitudes, dtype=np.float64)
    target_latitudes = np.asarray(target_latitudes, dtype=np.float64)
    target_longitudes = np.asarray(target_longitudes, dtype=np.float64)
    if latitudes.ndim != 1 or longitudes.ndim != 1:
        raise ValueError("source latitude and longitude coordinates must be one-dimensional")
    if target_latitudes.shape != target_longitudes.shape:
        raise ValueError("target latitude and longitude arrays must have the same shape")

    source_xyz, source_area = _source_geometry(latitudes, longitudes)
    targets_xyz = unit_xyz(target_latitudes, target_longitudes)
    radius_km = sigma_km * truncation_sigma
    angular_radius = radius_km / EARTH_RADIUS_KM
    chord_radius = 2.0 * np.sin(angular_radius / 2.0)
    neighborhoods = cKDTree(source_xyz).query_ball_point(
        targets_xyz, chord_radius, workers=workers, return_sorted=True
    )
    counts = np.fromiter((len(item) for item in neighborhoods), dtype=np.int64)
    if np.any(counts == 0):
        raise RuntimeError("a target has no grid points inside the Gaussian truncation radius")
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    indices = np.empty(indptr[-1], dtype=np.int32)
    data = np.empty(indptr[-1], dtype=np.float64)
    for row, neighbors in enumerate(neighborhoods):
        start, stop = indptr[row], indptr[row + 1]
        columns = np.asarray(neighbors, dtype=np.int64)
        chord = np.linalg.norm(source_xyz[columns] - targets_xyz[row], axis=1)
        angular = 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))
        distance_km = EARTH_RADIUS_KM * angular
        weights = np.exp(-0.5 * np.square(distance_km / sigma_km)) * source_area[columns]
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            raise RuntimeError(f"invalid Gaussian normalization at target {row}")
        indices[start:stop] = columns
        data[start:stop] = weights / total
    matrix = csr_matrix(
        (data, indices, indptr),
        shape=(len(target_latitudes), len(latitudes) * len(longitudes)),
    )
    row_error = np.max(np.abs(np.asarray(matrix.sum(axis=1)).ravel() - 1.0))
    diagnostics = KernelDiagnostics(
        sigma_km=float(sigma_km),
        truncation_sigma=float(truncation_sigma),
        targets=int(matrix.shape[0]),
        sources=int(matrix.shape[1]),
        nonzero_weights=int(matrix.nnz),
        minimum_neighbors=int(counts.min()),
        median_neighbors=float(np.median(counts)),
        maximum_neighbors=int(counts.max()),
        maximum_row_sum_error=float(row_error),
        omitted_radial_mass_upper_bound=float(np.exp(-0.5 * truncation_sigma**2)),
    )
    return matrix, diagnostics


def direct_gaussian_values(
    field: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    target_latitudes: np.ndarray,
    target_longitudes: np.ndarray,
    sigma_km: float,
    truncation_sigma: float = 4.0,
) -> np.ndarray:
    """Brute-force reference implementation used only for numerical tests."""
    source_xyz, source_area = _source_geometry(latitudes, longitudes)
    targets_xyz = unit_xyz(target_latitudes, target_longitudes)
    flattened = np.asarray(field, dtype=np.float64).ravel()
    values = []
    for target in targets_xyz:
        chord = np.linalg.norm(source_xyz - target, axis=1)
        angular = 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))
        distance = EARTH_RADIUS_KM * angular
        inside = distance <= sigma_km * truncation_sigma + 1e-9
        weights = (
            np.exp(-0.5 * np.square(distance[inside] / sigma_km)) * source_area[inside]
        )
        values.append(np.dot(weights, flattened[inside]) / weights.sum())
    return np.asarray(values)


def self_test() -> dict:
    """Small deterministic checks for constants, dateline and brute-force agreement."""
    latitudes = np.arange(-90.0, 90.1, 5.0)
    longitudes = np.arange(0.0, 360.0, 5.0)
    target_latitudes = np.array([0.0, 45.0, 80.0, 0.0])
    target_longitudes = np.array([180.0, 10.0, -170.0, -180.0])
    sigma = 500.0
    matrix, diagnostics = build_geodesic_gaussian_matrix(
        latitudes, longitudes, target_latitudes, target_longitudes, sigma, workers=1
    )
    latitude_grid, longitude_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
    synthetic = (
        np.sin(np.deg2rad(latitude_grid))
        + 0.3 * np.cos(2 * np.deg2rad(longitude_grid))
    )
    sparse_values = np.asarray(matrix @ synthetic.ravel()).ravel()
    direct_values = direct_gaussian_values(
        synthetic,
        latitudes,
        longitudes,
        target_latitudes,
        target_longitudes,
        sigma,
    )
    constant_error = float(np.max(np.abs(np.asarray(matrix @ np.ones(synthetic.size)) - 1)))
    direct_error = float(np.max(np.abs(sparse_values - direct_values)))
    dateline_error = float(abs(sparse_values[0] - sparse_values[3]))
    passed = constant_error < 1e-12 and direct_error < 1e-12 and dateline_error < 1e-12
    return {
        "passed": passed,
        "constant_field_max_abs_error": constant_error,
        "sparse_vs_direct_max_abs_error": direct_error,
        "equivalent_dateline_target_abs_error": dateline_error,
        "kernel": diagnostics.as_dict(),
    }
