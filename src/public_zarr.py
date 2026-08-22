"""Minimal reader for the public consolidated WeatherBench 2 Zarr-v2 stores."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import numcodecs
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class PublicZarr:
    """Read exactly the Zarr chunks needed by this pilot via anonymous HTTPS."""

    def __init__(self, path: str, cache_root: Path, timeout_seconds: int = 45) -> None:
        self.url = path.replace(
            "gs://weatherbench2/", "https://storage.googleapis.com/weatherbench2/"
        ).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.cache_dir = cache_root / hashlib.sha256(self.url.encode()).hexdigest()[:16]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock_guard = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}
        self.session = requests.Session()
        retry = Retry(
            total=8,
            connect=8,
            read=8,
            backoff_factor=1,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.metadata = json.loads(self._get(".zmetadata"))["metadata"]

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / hashlib.sha256(key.encode()).hexdigest()

    def _get(self, key: str) -> bytes:
        cached = self._cache_path(key)
        if cached.exists():
            return cached.read_bytes()
        # Multiple requested fields can share one time chunk.  A per-key lock
        # prevents worker threads from writing the same ``.partial`` file while
        # retaining parallel downloads for genuinely different chunks.
        with self._lock_guard:
            key_lock = self._key_locks.setdefault(key, threading.Lock())
        with key_lock:
            if cached.exists():
                return cached.read_bytes()
            # urllib3 retries responses that fail before headers arrive, but a
            # chunked transfer can also fail while ``response.content`` is
            # being read. Retry that latter case explicitly; no cache entry is
            # made until the full response has been received and replaced.
            error: requests.RequestException | None = None
            for attempt in range(1, 9):
                try:
                    response = self.session.get(
                        f"{self.url}/{key}", timeout=self.timeout_seconds
                    )
                    response.raise_for_status()
                    content = response.content
                    temporary = cached.with_suffix(".partial")
                    temporary.write_bytes(content)
                    temporary.replace(cached)
                    return content
                except requests.RequestException as caught:
                    error = caught
                    if attempt == 8:
                        break
                    time.sleep(min(2 ** (attempt - 1), 30))
            assert error is not None
            raise error

    def array_metadata(self, name: str) -> dict:
        return self.metadata[f"{name}/.zarray"]

    def attrs(self, name: str) -> dict:
        return self.metadata.get(f"{name}/.zattrs", {})

    def chunk(self, name: str, indices: tuple[int, ...], shape: tuple[int, ...]) -> np.ndarray:
        metadata = self.array_metadata(name)
        key = ".".join(map(str, indices))
        content = self._get(f"{name}/{key}")
        compressor = metadata.get("compressor")
        raw = numcodecs.get_codec(compressor).decode(content) if compressor else content
        return np.frombuffer(raw, dtype=np.dtype(metadata["dtype"])).reshape(shape, order=metadata["order"])

    def coordinate(self, name: str) -> np.ndarray:
        metadata = self.array_metadata(name)
        shape = tuple(metadata["shape"])
        chunks = tuple(metadata["chunks"])
        if any(length > chunk for length, chunk in zip(shape, chunks)):
            pieces = []
            for index in range((shape[0] + chunks[0] - 1) // chunks[0]):
                length = min(chunks[0], shape[0] - index * chunks[0])
                pieces.append(self.chunk(name, (index,), (length,)))
            return np.concatenate(pieces)
        return self.chunk(name, tuple(0 for _ in shape), shape)

    def times(self, name: str) -> pd.DatetimeIndex:
        values = self.coordinate(name)
        units = self.attrs(name)["units"]
        quantity, _, origin = units.partition(" since ")
        if not origin:
            raise ValueError(f"coordinate {name} has no origin in units {units!r}")
        return pd.DatetimeIndex(pd.Timestamp(origin) + pd.to_timedelta(values, unit=quantity))

    def timedeltas_hours(self, name: str) -> np.ndarray:
        values = self.coordinate(name)
        units = self.attrs(name)["units"].lower()
        factors = {"hours": 1, "hour": 1, "minutes": 1 / 60, "seconds": 1 / 3600}
        if units not in factors:
            raise ValueError(f"unsupported lead units: {units}")
        return values * factors[units]

    def field2d(self, name: str, time_index: int, lead_index: int | None = None) -> np.ndarray:
        """Read one full horizontal chunk. The selected public stores chunk it whole."""
        metadata = self.array_metadata(name)
        shape = tuple(metadata["shape"])
        chunks = tuple(metadata["chunks"])
        if len(shape) not in {3, 4} or tuple(chunks[-2:]) != tuple(shape[-2:]):
            raise ValueError(f"unexpected horizontal chunking for {name}: {chunks} / {shape}")
        indices = [time_index // chunks[0]]
        chunk_shape = [min(chunks[0], shape[0] - indices[0] * chunks[0])]
        if len(shape) == 4:
            if lead_index is None:
                raise ValueError(f"{name} needs a lead index")
            indices.append(lead_index // chunks[1])
            chunk_shape.append(min(chunks[1], shape[1] - indices[1] * chunks[1]))
        indices.extend([0, 0])
        chunk_shape.extend(shape[-2:])
        array = self.chunk(name, tuple(indices), tuple(chunk_shape))
        time_offset = time_index % chunks[0]
        if len(shape) == 4:
            lead_offset = lead_index % chunks[1]
            return array[time_offset, lead_offset]
        return array[time_offset]

    def geographic_field2d(
        self,
        name: str,
        time_index: int,
        lead_index: int | None = None,
        latitude_name: str = "latitude",
        longitude_name: str = "longitude",
    ) -> np.ndarray:
        """Return a horizontal field ordered as ``(latitude, longitude)``.

        WeatherBench2's native 0.25-degree stores generally use latitude then
        longitude, whereas several conservatively regridded stores use the
        reverse order.  Consumers doing interpolation or geodesic operations
        need one explicit convention rather than inferring it from array shape.
        """
        field = self.field2d(name, time_index, lead_index)
        dimensions = self.attrs(name).get("_ARRAY_DIMENSIONS", [])[-2:]
        expected = [latitude_name, longitude_name]
        if dimensions == expected:
            return field
        if dimensions == expected[::-1]:
            return field.T
        raise ValueError(
            f"cannot orient {name}: horizontal dimensions {dimensions!r}, "
            f"expected {expected!r} or {expected[::-1]!r}"
        )

    def static_field2d(self, name: str) -> np.ndarray:
        """Read one static full-horizontal 2-D field (for example, terrain)."""
        metadata = self.array_metadata(name)
        shape = tuple(metadata["shape"])
        chunks = tuple(metadata["chunks"])
        if len(shape) != 2 or chunks != shape:
            raise ValueError(f"unexpected static horizontal chunking for {name}: {chunks} / {shape}")
        return self.chunk(name, (0, 0), shape)


def bilinear(grid: np.ndarray, latitudes: np.ndarray, longitudes: np.ndarray,
             target_latitude: np.ndarray, target_longitude: np.ndarray) -> np.ndarray:
    """Bilinear interpolation on a regular latitude/longitude grid."""
    lats, lons, values = latitudes.copy(), longitudes.copy(), grid
    if lats[0] > lats[-1]:
        lats, values = lats[::-1], values[::-1, :]
    if lons[0] > lons[-1]:
        lons, values = lons[::-1], values[:, ::-1]
    target_lon = np.mod(target_longitude, 360)
    i = np.clip(np.searchsorted(lats, target_latitude) - 1, 0, len(lats) - 2)
    j = np.clip(np.searchsorted(lons, target_lon) - 1, 0, len(lons) - 2)
    fy = (target_latitude - lats[i]) / (lats[i + 1] - lats[i])
    fx = (target_lon - lons[j]) / (lons[j + 1] - lons[j])
    return ((1 - fy) * (1 - fx) * values[i, j] + (1 - fy) * fx * values[i, j + 1]
            + fy * (1 - fx) * values[i + 1, j] + fy * fx * values[i + 1, j + 1])


def nearest_indices(latitudes: np.ndarray, longitudes: np.ndarray,
                    target_latitude: np.ndarray, target_longitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Grid indices of the cell containing each target. Targets in one cell share them.

    A target exactly halfway between two grid points is resolved towards the
    higher coordinate. Plain ``argmin`` would resolve it towards whichever of
    the two the store happens to hold first, which differs between the
    south-to-north and north-to-south stores and would split one cell in two.
    """
    def resolve(coordinates: np.ndarray, distance: np.ndarray) -> np.ndarray:
        tied = distance <= distance.min(axis=0) + 1e-9
        return np.where(tied, coordinates[:, None], -np.inf).argmax(axis=0)

    target_lon = np.mod(target_longitude, 360)
    offset = np.mod(longitudes[:, None] - target_lon[None, :] + 180, 360) - 180
    return (resolve(latitudes, np.abs(latitudes[:, None] - target_latitude[None, :])),
            resolve(np.mod(longitudes, 360), np.abs(offset)))
