"""Checks that a station archive has to pass before it is analysed.

Every check here exists because the corresponding defect produces valid-looking
numbers rather than an error: a longitude with the wrong sign still lands
somewhere, a naive timestamp still compares, a station duplicated across two
networks still contributes twice. The pilot lost a morning to one of these —
a tie in the nearest-neighbour lookup that sent one station to two different
grid cells depending on the model — so they are automated rather than trusted.

Each check returns a Finding. A check that cannot decide returns severity
"unknown" instead of passing quietly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


@dataclass
class Finding:
    check: str
    severity: str  # "pass", "warn", "fail", "unknown"
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.severity.upper():7}] {self.check}: {self.message}"


def check_coordinates(stations: pd.DataFrame) -> list[Finding]:
    """Latitude and longitude present, in range, and not obviously swapped."""
    findings = []
    missing = stations[["latitude", "longitude"]].isna().any(axis=1).sum()
    findings.append(Finding("coordinates_present", "fail" if missing else "pass",
                            f"{missing} stations lack a coordinate", {"missing": int(missing)}))
    out_of_range = ((stations.latitude.abs() > 90) | (stations.longitude.abs() > 180)).sum()
    findings.append(Finding("coordinates_in_range", "fail" if out_of_range else "pass",
                            f"{out_of_range} stations fall outside the sphere", {"out_of_range": int(out_of_range)}))
    # A swapped pair is only detectable when the latitude would have been legal
    # as a longitude and the station lands somewhere implausible.
    suspicious = ((stations.latitude.abs() <= 180) & (stations.latitude.abs() > 90)).sum()
    findings.append(Finding("coordinates_not_swapped", "fail" if suspicious else "pass",
                            f"{suspicious} stations have a latitude only valid as a longitude",
                            {"suspicious": int(suspicious)}))
    at_origin = ((stations.latitude.abs() < 1e-6) & (stations.longitude.abs() < 1e-6)).sum()
    findings.append(Finding("coordinates_not_null_island", "warn" if at_origin else "pass",
                            f"{at_origin} stations sit at exactly (0, 0)", {"at_origin": int(at_origin)}))
    return findings


def check_elevation(stations: pd.DataFrame, column: str = "altitude_m") -> list[Finding]:
    """Elevation present and physically possible.

    This one matters more than it looks: the signed station-minus-grid mismatch
    is the covariate of the mechanism, so a wrong elevation does not add noise,
    it adds bias to the explanatory variable.
    """
    if column not in stations:
        return [Finding("elevation_present", "fail", f"no {column} column")]
    values = stations[column]
    missing = int(values.isna().sum())
    impossible = int(((values < -430) | (values > 8850)).sum())
    suspicious_zero = int((values == 0).sum())
    return [
        Finding("elevation_present", "fail" if missing else "pass",
                f"{missing} stations lack an elevation", {"missing": missing}),
        Finding("elevation_plausible", "fail" if impossible else "pass",
                f"{impossible} elevations fall outside the Dead Sea to Everest range", {"impossible": impossible}),
        Finding("elevation_not_sentinel_zero", "warn" if suspicious_zero > len(stations) * 0.02 else "pass",
                f"{suspicious_zero} stations report exactly 0 m, which is often a missing-value sentinel",
                {"exact_zero": suspicious_zero}),
    ]


def check_timestamps(times: pd.Series, expected_hours: set[int] | None = None,
                     series_ids: pd.Series | None = None) -> list[Finding]:
    """Timezone handling, uniqueness and the calendar the archive covers.

    When a table contains several station/model series, timestamps are expected
    to repeat across series.  In that case uniqueness is checked on the
    ``(series_id, timestamp)`` pair rather than on the timestamp alone.
    """
    findings = []
    stamps = pd.to_datetime(times)
    aware = stamps.dt.tz is not None
    findings.append(Finding("timestamps_timezone_declared", "pass" if aware else "warn",
                            "timestamps carry a timezone" if aware else
                            "timestamps are naive; they must be documented as UTC before use",
                            {"tz_aware": bool(aware)}))
    if aware:
        stamps = stamps.dt.tz_convert("UTC").dt.tz_localize(None)
    if series_ids is None:
        duplicated = int(stamps.duplicated().sum())
        uniqueness_unit = "timestamp"
    else:
        pairs = pd.DataFrame({"series_id": series_ids.to_numpy(), "timestamp": stamps.to_numpy()})
        duplicated = int(pairs.duplicated().sum())
        uniqueness_unit = "series_id + timestamp"
    findings.append(Finding("timestamps_unique", "fail" if duplicated else "pass",
                            f"{duplicated} duplicated {uniqueness_unit} pairs",
                            {"duplicated": duplicated, "unit": uniqueness_unit}))
    if expected_hours is not None:
        unexpected = sorted(set(stamps.dt.hour.unique()) - expected_hours)
        findings.append(Finding("timestamps_expected_hours", "fail" if unexpected else "pass",
                                f"unexpected hours present: {unexpected}" if unexpected else "only the expected hours",
                                {"unexpected_hours": unexpected}))
    leap = stamps[(stamps.dt.month == 2) & (stamps.dt.day == 29)]
    findings.append(Finding("timestamps_leap_day", "pass" if len(leap) or not _spans_leap_year(stamps) else "warn",
                            "leap day present or period contains none" if len(leap) or not _spans_leap_year(stamps)
                            else "the period contains a leap year but no 29 February",
                            {"leap_day_records": int(len(leap))}))
    return findings


def _spans_leap_year(stamps: pd.Series) -> bool:
    years = stamps.dt.year.unique()
    return any(year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) for year in years)


def check_temperature_units(values: pd.Series) -> list[Finding]:
    """Celsius or something else. Kelvin and Fahrenheit both survive a range check."""
    finite = values.dropna()
    if finite.empty:
        return [Finding("temperature_units", "unknown", "no finite temperatures to judge")]
    median = float(finite.median())
    if 150 < median < 350:
        return [Finding("temperature_units", "fail", f"median {median:.1f} looks like kelvin, not celsius",
                        {"median": median})]
    if 40 < median <= 150:
        return [Finding("temperature_units", "fail", f"median {median:.1f} looks like fahrenheit",
                        {"median": median})]
    return [Finding("temperature_units", "pass", f"median {median:.1f} is consistent with celsius", {"median": median})]


def haversine_km(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Great-circle distance between two arrays of (latitude, longitude) in degrees."""
    lat1, lon1 = np.radians(first[:, 0])[:, None], np.radians(first[:, 1])[:, None]
    lat2, lon2 = np.radians(second[:, 0])[None, :], np.radians(second[:, 1])[None, :]
    inner = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(inner, 0, 1)))


def find_duplicate_stations(stations: pd.DataFrame, distance_km: float = 0.5,
                            elevation_tolerance_m: float = 30.0) -> pd.DataFrame:
    """Candidate duplicates: the same site published by more than one network.

    Two stations count as candidates when they sit within ``distance_km`` of
    each other and their elevations agree to ``elevation_tolerance_m``. Pairs
    from the same network are reported too, because a network can also publish
    one site twice under two identifiers.
    """
    coordinates = stations[["latitude", "longitude"]].to_numpy(float)
    distance = haversine_km(coordinates, coordinates)
    np.fill_diagonal(distance, np.inf)
    close = np.argwhere(distance <= distance_km)
    rows = []
    for first, second in close:
        if first >= second:
            continue
        left, right = stations.iloc[first], stations.iloc[second]
        elevation_gap = np.nan
        if "altitude_m" in stations:
            elevation_gap = abs(float(left.altitude_m) - float(right.altitude_m))
        rows.append({"station_a": left.station_id, "station_b": right.station_id,
                     "network_a": left.get("network", "unknown"), "network_b": right.get("network", "unknown"),
                     "distance_km": float(distance[first, second]), "elevation_gap_m": elevation_gap,
                     "same_site": bool(np.isnan(elevation_gap) or elevation_gap <= elevation_tolerance_m)})
    return pd.DataFrame(rows, columns=["station_a", "station_b", "network_a", "network_b",
                                       "distance_km", "elevation_gap_m", "same_site"])


def summarise(findings: list[Finding]) -> dict[str, Any]:
    counts = {severity: sum(1 for item in findings if item.severity == severity)
              for severity in ("pass", "warn", "fail", "unknown")}
    return {"counts": counts, "passed": counts["fail"] == 0,
            "findings": [{"check": item.check, "severity": item.severity, "message": item.message,
                          "detail": item.detail} for item in findings]}
