#!/usr/bin/env python3
"""Classify candidate stations before any held-out forecast extraction.

This gate answers two different questions without conflating them:

* does a candidate describe a physical site already present in Weather5K/ISD?
* does its metadata expose an operational dissemination route that could have
  supplied ERA5 (GTS, SYNOP, METAR or MADIS)?

A negative match is not proof of independence. It only permits the conservative
``low_risk_candidate`` label when non-operational provenance is also documented.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "observational_independence_registry.yaml"
DEFAULT_REFERENCE = ROOT / "data" / "interim" / "weather5k_t2m_dry_run_2020" / "stations.csv"
EARTH_RADIUS_KM = 6371.0088


def truthy(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "si", "sí"}


def clean_identifier(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    if text in {"", "NA", "N/A", "NONE", "NAN", "NULL", "-999"}:
        return ""
    return text.replace(" ", "")


def unit_sphere(frame: pd.DataFrame) -> np.ndarray:
    latitude = np.radians(frame["latitude"].to_numpy(float))
    longitude = np.radians(frame["longitude"].to_numpy(float))
    cosine = np.cos(latitude)
    return np.column_stack((cosine * np.cos(longitude), cosine * np.sin(longitude), np.sin(latitude)))


def nearest_reference(candidates: pd.DataFrame, reference: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest reference row and exact great-circle distance."""
    tree = cKDTree(unit_sphere(reference))
    _, indices = tree.query(unit_sphere(candidates), k=1)
    left = np.radians(candidates[["latitude", "longitude"]].to_numpy(float))
    right = np.radians(reference.iloc[indices][["latitude", "longitude"]].to_numpy(float))
    delta_latitude = right[:, 0] - left[:, 0]
    delta_longitude = right[:, 1] - left[:, 1]
    inner = (np.sin(delta_latitude / 2) ** 2
             + np.cos(left[:, 0]) * np.cos(right[:, 0]) * np.sin(delta_longitude / 2) ** 2)
    distance = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(inner, 0, 1)))
    return indices, distance


def available_identifiers(frame: pd.DataFrame, configured: Iterable[str]) -> list[str]:
    extras = ("site_identifier", "isd_station_id")
    return [name for name in (*configured, *extras) if name in frame.columns]


def reference_identifier_map(reference: pd.DataFrame, columns: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for column in columns:
        if column not in reference:
            continue
        for station_id, value in zip(reference["station_id"], reference[column], strict=True):
            cleaned = clean_identifier(value)
            if cleaned:
                result.setdefault(cleaned, str(station_id))
    # ISD station IDs concatenate USAF (6 digits) and WBAN (5 digits). For
    # WMO-coded land stations, USAF usually stores the five-digit WMO number
    # plus a trailing zero. Deriving all three forms catches identifier overlap
    # even when the reference file exposes only the concatenated key.
    for station_id in reference["station_id"]:
        cleaned = clean_identifier(station_id)
        if cleaned.isdigit() and len(cleaned) == 11:
            usaf, wban = cleaned[:6], cleaned[6:]
            result.setdefault(usaf, str(station_id))
            if wban != "99999":
                result.setdefault(wban, str(station_id))
            if usaf.endswith("0"):
                result.setdefault(usaf[:5], str(station_id))
    return result


def audit(candidates: pd.DataFrame, reference: pd.DataFrame, config: dict) -> pd.DataFrame:
    required = {"station_id", "network", "latitude", "longitude"}
    for label, frame in (("candidate", candidates), ("reference", reference)):
        missing = ({"station_id", "latitude", "longitude"} if label == "reference" else required) - set(frame)
        if missing:
            raise ValueError(f"{label} inventory lacks required columns: {sorted(missing)}")
        if frame[["latitude", "longitude"]].isna().any().any():
            raise ValueError(f"{label} inventory contains missing coordinates")

    policy = config["matching"]
    radius = float(policy["physical_radius_km"])
    elevation_tolerance = float(policy["elevation_tolerance_m"])
    operational_types = {str(value).upper() for value in policy["operational_report_types"]}
    low_risk_types = {str(value).upper() for value in policy["low_risk_report_types"]}
    configured_identifiers = list(policy["identifier_columns"])

    nearest, distance = nearest_reference(candidates, reference)
    nearest_rows = reference.iloc[nearest].reset_index(drop=True)
    if "altitude_m" in candidates and "altitude_m" in reference:
        candidate_elevation = pd.to_numeric(candidates["altitude_m"], errors="coerce").to_numpy()
        reference_elevation = pd.to_numeric(nearest_rows["altitude_m"], errors="coerce").to_numpy()
        elevation_gap = np.abs(candidate_elevation - reference_elevation)
        elevation_agrees = ~np.isfinite(elevation_gap) | (elevation_gap <= elevation_tolerance)
    else:
        elevation_gap = np.full(len(candidates), np.nan)
        elevation_agrees = np.ones(len(candidates), dtype=bool)
    physical_overlap = (distance <= radius) & elevation_agrees

    reference_ids = reference_identifier_map(reference, available_identifiers(reference, configured_identifiers))
    candidate_id_columns = available_identifiers(candidates, configured_identifiers)
    international_columns = [name for name in configured_identifiers if name in candidates]
    identifier_match = []
    identifier_match_reference = []
    international_id_present = []
    for row in candidates.to_dict("records"):
        values = [clean_identifier(row.get(name)) for name in candidate_id_columns]
        matches = [reference_ids[value] for value in values if value and value in reference_ids]
        identifier_match.append(bool(matches))
        identifier_match_reference.append("|".join(sorted(set(matches))))
        international_id_present.append(any(clean_identifier(row.get(name)) for name in international_columns))

    rows = candidates.reset_index(drop=True).copy()
    rows["nearest_reference_station_id"] = nearest_rows["station_id"].astype(str).to_numpy()
    rows["nearest_reference_distance_km"] = distance
    rows["nearest_reference_elevation_gap_m"] = elevation_gap
    rows["physical_overlap_with_reference"] = physical_overlap
    rows["identifier_overlap_with_reference"] = identifier_match
    rows["identifier_overlap_reference_ids"] = identifier_match_reference
    rows["international_identifier_present"] = international_id_present

    report_types = (rows["report_type"].fillna("").astype(str).str.upper()
                    if "report_type" in rows else pd.Series([""] * len(rows)))
    operational_exchange = (rows["operational_exchange"].map(truthy)
                            if "operational_exchange" in rows else pd.Series([False] * len(rows)))
    documented_non_operational = (rows["documented_non_operational"].map(truthy)
                                  if "documented_non_operational" in rows else pd.Series([False] * len(rows)))

    classifications: list[str] = []
    reasons: list[str] = []
    registry = config["networks"]
    for index, row in rows.iterrows():
        network = str(row["network"]).strip().lower()
        base = registry.get(network, {}).get("status", "unknown")
        report_type = report_types.iloc[index]
        route_high = bool(operational_exchange.iloc[index] or report_type in operational_types
                          or row["international_identifier_present"])
        overlap = bool(row["physical_overlap_with_reference"] or row["identifier_overlap_with_reference"])
        route_low = bool(documented_non_operational.iloc[index] or report_type in low_risk_types)

        if overlap:
            classification = "potentially_assimilated"
            reason = "physical or identifier overlap with Weather5K/ISD"
        elif route_high:
            classification = "potentially_assimilated"
            reason = "operational report route or international identifier present"
        elif base == "potentially_assimilated":
            classification = base
            reason = "network registry is potentially assimilated"
        elif base == "confirmed_independent":
            classification = base
            reason = "network registry records direct non-assimilation evidence"
        elif base == "low_risk_candidate" and route_low:
            classification = base
            reason = "documented non-operational route and no reference overlap"
        elif report_type in low_risk_types and route_low:
            classification = "low_risk_candidate"
            reason = "low-risk report type and no reference overlap"
        else:
            classification = "unknown"
            reason = "no operational route found, but non-assimilation provenance is insufficient"
        classifications.append(classification)
        reasons.append(reason)

    rows["assimilation_risk_class"] = classifications
    rows["assimilation_risk_reason"] = reasons
    rows["eligible_low_risk_stratum"] = rows["assimilation_risk_class"].isin(
        ["confirmed_independent", "low_risk_candidate"])
    return rows


def self_test(config: dict) -> None:
    reference = pd.DataFrame({
        "station_id": ["isd-a"], "latitude": [50.0], "longitude": [1.0],
        "altitude_m": [100.0], "wmo_id": ["01234"],
    })
    candidates = pd.DataFrame({
        "station_id": ["same-site", "gts-site", "local-site", "unknown-site"],
        "network": ["midas_open_uk", "midas_open_uk", "avamet", "eccc_hourly"],
        "latitude": [50.0001, 52.0, 53.0, 54.0],
        "longitude": [1.0001, 2.0, 3.0, 4.0],
        "altitude_m": [102.0, 120.0, 130.0, 140.0],
        "report_type": ["NCM", "SYNOP", "LOCAL_NON_GTS", ""],
        "documented_non_operational": [True, False, True, False],
    })
    result = audit(candidates, reference, config).set_index("station_id")
    expected = {
        "same-site": "potentially_assimilated",
        "gts-site": "potentially_assimilated",
        "local-site": "low_risk_candidate",
        "unknown-site": "unknown",
    }
    actual = result["assimilation_risk_class"].to_dict()
    if actual != expected:
        raise AssertionError(f"self-test mismatch: {actual} != {expected}")
    print(json.dumps({"self_test": "passed", "classifications": actual}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path, nargs="?", help="standardised candidate-station CSV")
    parser.add_argument("--reference-stations", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "observational_independence_audit")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.self_test:
        self_test(config)
        return
    if args.candidates is None:
        parser.error("candidates is required unless --self-test is used")

    candidates = pd.read_csv(args.candidates, dtype="string").convert_dtypes()
    for name in ("latitude", "longitude", "altitude_m"):
        if name in candidates:
            candidates[name] = pd.to_numeric(candidates[name], errors="coerce")
    reference = pd.read_csv(args.reference_stations, dtype="string").convert_dtypes()
    for name in ("latitude", "longitude", "altitude_m"):
        if name in reference:
            reference[name] = pd.to_numeric(reference[name], errors="coerce")

    result = audit(candidates, reference, config)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "station_independence_manifest.csv"
    result.to_csv(manifest, index=False)
    counts = result["assimilation_risk_class"].value_counts(dropna=False).to_dict()
    report = {
        "role": "pre-extraction provenance gate; contains no forecast outcomes",
        "config": str(args.config),
        "candidate_source": str(args.candidates),
        "reference_source": str(args.reference_stations),
        "n_candidates": int(len(result)),
        "classification_counts": {str(key): int(value) for key, value in counts.items()},
        "eligible_low_risk_stratum": int(result["eligible_low_risk_stratum"].sum()),
        "physical_overlaps": int(result["physical_overlap_with_reference"].sum()),
        "identifier_overlaps": int(result["identifier_overlap_with_reference"].sum()),
        "manifest": str(manifest),
    }
    (args.output / "audit_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
