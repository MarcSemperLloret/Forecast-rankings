#!/usr/bin/env python3
"""Write a secret-free status report for the 2020 national-network acquisition."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "national_network_acquisition_2020"


def load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def sensitivity_summary(report: dict | None) -> dict:
    if report is None:
        return {"status": "not_started"}
    primary = report["family_point_results"]["50"]
    interval = report["bootstrap_family_slopes"]["50"]["7"]
    pattern = report["validation_pattern"]
    return {
        "status": "completed",
        "cells": report["support"]["cells"],
        "days": report["support"]["days"],
        "slope_50km_c_per_k": primary["slope_c_per_k"],
        "ci95_7day_blocks": [interval["ci_low"], interval["ci_high"]],
        "association_replication_supported": pattern["association_replication_supported"],
        "full_preregistered_confirmation_supported": pattern[
            "full_preregistered_confirmation_supported"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    inmet = load(ROOT / "data" / "interim" / "inmet_t2m_2020" / "manifest.json")
    jma = load(ROOT / "data" / "interim" / "jma_amedas_inventory_2020" / "manifest.json")
    midas = load(ROOT / "data" / "interim" / "midas_open_t2m_2020" / "manifest.json")
    midas_raw = load(
        ROOT / "data" / "raw" / "national_networks" / "midas_open_hourly_2020" / "manifest.json")
    inmet_audit = load(ROOT / "results" / "inmet_independence_2020" / "audit_summary.json")
    jma_audit = load(ROOT / "results" / "jma_amedas_independence_2020" / "audit_summary.json")
    midas_audit = load(ROOT / "results" / "midas_open_independence_2020" / "audit_summary.json")
    midas_sensitivity = load(ROOT / "results" / "midas_unknown_validation_2020" / "summary.json")
    inmet_sensitivity = load(ROOT / "results" / "inmet_unknown_validation_2020" / "summary.json")
    combined_sensitivity = load(
        ROOT / "results" / "national_unknown_combined_validation_2020" / "summary.json")
    credentials = {
        "ceda_token_present": bool(os.environ.get("CEDA_TOKEN")),
        "frost_client_id_present": bool(os.environ.get("FROST_CLIENT_ID")),
        "niwa_customer_id_present": bool(os.environ.get("NIWA_CUSTOMER_ID")),
        "niwa_api_key_present": bool(os.environ.get("NIWA_API_KEY")),
    }
    report = {
        "analysis_year": 2020,
        "credential_values_recorded": False,
        "credentials_present": credentials,
        "networks": {
            "inmet_hourly": {
                "status": "observations_materialised" if inmet else "not_started",
                "stations_meeting_coverage": inmet.get("stations_meeting_min_samples") if inmet else None,
                "observation_rows": inmet.get("rows_after_station_coverage") if inmet else None,
                "independence_classification": (
                    inmet_audit.get("classification_counts") if inmet_audit else None),
                "next_action": "obtain report-route evidence; do not extract forecasts while all sites are unknown",
            },
            "jma_amedas": {
                "status": "inventory_materialised_observations_blocked" if jma else "not_started",
                "native_stations_active_2020": jma.get("native_stations") if jma else None,
                "independence_classification": (
                    jma_audit.get("classification_counts") if jma_audit else None),
                "blocker": (
                    "JMA public UI enforces request-size limits and directs large-volume users "
                    "to the Japan Meteorological Business Support Center"
                ),
            },
            "midas_open_uk": {
                "status": "observations_materialised" if midas else "not_started",
                "raw_2020_station_files": (
                    midas_raw.get("materialised_files") if midas_raw else None),
                "stations_meeting_coverage": (
                    midas.get("stations_meeting_min_samples") if midas else None),
                "observation_rows": midas.get("rows_after_station_coverage") if midas else None,
                "independence_classification": (
                    midas_audit.get("classification_counts") if midas_audit else None),
                "credential_detected_now": credentials["ceda_token_present"],
                "credential_value_recorded": False,
                "next_action": (
                    "retain unknown non-overlapping sites as sensitivity evidence; require "
                    "documented non-assimilation before confirmatory promotion"
                ),
            },
            "met_norway_frost": {
                "status": "blocked_missing_client_id",
                "required": "free Frost account/client ID in FROST_CLIENT_ID",
                "credential_detected": credentials["frost_client_id_present"],
            },
            "niwa_climate_station": {
                "status": "blocked_account_and_licence_acceptance",
                "required": (
                    "DataHub account, applicable licence acceptance, NIWA_CUSTOMER_ID and NIWA_API_KEY"
                ),
                "credentials_detected": bool(
                    credentials["niwa_customer_id_present"] and credentials["niwa_api_key_present"]),
            },
        },
        "national_sensitivity_runs": {
            "midas_unknown_nonoverlap": sensitivity_summary(midas_sensitivity),
            "inmet_unknown_nonoverlap": sensitivity_summary(inmet_sensitivity),
            "midas_inmet_combined_nonoverlap": sensitivity_summary(combined_sensitivity),
        },
        "scientific_rule": (
            "download completion does not imply ERA5 independence; forecast extraction begins only "
            "after station-level route and overlap classification"
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "status.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
