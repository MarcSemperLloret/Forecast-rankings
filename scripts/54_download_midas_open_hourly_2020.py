#!/usr/bin/env python3
"""Download MIDAS Open hourly weather observations for calendar year 2020.

The script discovers counties and station directories from the CEDA archive,
then requests only the QC-version-1 annual file for 2020. Authentication is
read exclusively from ``CEDA_TOKEN`` and is never written to disk.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_URL = (
    "https://dap.ceda.ac.uk/badc/ukmo-midas-open/data/"
    "uk-hourly-weather-obs/dataset-version-202107/"
)
DEFAULT_OUTPUT = (
    ROOT / "data" / "raw" / "national_networks" / "midas_open_hourly_2020"
)
STATION_RE = re.compile(r"^\d{5}_[^/]+/$")


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)


@dataclass(frozen=True)
class Candidate:
    county: str
    station: str
    url: str
    destination: str


def request(url: str, token: str, timeout: int = 90) -> urllib.response.addinfourl:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "weather-rank-reference-pilot/1.0",
        },
    )
    return urllib.request.urlopen(req, timeout=timeout)


def list_links(url: str, token: str) -> list[str]:
    with request(url, token) as response:
        body = response.read().decode("utf-8", errors="replace")
    parser = LinkParser()
    parser.feed(body)
    return parser.links


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_candidates(
    token: str, output: Path, year: int, workers: int
) -> tuple[list[Candidate], list[str], dict[str, str]]:
    county_links = [
        href
        for href in list_links(BASE_URL, token)
        if href.endswith("/")
        and href != "../"
        and not href.startswith("change_log")
        and not urllib.parse.urlparse(href).scheme
    ]
    counties = sorted({href.strip("/") for href in county_links})
    failures: dict[str, str] = {}

    def stations_for_county(county: str) -> tuple[str, list[str]]:
        url = urllib.parse.urljoin(BASE_URL, f"{county}/")
        stations = sorted({
            href.strip("/")
            for href in list_links(url, token)
            if STATION_RE.match(href)
        })
        return county, stations

    station_map: dict[str, list[str]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(stations_for_county, county): county for county in counties}
        for future in concurrent.futures.as_completed(futures):
            county = futures[future]
            try:
                found_county, stations = future.result()
                station_map[found_county] = stations
            except Exception as exc:  # retained in the secret-free manifest
                failures[county] = f"{type(exc).__name__}: {exc}"

    candidates: list[Candidate] = []
    for county in sorted(station_map):
        for station in station_map[county]:
            filename = (
                "midas-open_uk-hourly-weather-obs_dv-202107_"
                f"{county}_{station}_qcv-1_{year}.csv"
            )
            relative = Path("files") / county / filename
            url = urllib.parse.urljoin(
                BASE_URL,
                f"{county}/{station}/qc-version-1/{filename}",
            )
            candidates.append(
                Candidate(county, station, url, relative.as_posix())
            )
    return candidates, counties, failures


def fetch_candidate(
    candidate: Candidate, token: str, output: Path, retries: int
) -> dict[str, object]:
    destination = output / candidate.destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return {
            **asdict(candidate),
            "status": "existing",
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }

    temporary = destination.with_suffix(
        destination.suffix + f".part.{os.getpid()}.{threading.get_ident()}"
    )
    for attempt in range(retries + 1):
        try:
            digest = hashlib.sha256()
            size = 0
            with request(candidate.url, token) as response, temporary.open("wb") as stream:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(block)
                    digest.update(block)
                    size += len(block)
            if size == 0:
                raise OSError("empty response")
            os.replace(temporary, destination)
            return {
                **asdict(candidate),
                "status": "downloaded",
                "bytes": size,
                "sha256": digest.hexdigest(),
            }
        except urllib.error.HTTPError as exc:
            temporary.unlink(missing_ok=True)
            if exc.code == 404:
                return {**asdict(candidate), "status": "absent_2020", "http_status": 404}
            if attempt == retries:
                return {
                    **asdict(candidate),
                    "status": "error",
                    "error": f"HTTP {exc.code}",
                }
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            if attempt == retries:
                return {
                    **asdict(candidate),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")
    token = os.environ.get("CEDA_TOKEN")
    if not token:
        parser.error("CEDA_TOKEN is required")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidates, counties, discovery_failures = discover_candidates(
        token, output, args.year, args.workers
    )
    results: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(fetch_candidate, candidate, token, output, args.retries)
            for candidate in candidates
        ]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            results.append(future.result())
            if index % 100 == 0 or index == len(futures):
                print(f"Checked {index}/{len(futures)} station candidates", flush=True)

    results.sort(key=lambda row: (str(row["county"]), str(row["station"])))
    counts: dict[str, int] = {}
    for row in results:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
    materialised = [row for row in results if row["status"] in {"downloaded", "existing"}]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_year": args.year,
        "dataset": "MIDAS Open: UK hourly weather observation data, v202107",
        "doi": "10.5285/3bd7221d4844435dad2fa030f26ab5fd",
        "base_url": BASE_URL,
        "qc_version": 1,
        "credential_value_recorded": False,
        "county_count": len(counties),
        "station_candidate_count": len(candidates),
        "discovery_failures": discovery_failures,
        "status_counts": counts,
        "materialised_files": len(materialised),
        "materialised_bytes": sum(int(row["bytes"]) for row in materialised),
        "files": results,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "county_count": len(counties),
                "station_candidate_count": len(candidates),
                "status_counts": counts,
                "materialised_bytes": manifest["materialised_bytes"],
                "discovery_failure_count": len(discovery_failures),
            },
            indent=2,
        )
    )
    if discovery_failures or counts.get("error", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
