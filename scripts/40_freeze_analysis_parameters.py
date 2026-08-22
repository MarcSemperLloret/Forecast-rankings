#!/usr/bin/env python3
"""Lock the pre-registered parameters, and refuse to run if they moved.

A pre-registration that lives only in prose is a promise. This makes it a
check: the sealed configuration sections are hashed into a lock file, and the
confirmatory analysis verifies that hash before it computes anything. Changing
a scale, a radius, a minimum or a bootstrap width after seeing a result now
breaks the pipeline instead of passing unnoticed.

    python scripts/40_freeze_analysis_parameters.py --freeze   once, before the global data
    python scripts/40_freeze_analysis_parameters.py --verify   at the head of every confirmatory run

If a parameter genuinely has to change, the honest route is to re-freeze with
an explicit reason, which the lock file records and keeps alongside the old
entry. The history stays in the file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "regional_t2m_2020.yaml"
LOCK = ROOT / "config" / "frozen_analysis.lock.json"

# Exactly the sections the pre-registration commits to. Anything outside this
# list is free to change without breaking the seal; anything inside is not.
SEALED_SECTIONS = ("controlled_smoothing", "reference_support_ladder", "field_smoothness",
                   "daily_resolution_degradation", "mechanism_replication", "spatial_weighting",
                   "bootstrap")


def sealed_values(cfg: dict) -> dict:
    missing = [section for section in SEALED_SECTIONS if section not in cfg]
    if missing:
        raise RuntimeError(f"the configuration lacks sealed sections: {missing}")
    return {section: cfg[section] for section in SEALED_SECTIONS}


def digest(values: dict) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def freeze(reason: str) -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    values = sealed_values(cfg)
    entry = {"frozen_on": date.today().isoformat(), "reason": reason, "sha256": digest(values),
             "sections": SEALED_SECTIONS, "values": values}
    history = json.loads(LOCK.read_text(encoding="utf-8")).get("history", []) if LOCK.exists() else []
    if history and history[-1]["sha256"] == entry["sha256"]:
        print(f"already frozen at {entry['sha256'][:12]}; nothing to do")
        return
    history.append(entry)
    LOCK.write_text(json.dumps({"current": entry["sha256"], "history": history}, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print(f"frozen at {entry['sha256'][:12]} on {entry['frozen_on']}")
    if len(history) > 1:
        print(f"this is re-freeze number {len(history)}; the previous entries stay in the lock file")


def verify() -> bool:
    if not LOCK.exists():
        print("no lock file; run --freeze before the confirmatory analysis", file=sys.stderr)
        return False
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    current = digest(sealed_values(cfg))
    if current == lock["current"]:
        print(f"sealed parameters match the lock ({current[:12]})")
        return True
    print(f"SEALED PARAMETERS HAVE CHANGED\n  lock:   {lock['current'][:12]}\n  config: {current[:12]}",
          file=sys.stderr)
    frozen = lock["history"][-1]["values"]
    live = sealed_values(cfg)
    for section in SEALED_SECTIONS:
        if frozen.get(section) != live.get(section):
            for key in sorted(set(frozen.get(section, {})) | set(live.get(section, {}))):
                if frozen.get(section, {}).get(key) != live.get(section, {}).get(key):
                    print(f"  {section}.{key}: {frozen.get(section, {}).get(key)!r} -> "
                          f"{live.get(section, {}).get(key)!r}", file=sys.stderr)
    return False


def show() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    print(yaml.safe_dump(sealed_values(cfg), allow_unicode=True, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--verify", action="store_true")
    group.add_argument("--show", action="store_true")
    parser.add_argument("--reason", default="sealed before the global confirmatory phase")
    args = parser.parse_args()
    if args.freeze:
        freeze(args.reason)
    elif args.show:
        show()
    elif not verify():
        sys.exit(1)


if __name__ == "__main__":
    main()
