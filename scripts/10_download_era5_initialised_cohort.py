#!/usr/bin/env python3
"""Queue the pre-specified ERA5-initialised deterministic extensions."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("era5_forecast", "graphcast_era5", "pangu_era5")


def main() -> None:
    for model in MODELS:
        print(f"starting queued extension={model}", flush=True)
        subprocess.run([sys.executable, str(ROOT / "scripts" / "09_extract_model_extension.py"), model], check=True)
    print("ERA5-initialised deterministic cohort complete", flush=True)


if __name__ == "__main__":
    main()
