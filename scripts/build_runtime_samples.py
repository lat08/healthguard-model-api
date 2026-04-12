#!/usr/bin/env python3
"""Copy dataset CSV variants from ``data/datasets`` into ``data/runtime/{fall,health,sleep}``.

Does **not** generate ``iot_sample_input.json``; those samples live under ``data/runtime/*/``
(and small references under ``healthguard-ai/models/*/samples/``) and are maintained separately.

Usage (from repo root)::

    python scripts/build_runtime_samples.py
    python scripts/build_runtime_samples.py --dataset-profile v1
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "data" / "datasets"
RUNTIME_FALL = ROOT / "data" / "runtime" / "fall"
RUNTIME_HEALTH = ROOT / "data" / "runtime" / "health"
RUNTIME_SLEEP = ROOT / "data" / "runtime" / "sleep"

PROFILE_FILES = {
    "default": {
        "fall": "fall_detection.csv",
        "health": "human_vital_signs_dataset_2024.csv",
        "sleep": "smartwatch_sleep_dataset.csv",
    },
    "v1": {
        "fall": "fall_detection_v1.csv",
        "health": "human_vital_signs_dataset_2024_v1.csv",
        "sleep": "smartwatch_sleep_dataset_v1.csv",
    },
    "v2": {
        "fall": "fall_detection_v2.csv",
        "health": "human_vital_signs_dataset_2024_v2.csv",
        "sleep": "smartwatch_sleep_dataset_v2.csv",
    },
}


def main() -> None:
    p = argparse.ArgumentParser(description="Sync CSV datasets into data/runtime.")
    p.add_argument(
        "--dataset-profile",
        choices=list(PROFILE_FILES.keys()),
        default="default",
        help="Which *_v1 / *_v2 variant filenames to copy.",
    )
    args = p.parse_args()
    mapping = PROFILE_FILES[args.dataset_profile]

    if not DATASETS.exists():
        print("skip: data/datasets not found — create CSVs or symlink training data here")
        sys.exit(0)

    for key, sub in ((RUNTIME_FALL, "fall"), (RUNTIME_HEALTH, "health"), (RUNTIME_SLEEP, "sleep")):
        key.mkdir(parents=True, exist_ok=True)
        src = DATASETS / mapping[sub]
        if not src.exists():
            print("skip missing", src)
            continue
        dst = key / src.name
        shutil.copy2(src, dst)
        print("copied", src.name, "->", dst)

    marker = ROOT / "data" / "runtime" / ".dataset_profile"
    marker.write_text(args.dataset_profile, encoding="utf-8")
    print("wrote", marker)


if __name__ == "__main__":
    main()
