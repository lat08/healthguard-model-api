#!/usr/bin/env python3
"""Load joblib bundles under ``models/`` and print structure (artifact audit)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_ROOT = ROOT / "models"


def _describe(obj: object, depth: int = 0) -> None:
    ind = "  " * depth
    t = type(obj).__name__
    if isinstance(obj, dict):
        print(f"{ind}dict keys: {list(obj.keys())}")
        for k, v in obj.items():
            if k in ("preprocessor", "model"):
                print(f"{ind}  [{k}] -> {type(v).__name__}")
            elif k == "feature_names" and isinstance(v, list):
                print(f"{ind}  [{k}] len={len(v)}")
            elif k == "metadata" and isinstance(v, dict):
                print(f"{ind}  [metadata] keys: {list(v.keys())[:12]}...")
            else:
                print(f"{ind}  [{k}] -> {type(v).__name__}")
    else:
        print(f"{ind}{t}")


def main() -> None:
    paths = [
        MODELS_ROOT / "fall" / "fall_bundle.joblib",
        MODELS_ROOT / "healthguard" / "healthguard_bundle.joblib",
        MODELS_ROOT / "Sleep" / "sleep_score_bundle.joblib",
    ]
    for p in paths:
        print("=" * 60)
        print(p)
        if not p.exists():
            print("  (missing)")
            continue
        import joblib

        try:
            bundle = joblib.load(p)
        except Exception as exc:
            print(f"  ERROR loading: {exc}")
            continue
        _describe(bundle)


if __name__ == "__main__":
    main()
