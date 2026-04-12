"""Ghi từng file JSON = body ``POST /predict`` vào ``.../cases/<id>.json`` để copy-paste Swagger."""

from __future__ import annotations

import json
from pathlib import Path


def write_cases_payload_dir(
    cases_dir: Path,
    cases: list[dict],
    *,
    readme_body: str,
) -> None:
    cases_dir.mkdir(parents=True, exist_ok=True)
    (cases_dir / "README.txt").write_text(readme_body.strip() + "\n", encoding="utf-8")
    for c in cases:
        cid = str(c["id"])
        req = c["request"]
        out = cases_dir / f"{cid}.json"
        out.write_text(
            json.dumps(req, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
