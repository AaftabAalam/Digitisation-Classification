from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def log_prediction_event(
    path: Path,
    image_path: str,
    label: str,
    confidence: float,
    scores: dict[str, float],
    interface: str,
) -> None:
    append_jsonl(
        path,
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "image_path": image_path,
            "label": label,
            "confidence": confidence,
            "scores": scores,
            "interface": interface,
        },
    )
