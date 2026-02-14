from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from contract_ai.classification.clip_backend import ClipBackend


def load_feedback_labeled(feedback_dir: Path) -> list[dict]:
    if not feedback_dir.exists():
        return []
    out: list[dict] = []
    for p in feedback_dir.glob("*.json"):
        record = json.loads(p.read_text(encoding="utf-8"))
        if record.get("true_label"):
            out.append(record)
    return out


def build_training_arrays(records: list[dict], backend: ClipBackend) -> tuple[np.ndarray, np.ndarray]:
    embs: list[np.ndarray] = []
    labels: list[str] = []
    for rec in records:
        path = Path(rec["image_path"])
        if not path.exists():
            continue
        image = Image.open(path).convert("RGB")
        embs.append(backend.embed_image(image))
        labels.append(rec["true_label"])
    if not embs:
        return np.empty((0, 512), dtype=np.float32), np.empty((0,), dtype=object)
    return np.stack(embs), np.array(labels)
