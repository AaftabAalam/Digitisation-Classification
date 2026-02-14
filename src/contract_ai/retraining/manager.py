from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from contract_ai.common.schemas import FeedbackRecord, TrainArtifact
from contract_ai.common.settings import settings


class FeedbackManager:
    def __init__(self, feedback_dir: Path | None = None):
        self.feedback_dir = feedback_dir or settings.feedback_dir
        self.feedback_dir.mkdir(parents=True, exist_ok=True)

    def add_feedback(self, record: FeedbackRecord) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = self.feedback_dir / f"feedback_{ts}_{abs(hash(record.image_path)) % 999999}.json"
        out.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        return out


class RetrainOrchestrator:
    def __init__(self):
        self.model_root = settings.model_dir
        self.feedback_dir = settings.feedback_dir
        self.model_root.mkdir(parents=True, exist_ok=True)

    def maybe_retrain(self, min_samples: int | None = None, device: str = "cpu") -> TrainArtifact | None:
        from contract_ai.classification.classifier import ProductClassifier
        from contract_ai.classification.clip_backend import ClipBackend
        from contract_ai.classification.dataset import build_training_arrays, load_feedback_labeled

        threshold = min_samples or settings.retrain_min_samples
        records = load_feedback_labeled(self.feedback_dir)
        if len(records) < threshold:
            return None

        backend = ClipBackend(device=device)
        X, y = build_training_arrays(records, backend)
        if len(y) < threshold:
            return None

        version = datetime.now(timezone.utc).strftime("v%Y%m%d%H%M%S")
        version_dir = self.model_root / version
        model_path = version_dir / "classifier.joblib"

        classifier = ProductClassifier(model_path=None, device=device)
        classifier.train(X, y, model_path)

        current_dir = self.model_root / "current"
        current_dir.mkdir(parents=True, exist_ok=True)
        current_model = current_dir / "classifier.joblib"
        current_model.write_bytes(model_path.read_bytes())

        artifact = TrainArtifact(
            version=version,
            labels=sorted(set(y.tolist())),
            model_path=str(model_path),
            created_from=len(records),
        )
        (version_dir / "metadata.json").write_text(json.dumps(artifact.model_dump(), indent=2), encoding="utf-8")
        return artifact
