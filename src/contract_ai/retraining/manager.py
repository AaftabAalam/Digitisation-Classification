from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from contract_ai.common.schemas import FeedbackRecord, TrainArtifact
from contract_ai.common.settings import settings
from contract_ai.mlops.gates import evaluate_promotion_gates
from contract_ai.mlops.tracking import log_training_run


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
        version_dir.mkdir(parents=True, exist_ok=True)
        candidate_model_path = version_dir / "classifier.joblib"

        train_idx, val_idx = self._split_indices(y)
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        classifier = ProductClassifier(model_path=None, device=device)
        classifier.train(X_train, y_train, candidate_model_path)

        eval_model = ProductClassifier(model_path=candidate_model_path, device=device)
        val_pred = eval_model.pipeline.predict(X_val)  # type: ignore[union-attr]
        metrics = self._compute_metrics(y_val, val_pred)

        decision = evaluate_promotion_gates(
            metrics=metrics,
            min_f1=settings.promote_min_f1,
            min_precision=settings.promote_min_precision,
            min_recall=settings.promote_min_recall,
        )

        mlflow_run_id = log_training_run(
            tracking_uri=settings.mlflow_tracking_uri,
            experiment_name=settings.mlflow_experiment_name,
            run_name=version,
            params={
                "threshold": threshold,
                "device": device,
                "val_size": len(y_val),
                "train_size": len(y_train),
                "promote_min_f1": settings.promote_min_f1,
                "promote_min_precision": settings.promote_min_precision,
                "promote_min_recall": settings.promote_min_recall,
            },
            metrics=metrics,
            tags={"promoted": str(decision.approved), "project": "contract-digitization-ai"},
            artifacts=[candidate_model_path],
        )

        active_model_path = candidate_model_path
        if decision.approved:
            current_dir = self.model_root / "current"
            current_dir.mkdir(parents=True, exist_ok=True)
            current_model = current_dir / "classifier.joblib"
            current_model.write_bytes(candidate_model_path.read_bytes())
            active_model_path = current_model

        artifact = TrainArtifact(
            version=version,
            labels=sorted(set(y.tolist())),
            model_path=str(active_model_path),
            created_from=len(records),
            metrics=metrics,
            promoted=decision.approved,
            gate_reasons=decision.reasons,
            mlflow_run_id=mlflow_run_id,
        )

        metadata_path = version_dir / "metadata.json"
        metadata_path.write_text(json.dumps(artifact.model_dump(), indent=2), encoding="utf-8")
        return artifact

    @staticmethod
    def _split_indices(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(y)
        indices = np.arange(n)
        unique, counts = np.unique(y, return_counts=True)
        can_stratify = len(unique) > 1 and counts.min() >= 2

        if n < 10:
            cut = max(1, int(0.8 * n))
            return indices[:cut], indices[cut:]

        val_size = min(max(settings.validation_split, 0.1), 0.4)
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_size,
            random_state=42,
            stratify=y if can_stratify else None,
        )
        return train_idx, val_idx

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
        accuracy = accuracy_score(y_true, y_pred)
        return {
            "accuracy": float(accuracy),
            "precision_weighted": float(precision),
            "recall_weighted": float(recall),
            "f1_weighted": float(f1),
        }
