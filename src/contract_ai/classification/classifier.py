from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from contract_ai.classification.clip_backend import ClipBackend
from contract_ai.common.schemas import ClassificationResult


class ProductClassifier:
    def __init__(self, model_path: Path | None = None, device: str = "cpu"):
        self.backend = ClipBackend(device=device)
        self.model_path = model_path
        self.pipeline: Pipeline | None = None
        self.labels: list[str] = []

        if model_path and model_path.exists():
            payload = joblib.load(model_path)
            self.pipeline = payload["pipeline"]
            self.labels = payload["labels"]

    def predict(self, image_path: Path, candidate_labels: list[str]) -> ClassificationResult:
        img = Image.open(image_path).convert("RGB")

        if self.pipeline is not None and self.labels:
            embedding = self.backend.embed_image(img).reshape(1, -1)
            probs = self.pipeline.predict_proba(embedding)[0]
            scores = {label: float(score) for label, score in zip(self.pipeline.classes_, probs)}
        else:
            scores = self.backend.zero_shot_scores(img, candidate_labels)

        label = max(scores, key=scores.get)
        return ClassificationResult(label=label, confidence=scores[label], scores=scores)

    def train(self, embeddings: np.ndarray, y: np.ndarray, output_model_path: Path) -> None:
        classifier = SGDClassifier(loss="log_loss", max_iter=2000, tol=1e-4)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", classifier)])
        pipe.fit(embeddings, y)

        payload = {"pipeline": pipe, "labels": sorted(set(y.tolist()))}
        output_model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, output_model_path)
