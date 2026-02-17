from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CONTRACT_AI_", extra="ignore")

    project_root: Path = Path.cwd()
    data_dir: Path = Path("data")
    model_dir: Path = Path("data/models/classifier")
    feedback_dir: Path = Path("data/feedback/classification")
    exports_dir: Path = Path("data/exports")
    monitoring_dir: Path = Path("data/monitoring")
    tesseract_cmd: str | None = None
    low_confidence_threshold: float = 0.5
    retrain_min_samples: int = 20
    validation_split: float = 0.2
    promote_min_f1: float = 0.70
    promote_min_precision: float = 0.70
    promote_min_recall: float = 0.70
    mlflow_tracking_uri: str = "file:./data/mlruns"
    mlflow_experiment_name: str = "contract-ai-retraining"

    @property
    def active_model_path(self) -> Path:
        return self.model_dir / "current" / "classifier.joblib"

    @property
    def prediction_log_path(self) -> Path:
        return self.monitoring_dir / "predictions.jsonl"


settings = Settings()
