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
    tesseract_cmd: str | None = None
    low_confidence_threshold: float = 0.5
    retrain_min_samples: int = 20

    @property
    def active_model_path(self) -> Path:
        return self.model_dir / "current" / "classifier.joblib"


settings = Settings()
