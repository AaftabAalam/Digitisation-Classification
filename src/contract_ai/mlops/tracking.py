from __future__ import annotations

from pathlib import Path
from typing import Any


def _try_import_mlflow():
    try:
        import mlflow  # type: ignore

        return mlflow
    except Exception:
        return None


def log_training_run(
    tracking_uri: str,
    experiment_name: str,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    tags: dict[str, str] | None = None,
    artifacts: list[Path] | None = None,
) -> str | None:
    mlflow = _try_import_mlflow()
    if mlflow is None:
        return None

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    tags = tags or {}
    artifacts = artifacts or []

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if tags:
            mlflow.set_tags(tags)
        for artifact_path in artifacts:
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))
        return run.info.run_id
