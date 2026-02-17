from __future__ import annotations

from typing import Any

try:
    from prefect import flow, task
except Exception:  # pragma: no cover
    def flow(*_args, **_kwargs):
        def deco(fn):
            return fn

        return deco

    def task(*_args, **_kwargs):
        def deco(fn):
            return fn

        return deco


@task(name="run-retrain")
def run_retrain_task(min_samples: int | None = None, device: str = "cpu") -> dict[str, Any]:
    from contract_ai.retraining.manager import RetrainOrchestrator

    artifact = RetrainOrchestrator().maybe_retrain(min_samples=min_samples, device=device)
    if artifact is None:
        return {"status": "skipped", "reason": "not enough labeled feedback"}
    return {"status": "ok", "artifact": artifact.model_dump()}


@flow(name="contract-ai-retrain-flow")
def retrain_flow(min_samples: int | None = None, device: str = "cpu") -> dict[str, Any]:
    return run_retrain_task(min_samples=min_samples, device=device)
