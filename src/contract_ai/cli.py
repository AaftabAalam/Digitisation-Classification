from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from contract_ai.classification.classifier import ProductClassifier
from contract_ai.common.schemas import FeedbackRecord
from contract_ai.common.settings import settings
from contract_ai.contracts.pipeline import ContractExtractor
from contract_ai.mlops.monitoring import log_prediction_event
from contract_ai.retraining.manager import FeedbackManager, RetrainOrchestrator

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command("extract-contract")
def extract_contract(
    input: Path = typer.Option(..., "--input", "-i", exists=True, file_okay=True, dir_okay=False),
    output_dir: Path = typer.Option(..., "--output-dir", "-o"),
) -> None:
    result = ContractExtractor().extract(input, output_dir)
    print(f"[green]Done[/green] elements={len(result.elements)} output={output_dir}")


@app.command("classify-image")
def classify_image(
    input: Path = typer.Option(..., "--input", "-i", exists=True, file_okay=True, dir_okay=False),
    labels: str = typer.Option(..., "--labels", "-l"),
) -> None:
    label_list = [x.strip() for x in labels.split(",") if x.strip()]
    clf = ProductClassifier(model_path=settings.active_model_path if settings.active_model_path.exists() else None)
    res = clf.predict(input, label_list)
    log_prediction_event(
        settings.prediction_log_path,
        image_path=str(input),
        label=res.label,
        confidence=res.confidence,
        scores=res.scores,
        interface="cli",
    )

    queued = False
    if res.confidence < settings.low_confidence_threshold:
        FeedbackManager().add_feedback(
            FeedbackRecord(
                image_path=str(input),
                true_label="",
                predicted_label=res.label,
                confidence=res.confidence,
                metadata={"reason": "low_confidence_auto_queue"},
            )
        )
        queued = True

    print(res.model_dump())
    if queued:
        print("[yellow]Low confidence sample queued for labeling[/yellow]")


@app.command("add-feedback")
def add_feedback(
    input: Path = typer.Option(..., "--input", "-i", exists=True, file_okay=True, dir_okay=False),
    true_label: str = typer.Option(..., "--true-label"),
    predicted_label: str = typer.Option("", "--predicted-label"),
    confidence: float = typer.Option(0.0, "--confidence"),
) -> None:
    rec = FeedbackRecord(
        image_path=str(input),
        true_label=true_label,
        predicted_label=predicted_label or None,
        confidence=confidence,
    )
    out = FeedbackManager().add_feedback(rec)
    print(f"[green]Saved[/green] {out}")


@app.command("retrain")
def retrain() -> None:
    artifact = RetrainOrchestrator().maybe_retrain()
    if artifact is None:
        print("[yellow]Skipped[/yellow] not enough labeled feedback samples")
        return
    if artifact.promoted:
        print(
            f"[green]Promoted[/green] model={artifact.version} labels={artifact.labels} "
            f"metrics={artifact.metrics}"
        )
    else:
        print(
            f"[yellow]Trained but not promoted[/yellow] model={artifact.version} "
            f"gate_reasons={artifact.gate_reasons} metrics={artifact.metrics}"
        )


@app.command("mlops-run-flow")
def mlops_run_flow(min_samples: int = typer.Option(20, "--min-samples"), device: str = typer.Option("cpu", "--device")) -> None:
    from contract_ai.mlops.flows import retrain_flow

    print(retrain_flow(min_samples=min_samples, device=device))


if __name__ == "__main__":
    app()
