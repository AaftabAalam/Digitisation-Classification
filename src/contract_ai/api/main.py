from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile

from contract_ai.classification.classifier import ProductClassifier
from contract_ai.common.schemas import FeedbackRecord
from contract_ai.common.settings import settings
from contract_ai.contracts.pipeline import ContractExtractor
from contract_ai.retraining.manager import FeedbackManager, RetrainOrchestrator

app = FastAPI(title="Contract Digitization AI", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/contracts/extract")
async def extract_contract(pdf: UploadFile = File(...)) -> dict:
    out_dir = settings.exports_dir / Path(pdf.filename).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / pdf.filename
    pdf_path.write_bytes(await pdf.read())

    result = ContractExtractor().extract(pdf_path, out_dir)
    return result.model_dump()


@app.post("/classification/predict")
async def classify_product(
    image: UploadFile = File(...),
    labels: str = Form(...),
) -> dict:
    labels_list = [x.strip() for x in labels.split(",") if x.strip()]
    temp_path = settings.data_dir / "incoming" / image.filename
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(await image.read())

    clf = ProductClassifier(model_path=settings.active_model_path if settings.active_model_path.exists() else None)
    res = clf.predict(temp_path, labels_list)

    if res.confidence < settings.low_confidence_threshold:
        fb = FeedbackManager()
        fb.add_feedback(
            FeedbackRecord(
                image_path=str(temp_path),
                true_label="",
                predicted_label=res.label,
                confidence=res.confidence,
                metadata={"reason": "low_confidence_auto_queue"},
            )
        )
        res.queued_for_feedback = True

    return res.model_dump()


@app.post("/feedback/classification")
def feedback_classification(
    image_path: str = Form(...),
    true_label: str = Form(...),
    predicted_label: str = Form(default=""),
    confidence: float = Form(default=0.0),
) -> dict[str, str]:
    path = FeedbackManager().add_feedback(
        FeedbackRecord(
            image_path=image_path,
            true_label=true_label,
            predicted_label=predicted_label or None,
            confidence=confidence,
        )
    )
    return {"saved": str(path)}


@app.post("/retrain/run")
def run_retrain() -> dict:
    artifact = RetrainOrchestrator().maybe_retrain()
    if artifact is None:
        return {"status": "skipped", "reason": "not enough labeled samples"}
    return {"status": "ok", "artifact": artifact.model_dump()}
