from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class BBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class ContractElement(BaseModel):
    element_id: str
    element_type: str
    page_number: int
    bbox: BBox
    order_index: int
    text: str | None = None
    image_path: str | None = None
    table_data: list[list[str]] | None = None
    style: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContractPage(BaseModel):
    page_number: int
    width: float
    height: float
    image_path: str | None = None


class ContractExtractionResult(BaseModel):
    source_pdf: str
    page_count: int
    pages: list[ContractPage] = Field(default_factory=list)
    elements: list[ContractElement]
    output_dir: str


class ClassificationResult(BaseModel):
    label: str
    confidence: float
    scores: dict[str, float]
    queued_for_feedback: bool = False


class FeedbackRecord(BaseModel):
    image_path: str
    true_label: str
    predicted_label: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrainArtifact(BaseModel):
    version: str
    labels: list[str]
    model_path: str
    created_from: int
    metrics: dict[str, float] = Field(default_factory=dict)
    promoted: bool = True
    gate_reasons: list[str] = Field(default_factory=list)
    mlflow_run_id: str | None = None


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
