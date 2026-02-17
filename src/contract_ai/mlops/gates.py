from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PromotionDecision:
    approved: bool
    reasons: list[str]


def evaluate_promotion_gates(
    metrics: dict[str, float],
    min_f1: float,
    min_precision: float,
    min_recall: float,
) -> PromotionDecision:
    reasons: list[str] = []

    f1 = float(metrics.get("f1_weighted", 0.0))
    precision = float(metrics.get("precision_weighted", 0.0))
    recall = float(metrics.get("recall_weighted", 0.0))

    if f1 < min_f1:
        reasons.append(f"f1_weighted={f1:.4f} < min_f1={min_f1:.4f}")
    if precision < min_precision:
        reasons.append(f"precision_weighted={precision:.4f} < min_precision={min_precision:.4f}")
    if recall < min_recall:
        reasons.append(f"recall_weighted={recall:.4f} < min_recall={min_recall:.4f}")

    return PromotionDecision(approved=len(reasons) == 0, reasons=reasons)
