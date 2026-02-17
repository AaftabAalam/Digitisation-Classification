from contract_ai.mlops.gates import evaluate_promotion_gates


def test_gates_pass_when_metrics_meet_thresholds():
    decision = evaluate_promotion_gates(
        metrics={"f1_weighted": 0.8, "precision_weighted": 0.79, "recall_weighted": 0.81},
        min_f1=0.7,
        min_precision=0.7,
        min_recall=0.7,
    )
    assert decision.approved is True
    assert decision.reasons == []


def test_gates_fail_when_metrics_below_thresholds():
    decision = evaluate_promotion_gates(
        metrics={"f1_weighted": 0.62, "precision_weighted": 0.79, "recall_weighted": 0.65},
        min_f1=0.7,
        min_precision=0.7,
        min_recall=0.7,
    )
    assert decision.approved is False
    assert len(decision.reasons) == 2
