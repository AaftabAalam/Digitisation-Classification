from pathlib import Path

from contract_ai.common.schemas import FeedbackRecord
from contract_ai.retraining.manager import FeedbackManager


def test_feedback_written(tmp_path: Path):
    mgr = FeedbackManager(feedback_dir=tmp_path)
    p = mgr.add_feedback(
        FeedbackRecord(image_path="/tmp/x.jpg", true_label="fridge", predicted_label="hvac", confidence=0.41)
    )
    assert p.exists()
    assert p.read_text(encoding="utf-8")
