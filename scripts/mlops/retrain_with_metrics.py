#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from contract_ai.retraining.manager import RetrainOrchestrator


def main() -> None:
    artifact = RetrainOrchestrator().maybe_retrain()
    out_path = Path("data/models/classifier/latest_retrain_result.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"status": "skipped" if artifact is None else "ok"}
    if artifact is not None:
        payload["artifact"] = artifact.model_dump()

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
