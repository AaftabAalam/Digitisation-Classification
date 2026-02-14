#!/usr/bin/env python3
from __future__ import annotations

import time

from contract_ai.retraining.manager import RetrainOrchestrator


def main(interval_sec: int = 300) -> None:
    orch = RetrainOrchestrator()
    while True:
        artifact = orch.maybe_retrain()
        if artifact:
            print(f"[retrain] promoted model {artifact.version} labels={artifact.labels}")
        else:
            print("[retrain] skipped (not enough labeled feedback)")
        time.sleep(interval_sec)


if __name__ == "__main__":
    main()
