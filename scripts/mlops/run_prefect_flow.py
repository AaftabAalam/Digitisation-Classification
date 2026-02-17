#!/usr/bin/env python3
from __future__ import annotations

import argparse

from contract_ai.mlops.flows import retrain_flow


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retraining via Prefect flow")
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    result = retrain_flow(min_samples=args.min_samples, device=args.device)
    print(result)


if __name__ == "__main__":
    main()
