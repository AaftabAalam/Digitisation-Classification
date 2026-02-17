#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def main() -> None:
    path = Path("data/monitoring/predictions.jsonl")
    out = Path("data/monitoring/summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        payload = {"events": 0, "avg_confidence": 0.0, "label_distribution": {}}
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return

    total = 0
    conf_sum = 0.0
    labels = Counter()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1
            conf_sum += float(rec.get("confidence", 0.0))
            labels[rec.get("label", "unknown")] += 1

    payload = {
        "events": total,
        "avg_confidence": (conf_sum / total) if total else 0.0,
        "label_distribution": dict(labels),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
