#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_last_record(path: Path) -> dict | None:
    if not path.exists():
        return None
    last = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                last = json.loads(text)
    return last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("suite_dir", type=Path)
    args = ap.parse_args()

    suite_dir = args.suite_dir.resolve()
    grouped = defaultdict(list)
    for metrics_path in sorted(suite_dir.glob("problem=*__row=*__col=*/seed=*/metrics.jsonl")):
        parts = metrics_path.parts
        case_name = parts[-3]
        grouped[case_name].append(load_last_record(metrics_path))

    for case_name in sorted(grouped):
        records = [record for record in grouped[case_name] if record is not None]
        if not records:
            continue
        n = len(records)
        avg_res = sum(float(record["residual_norm"]) for record in records) / n
        avg_l2 = sum(float(record["l2_error"]) for record in records) / n
        avg_cons = sum(float(record["consensus_error"]) for record in records) / n
        avg_row_dis = sum(float(record.get("row_disagreement_energy", 0.0)) for record in records) / n
        avg_row_ratio = sum(float(record.get("row_disagreement_ratio", 0.0)) for record in records) / n
        print(
            f"{case_name}: "
            f"residual={avg_res:.8e}  "
            f"l2={avg_l2:.8e}  "
            f"consensus={avg_cons:.8e}  "
            f"row_dis={avg_row_dis:.8e}  "
            f"row_dis_ratio={avg_row_ratio:.8e}"
        )


if __name__ == "__main__":
    main()
