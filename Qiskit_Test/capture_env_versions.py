#!/usr/bin/env python3
"""Capture important package versions for reproducibility."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write environment package versions to JSON.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", default="snapshot")
    return parser.parse_args()


def safe_version(module_name: str) -> str:
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        return f"MISSING: {type(exc).__name__}: {exc}"


def safe_dist_version(dist_name: str) -> str:
    try:
        return importlib.metadata.version(dist_name)
    except Exception as exc:  # noqa: BLE001
        return f"MISSING: {type(exc).__name__}: {exc}"


def main() -> int:
    args = parse_args()
    payload = {
        "label": args.label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "packages": {
            "qiskit": safe_version("qiskit"),
            "qiskit_aer": safe_version("qiskit_aer"),
            "qiskit_algorithms": safe_version("qiskit_algorithms"),
            "numpy": safe_version("numpy"),
            "qiskit-aer-gpu": safe_dist_version("qiskit-aer-gpu"),
            "qiskit-aer-gpu-cu11": safe_dist_version("qiskit-aer-gpu-cu11"),
            "cuquantum-cu11": safe_dist_version("cuquantum-cu11"),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
