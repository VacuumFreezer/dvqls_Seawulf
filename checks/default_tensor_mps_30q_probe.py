import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pennylane as qml
import psutil


def rss_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)


def make_cluster_probe(max_bond_dim: int):
    n = 30
    wires = list(range(n))
    dev = qml.device(
        "default.tensor",
        wires=n,
        method="mps",
        max_bond_dim=max_bond_dim,
    )

    @qml.qnode(dev)
    def cluster_probe():
        for w in wires:
            qml.Hadamard(wires=w)
        for a, b in zip(wires[:-1], wires[1:]):
            qml.CZ(wires=[a, b])
        return [
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1) @ qml.PauliZ(2)),
            qml.expval(qml.PauliZ(6) @ qml.PauliX(7) @ qml.PauliZ(8)),
            qml.expval(qml.PauliZ(12) @ qml.PauliX(13) @ qml.PauliZ(14)),
            qml.expval(qml.PauliZ(18) @ qml.PauliX(19) @ qml.PauliZ(20)),
            qml.expval(qml.PauliZ(24) @ qml.PauliX(25) @ qml.PauliZ(26)),
        ]

    return cluster_probe


def make_hadamard_probe(max_bond_dim: int):
    n_data = 29
    anc = n_data
    dev = qml.device(
        "default.tensor",
        wires=n_data + 1,
        method="mps",
        max_bond_dim=max_bond_dim,
    )

    @qml.qnode(dev)
    def hadamard_probe():
        for w in range(n_data):
            qml.Hadamard(wires=w)
        for a, b in zip(range(n_data - 1), range(1, n_data)):
            qml.CZ(wires=[a, b])

        qml.Hadamard(wires=anc)
        qml.ctrl(lambda: qml.PauliZ(wires=0), control=anc)()
        qml.ctrl(lambda: qml.PauliX(wires=1), control=anc)()
        qml.ctrl(lambda: qml.PauliZ(wires=2), control=anc)()
        qml.Hadamard(wires=anc)
        return qml.expval(qml.PauliZ(anc))

    return hadamard_probe


def run_probe(name: str, fn):
    rss_before = rss_gb()
    t0 = time.time()
    try:
        value = fn()
        elapsed = time.time() - t0
        rss_after = rss_gb()
        return {
            "name": name,
            "ok": True,
            "elapsed_s": elapsed,
            "rss_before_gb": rss_before,
            "rss_after_gb": rss_after,
            "rss_delta_gb": rss_after - rss_before,
            "value": np.asarray(value).tolist(),
        }
    except Exception as exc:
        elapsed = time.time() - t0
        rss_after = rss_gb()
        return {
            "name": name,
            "ok": False,
            "elapsed_s": elapsed,
            "rss_before_gb": rss_before,
            "rss_after_gb": rss_after,
            "rss_delta_gb": rss_after - rss_before,
            "error": repr(exc),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_bond_dim", type=int, default=32)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "python_pid": os.getpid(),
        "max_bond_dim": int(args.max_bond_dim),
        "initial_rss_gb": rss_gb(),
        "results": [],
    }

    report["results"].append(run_probe("cluster_30_stabilizers", make_cluster_probe(args.max_bond_dim)))
    report["results"].append(run_probe("hadamard_test_like_30w", make_hadamard_probe(args.max_bond_dim)))
    report["final_rss_gb"] = rss_gb()

    out_json = out_dir / "mps_probe_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"report written to: {out_json}")


if __name__ == "__main__":
    main()
