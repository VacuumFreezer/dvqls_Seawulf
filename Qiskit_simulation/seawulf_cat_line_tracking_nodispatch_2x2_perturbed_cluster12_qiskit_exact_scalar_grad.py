#!/usr/bin/env python3
"""Wrapper for the 12-qubit perturbed-cluster Qiskit benchmark with exact sigma/lambda gradients."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "--static_ops" not in sys.argv:
    sys.argv.extend(["--static_ops", "Qiskit_simulation.static_ops_2x2_perturbed_cluster12_qiskit"])

from Qiskit_simulation.seawulf_cat_line_tracking_nodispatch_2x2_cluster30_qiskit_exact_scalar_grad import main


if __name__ == "__main__":
    main()
