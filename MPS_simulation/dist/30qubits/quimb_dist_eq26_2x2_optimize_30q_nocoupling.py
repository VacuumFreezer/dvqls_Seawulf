#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MPS_simulation.dist.quimb_dist_eq26_2x2_optimize_nocompress import main as shared_main  # noqa: E402


def main() -> None:
    argv = list(sys.argv[1:])
    if "--global-qubits" not in argv:
        argv.extend(["--global-qubits", "30"])
    if "--local-qubits" not in argv:
        argv.extend(["--local-qubits", "29"])
    if "--j-coupling" not in argv:
        argv.extend(["--j-coupling", "0.0"])
    shared_main(argv)


if __name__ == "__main__":
    main()
