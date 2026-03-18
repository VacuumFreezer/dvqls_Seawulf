#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimization.seawulf_cat_line_tracking_nodispatch_2x2_cluster12_stabilizer import main as shared_main


def main():
    argv = list(sys.argv[1:])
    if "--ansatz" not in argv:
        argv.extend(["--ansatz", "brickwall_ry_cz"])
    if "--layers" not in argv:
        argv.extend(["--layers", "1"])
    shared_main(argv)


if __name__ == "__main__":
    main()
