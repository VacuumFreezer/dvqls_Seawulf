"""30-qubit cluster-stabilizer wrapper around the shared stabilizer optimizer."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimization.seawulf_cat_line_tracking_nodispatch_2x2_cluster12_stabilizer import main


if __name__ == "__main__":
    main()
