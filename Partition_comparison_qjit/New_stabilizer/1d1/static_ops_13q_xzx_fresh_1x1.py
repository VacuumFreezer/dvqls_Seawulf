from __future__ import annotations

from Partition_comparison_qjit.New_stabilizer.benchmark_13q_xzx_fresh_common import (
    DEFAULT_EPSILON,
    build_partition_namespace,
)


globals().update(build_partition_namespace(0, epsilon=DEFAULT_EPSILON))
