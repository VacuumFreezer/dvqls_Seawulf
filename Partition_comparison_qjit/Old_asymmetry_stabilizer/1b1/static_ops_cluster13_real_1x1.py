from __future__ import annotations

from Partition_comparison_qjit.Old_asymmetry_stabilizer.benchmark_13q_real_cluster_common import (
    DEFAULT_EPSILON,
    build_partition_namespace,
)


globals().update(build_partition_namespace(0, epsilon=DEFAULT_EPSILON))
