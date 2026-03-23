from __future__ import annotations

from Partition_comparison_qjit.New_stabilizer.benchmark_13q_xzx_fresh_common import (
    B_PREP_HADAMARD,
    B_STATE_ALL_PLUS,
    DEFAULT_EPSILON,
    build_partition_namespace,
)


globals().update(
    build_partition_namespace(3, epsilon=DEFAULT_EPSILON, b_state_kind=B_STATE_ALL_PLUS, b_prep_kind=B_PREP_HADAMARD)
)
