## Catalyst Env Repair Log

Date: 2026-03-20

Environment:
- Conda env: `pennylane`
- Python: `/gpfs/home/tonshen/.conda/envs/pennylane/bin/python`

### Goal

Repair the Catalyst gradient compilation failure observed in the 13-qubit partition benchmark workload.

### Dependency State Before Changes

Key package versions before any repair:

| Package | Version |
|---|---|
| `pennylane` | `0.44.0` |
| `pennylane_catalyst` | `0.14.0` |
| `jax` | `0.7.1` |
| `jaxlib` | `0.7.1` |
| `xdsl` | `0.55.4` |
| `xdsl-jax` | `0.1.1` |
| `numpy` | `2.4.3` |
| `scipy` | `1.17.1` |

Snapshot:
- [pennylane_env_before_freeze.txt](/gpfs/home/tonshen/Seawulf_simulation/tmp/env_repair/pennylane_env_before_freeze.txt)

### Reproduction Results

1. Minimal Catalyst gradient test:
   - Script: [catalyst_repro_min.py](/gpfs/home/tonshen/Seawulf_simulation/tmp/catalyst_repro_min.py)
   - Result before changes: passed
   - Meaning: the env was not globally broken for all `qml.qjit + catalyst.grad` usage.

2. Real workload reproduction:
   - Command target: [seawulf_partition_comparison_qjit.py](/gpfs/home/tonshen/Seawulf_simulation/Partition_comparison_qjit/seawulf_partition_comparison_qjit.py)
   - Static ops: [static_ops_cluster13_real_1x1.py](/gpfs/home/tonshen/Seawulf_simulation/Partition_comparison_qjit/1b1/static_ops_cluster13_real_1x1.py)
   - Result before changes: failed during Catalyst gradient compilation with:
     - `CompileError: Illegal updateAnalysis prev:{[-1]:Pointer} new: {[-1]:Integer}`

### Dependency Changes Applied

#### Change 1: Upgrade PennyLane/Catalyst to latest patch releases

Applied:

```bash
/gpfs/home/tonshen/.conda/envs/pennylane/bin/pip install --upgrade --no-deps 'pennylane==0.44.1' 'pennylane-catalyst==0.14.1'
```

Actual persistent version changes:

| Package | Before | After |
|---|---:|---:|
| `pennylane` | `0.44.0` | `0.44.1` |
| `pennylane_catalyst` | `0.14.0` | `0.14.1` |

Patch-upgrade diff snapshot:
- [pennylane_env_after_patch_freeze.txt](/gpfs/home/tonshen/Seawulf_simulation/tmp/env_repair/pennylane_env_after_patch_freeze.txt)

Result:
- Minimal Catalyst gradient test: passed
- Real 13q benchmark workload: still failed with the same Catalyst compile error

#### Change 2: Experimental JAX/JAXlib downgrade to `0.7.0`

Applied temporarily:

```bash
/gpfs/home/tonshen/.conda/envs/pennylane/bin/pip install --upgrade --no-deps 'jax==0.7.0' 'jaxlib==0.7.0'
```

Result:
- Catalyst import broke immediately
- Observed import failure:
  - `ImportError: cannot import name '_jaxpr_replicas' from 'jax._src.interpreters.pxla'`

Conclusion:
- This downgrade is incompatible with the installed Catalyst wheel.

#### Change 3: Roll back JAX/JAXlib to Catalyst-required versions

Applied:

```bash
/gpfs/home/tonshen/.conda/envs/pennylane/bin/pip install --upgrade --no-deps 'jax==0.7.1' 'jaxlib==0.7.1'
```

Final versions:

| Package | Final Version |
|---|---:|
| `pennylane` | `0.44.1` |
| `pennylane_catalyst` | `0.14.1` |
| `jax` | `0.7.1` |
| `jaxlib` | `0.7.1` |
| `xdsl` | `0.55.4` |
| `xdsl-jax` | `0.1.1` |

Final snapshot:
- [pennylane_env_final_freeze.txt](/gpfs/home/tonshen/Seawulf_simulation/env_repair/pennylane_env_final_freeze.txt)

### Catalyst Wheel Constraints

The installed `pennylane_catalyst 0.14.1` wheel declares these exact relevant requirements:

- `pennylane>=0.44.1`
- `jax==0.7.1`
- `jaxlib==0.7.1`
- `xdsl==0.55.4`
- `xdsl-jax==0.1.1`

That means there is no additional safe dependency shuffle available inside this env without violating the wheel's own pinned compatibility contract.

### Final Verification

1. Minimal Catalyst gradient test after final repair state:
   - passed
   - output:

```text
f 0.9924450321351935
g -0.12269009002431534
```

2. Real 13q benchmark workload after final repair state:
   - still fails inside Catalyst gradient compilation
   - observed summary:

```text
Catalyst gradient compilation failed for this workload; falling back to jax.grad while keeping the qjit'd forward loss.
CompileError: Illegal updateAnalysis prev:{[-1]:Pointer} new: {[-1]:Integer}
```

Workload output folder from the final verification run:
- [1b1_repro](/gpfs/home/tonshen/Seawulf_simulation/tmp/repro_env_final/1b1_repro)

### Conclusion

The conda env is now on the newest safe PennyLane/Catalyst patch pair and back on the Catalyst-required JAX stack. This did **not** repair the real benchmark's Catalyst gradient compilation failure.

Current diagnosis:
- the env is healthy enough for simple Catalyst gradients
- the remaining failure is a workload-specific Catalyst compiler bug on the pinned supported stack, not a general broken-env issue

### Workflow Alignment

I also checked the scripts referenced as the formal workflow:

- [optimization/met_line_tracking.py](/gpfs/home/tonshen/Seawulf_simulation/optimization/met_line_tracking.py)
- [optimization/seawulf_cat_line_tracking_nodispatch_2x2_cluster12_stabilizer.py](/gpfs/home/tonshen/Seawulf_simulation/optimization/seawulf_cat_line_tracking_nodispatch_2x2_cluster12_stabilizer.py)

Important clarification:
- `optimization/met_line_tracking.py` does **not** use Catalyst gradients; it uses `jax.jit(jax.value_and_grad(...))`.
- `optimization/seawulf_cat_line_tracking_nodispatch_2x2_cluster12_stabilizer.py` in this env still hits the same Catalyst compile failure as the 13q partition runner.

To align the stable workflow with the formal optimizer, I updated these scripts:

- [Partition_comparison_qjit/seawulf_partition_comparison_qjit.py](/gpfs/home/tonshen/Seawulf_simulation/Partition_comparison_qjit/seawulf_partition_comparison_qjit.py)
- [optimization/seawulf_cat_line_tracking_nodispatch_2x2_cluster12_stabilizer.py](/gpfs/home/tonshen/Seawulf_simulation/optimization/seawulf_cat_line_tracking_nodispatch_2x2_cluster12_stabilizer.py)

New behavior:
- try `catalyst.grad(...)` first
- if Catalyst compilation fails, switch the optimization backend to `jax.jit(jax.value_and_grad(total_loss_fn))`
- keep the qjit'd forward loss path available

This does not eliminate the Catalyst compiler bug, but it gives a stable compiled fallback that matches the successful optimization structure used in `met_line_tracking.py`.

### Persistent Dependency Changes

Only these dependency changes remain in the env at the end of this repair task:

| Package | Old | New |
|---|---:|---:|
| `pennylane` | `0.44.0` | `0.44.1` |
| `pennylane_catalyst` | `0.14.0` | `0.14.1` |

The temporary `jax/jaxlib 0.7.0` downgrade was fully reverted.

### 2026-03-20 Follow-Up: qjit+Catalyst Reference Workflow

User clarification:
- the intended reference workflow is [optimization/seawulf_cat_line_tracking_nodispatch.py](/gpfs/home/tonshen/Seawulf_simulation/optimization/seawulf_cat_line_tracking_nodispatch.py), not `met_line_tracking.py`

Actions taken:
- restored the 13q partition runner to match the same top-level pattern:
  - `compute_loss`: `@qml.qjit`
  - `compute_grad`: `@qml.qjit` calling `catalyst.grad(...)`
- added a plain loss entry point in [objective/builder_cluster_nodispatch.py](/gpfs/home/tonshen/Seawulf_simulation/objective/builder_cluster_nodispatch.py) so the runner can fall back cleanly if Catalyst gradient compilation fails
- kept the qjit loss path active even in fallback mode

Root cause isolation:
- the failure was not the outer optimizer wrapper
- the failure was inside the cluster Hadamard-test circuits, specifically the `qml.adjoint(function)` path used for the variational ansatz inverse in `OMEGA/DELTA/ZETA/TAU`
- isolated term tests showed `OMEGA` failed before the patch while `BETA` already compiled, which narrowed the issue to the controlled ansatz-adjoint path

Fix applied:
- replaced the function-level ansatz adjoint with an explicit inverse circuit in [objective/circuits_cluster_nodispatch.py](/gpfs/home/tonshen/Seawulf_simulation/objective/circuits_cluster_nodispatch.py)
- the optimizer in [Partition_comparison_qjit/seawulf_partition_comparison_qjit.py](/gpfs/home/tonshen/Seawulf_simulation/Partition_comparison_qjit/seawulf_partition_comparison_qjit.py) still matches the reference `qjit(loss) + qjit(catalyst.grad)` workflow
- the gradient-only fallback remains as a safety net but is no longer triggered on the verified `2x2` benchmark

Verification:
- isolated Catalyst gradients for all five local terms now compile:
  - `OMEGA`
  - `DELTA`
  - `ZETA`
  - `TAU`
  - `BETA`
- full 13q `2x2` benchmark now completes initial Catalyst gradient compilation successfully without fallback
- the first logged optimization step completed after compilation with `wall_s = 0.82`

Smoke test:
- [qjit_cluster_smoke_2b2_fallback](/gpfs/home/tonshen/Seawulf_simulation/tmp/qjit_cluster_smoke_2b2_fallback)

Observed behavior from the repaired smoke test:
- initial qjit forward loss: about `19.18 s`
- initial Catalyst gradient compile: about `88.45 s`
- first logged epoch after compilation completes successfully
- no fallback message appears

Back-to-back qjit loss timing on the same `2x2` benchmark:
- first loss call: about `19.66 s`
- second loss call: about `0.044 s`

This confirms the qjit forward-loss cache behavior expected from the reference workflow.

Dependency changes in this follow-up:
- none
