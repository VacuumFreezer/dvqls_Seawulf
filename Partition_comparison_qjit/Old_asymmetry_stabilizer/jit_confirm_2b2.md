# JIT Confirmation on 2x2 Benchmark

Date: 2026-03-20

The optimization path in:

- [seawulf_partition_comparison_qjit.py](/gpfs/home/tonshen/Seawulf_simulation/Partition_comparison_qjit/seawulf_partition_comparison_qjit.py)
- [seawulf_cat_line_tracking_nodispatch_2x2_cluster12_stabilizer.py](/gpfs/home/tonshen/Seawulf_simulation/optimization/seawulf_cat_line_tracking_nodispatch_2x2_cluster12_stabilizer.py)

now matches the `optimization/met_line_tracking.py` style:

```python
loss_and_grad = jax.jit(jax.value_and_grad(total_loss_fn))
```

## Direct Timing Check

Using the 13-qubit real-cluster `2x2` partition benchmark:

- [static_ops_cluster13_real_2x2.py](/gpfs/home/tonshen/Seawulf_simulation/Partition_comparison_qjit/2b2/static_ops_cluster13_real_2x2.py)

timing of repeated calls on the same flattened parameters was:

- call 1: `12.6535 s`
- call 2: `0.9252 s`
- call 3: `0.9338 s`

with the same loss value on every call:

- loss: `2.93720704e-01`

This confirms the expected JIT behavior: the first call pays the compilation cost, while later calls reuse the compiled program and are much faster.

## Full 2x2 Benchmark Run

Run output:

- [repro_2b2_metstyle_confirm](/gpfs/home/tonshen/Seawulf_simulation/tmp/repro_2b2_metstyle_confirm)

Observed timings from the benchmark log:

- initial loss+gradient evaluation: `12.71 s`
- epoch 1 log interval: `1.56 s`
- epoch 2 log interval: `1.04 s`
- epoch 3 log interval: `0.92 s`

These timings are consistent with the direct timing check and with the behavior seen in `optimization/met_line_tracking.py`.
