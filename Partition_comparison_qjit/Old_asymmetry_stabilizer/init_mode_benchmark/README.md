# Init Mode Benchmark

This folder contains a focused 13-qubit benchmark for the `1x1` and `2x2` partitions with two initialization variants:

- `deep5_uniform_sigma1_lambda1`
  `5` ansatz layers, random angles in `[-pi, pi]`, fixed `sigma=lambda=1`.
- `shallow1_pi2_gaussian`
  `1` ansatz layer, all angles initialized around `pi/2` with Gaussian noise, without the old block-structured warm start.

Generated runs are stored under:

- `Partition_comparison_qjit/init_mode_benchmark/run/<tag>/...`

Submit with:

```bash
python Partition_comparison_qjit/init_mode_benchmark/gen_tasks.py --tag my_tag
sbatch --array=0-3 --export=ALL,TAG=my_tag Partition_comparison_qjit/init_mode_benchmark/submit_array.slurm
```
