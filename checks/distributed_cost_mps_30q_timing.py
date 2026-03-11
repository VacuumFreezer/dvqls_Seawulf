import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import jax.numpy as jnp
import numpy as np
import pennylane as qml
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The installed Catalyst/JAX combination is broken in this env. Replace qjit with a
# no-op decorator so we can import and execute the same builder algebra in Python.
def _identity_qjit(fn=None, *args, **kwargs):
    del args, kwargs
    if fn is None:
        return lambda wrapped: wrapped
    return fn


qml.qjit = _identity_qjit

import objective.builder_cat_nodispatch as ib
from problems.static_ops_2x2_cluster30 import SYSTEM


TOPOLOGY_LINE_2 = {0: [1], 1: [0]}


def rss_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)


def make_mps_dev_cpu(max_bond_dim: int, c_dtype):
    cache = {}

    def dev_cpu(nwires: int, device_name: str = 'lightning.qubit'):
        del device_name
        key = (int(nwires), int(max_bond_dim), np.dtype(c_dtype).name)
        if key not in cache:
            cache[key] = qml.device(
                'default.tensor',
                wires=int(nwires),
                method='mps',
                max_bond_dim=int(max_bond_dim),
                c_dtype=c_dtype,
            )
        return cache[key]

    return dev_cpu


def initialize_params(system, n_qubits: int, layers: int, seed: int):
    rng = np.random.default_rng(seed)
    n_agents = int(system.n)
    params = {'alpha': [], 'beta': [], 'sigma': [], 'lambda': [], 'b_norm': []}

    for sys_id in range(n_agents):
        local_b_norms = system.get_local_b_norms(sys_id)

        row_alpha, row_beta = [], []
        row_sigma, row_lambda = [], []
        row_bnorm = []

        for agent_id in range(n_agents):
            alpha = jnp.asarray(
                rng.uniform(-np.pi, np.pi, size=(layers, n_qubits)), dtype=jnp.float32
            )
            beta = jnp.asarray(
                rng.uniform(-np.pi, np.pi, size=(layers, n_qubits)), dtype=jnp.float32
            )
            sigma = jnp.asarray(rng.uniform(0.0, 2.0), dtype=jnp.float32)
            lam = jnp.asarray(rng.uniform(0.0, 2.0), dtype=jnp.float32)
            b_norm = jnp.asarray(float(local_b_norms[agent_id]), dtype=jnp.float32)

            row_alpha.append(alpha)
            row_beta.append(beta)
            row_sigma.append(sigma)
            row_lambda.append(lam)
            row_bnorm.append(b_norm)

        params['alpha'].append(row_alpha)
        params['beta'].append(row_beta)
        params['sigma'].append(row_sigma)
        params['lambda'].append(row_lambda)
        params['b_norm'].append(row_bnorm)

    return params


def estimate_qnode_calls(system, topology):
    details = []
    total = 0
    for sys_id in range(int(system.n)):
        for agent_id in range(int(system.n)):
            L = len(system.gates_grid[sys_id][agent_id])
            degree = len(topology[agent_id])
            m = degree + 1
            qnodes = (L * L) + ((m + 1) * L) + m + (m * (m - 1) // 2)
            total += qnodes
            details.append({
                'sys_id': sys_id,
                'agent_id': agent_id,
                'L': L,
                'degree': degree,
                'qnode_calls': qnodes,
            })
    return total, details


def eval_total_loss_python(current_params):
    total = jnp.asarray(0.0, dtype=jnp.float32)
    n_entries = len(ib._ENTRY_SYS)
    for entry_idx in range(n_entries):
        agent_params = ib._build_agent_params(entry_idx, current_params, interface='jax')
        total = total + ib._combine_local_loss(entry_idx, agent_params, interface='jax')
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--layers', type=int, default=1)
    ap.add_argument('--max_bond_dim', type=int, default=4)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_qubits = len(SYSTEM.data_wires)
    layers = int(args.layers)

    ib._DEV.clear()
    ib.dev_cpu = make_mps_dev_cpu(args.max_bond_dim, np.complex64)

    params = initialize_params(SYSTEM, n_qubits=n_qubits, layers=layers, seed=args.seed)
    total_qnode_calls, per_entry = estimate_qnode_calls(SYSTEM, TOPOLOGY_LINE_2)

    t0 = time.time()
    ib.prebuild_local_evals(
        SYSTEM,
        TOPOLOGY_LINE_2,
        n_input_qubit=n_qubits,
        diff_method='parameter-shift',
        interface='jax',
    )
    prebuild_s = time.time() - t0

    rss_before = rss_gb()
    t1 = time.time()
    loss = eval_total_loss_python(params)
    loss_s = time.time() - t1
    rss_after = rss_gb()

    report = {
        'system_name': getattr(SYSTEM, 'name', 'unknown'),
        'n_agents': int(SYSTEM.n),
        'n_local_qubits': n_qubits,
        'layers': layers,
        'device_name': 'default.tensor',
        'method': 'mps',
        'c_dtype': 'complex64',
        'max_bond_dim': int(args.max_bond_dim),
        'diff_method': 'parameter-shift',
        'seed': int(args.seed),
        'prebuild_local_evals_s': prebuild_s,
        'loss_eval_s': loss_s,
        'loss_value': float(np.asarray(loss)),
        'rss_before_gb': rss_before,
        'rss_after_gb': rss_after,
        'rss_delta_gb': rss_after - rss_before,
        'total_qnode_calls_per_loss': total_qnode_calls,
        'avg_time_per_qnode_s': loss_s / max(total_qnode_calls, 1),
        'per_entry': per_entry,
    }

    out_json = out_dir / 'distributed_cost_mps_30q_timing_report.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f'report written to: {out_json}')


if __name__ == '__main__':
    main()
