import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path


def _case_worker(queue, fn):
    try:
        queue.put(fn())
    except Exception as exc:
        queue.put({
            'ok': False,
            'error': repr(exc),
        })


def _run_with_timeout(fn, timeout_s: int):
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    proc = ctx.Process(target=_case_worker, args=(q, fn))
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        return {
            'ok': False,
            'timeout': True,
            'timeout_s': timeout_s,
        }
    if not q.empty():
        return q.get()
    return {
        'ok': False,
        'error': f'child process exited with code {proc.exitcode} and produced no result',
    }


def case_jax_python_parameter_shift_30w():
    import numpy as np
    import jax
    import jax.numpy as jnp
    import pennylane as qml
    import psutil

    jax.config.update('jax_enable_x64', False)

    n = 30
    max_bond_dim = 4
    dev = qml.device(
        'default.tensor',
        wires=n,
        method='mps',
        max_bond_dim=max_bond_dim,
        c_dtype=np.complex64,
    )

    @qml.qnode(dev, interface='jax-python', diff_method='parameter-shift')
    def qnode(params):
        for w in range(n):
            qml.Hadamard(wires=w)
        for a, b in zip(range(n - 1), range(1, n)):
            qml.CZ(wires=[a, b])
        qml.RY(params[0], wires=0)
        qml.RY(params[1], wires=14)
        qml.RY(params[2], wires=29)
        return (
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1) @ qml.PauliZ(2)),
            qml.expval(qml.PauliZ(12) @ qml.PauliX(13) @ qml.PauliZ(14)),
            qml.expval(qml.PauliZ(27) @ qml.PauliX(28) @ qml.PauliZ(29)),
        )

    def cost_fn(params):
        vals = qnode(params)
        return (vals[0] + vals[1] + vals[2]) / 3.0

    x = jnp.array([0.17, -0.23, 0.31], dtype=jnp.float32)
    rss0 = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    t0 = time.time()
    val = cost_fn(x)
    grad = jax.grad(cost_fn)(x)
    elapsed = time.time() - t0
    rss1 = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)

    return {
        'ok': True,
        'case': 'jax_python_parameter_shift_30w',
        'elapsed_s': elapsed,
        'value': float(val),
        'grad': np.asarray(grad).tolist(),
        'rss_before_gb': rss0,
        'rss_after_gb': rss1,
        'rss_delta_gb': rss1 - rss0,
        'dtype': 'complex64',
        'max_bond_dim': max_bond_dim,
    }


def case_jax_parameter_shift_30w():
    import numpy as np
    import jax
    import jax.numpy as jnp
    import pennylane as qml

    jax.config.update('jax_enable_x64', False)

    n = 30
    dev = qml.device(
        'default.tensor',
        wires=n,
        method='mps',
        max_bond_dim=4,
        c_dtype=np.complex64,
    )

    @qml.qnode(dev, interface='jax', diff_method='parameter-shift')
    def cost_fn(params):
        for w in range(n):
            qml.Hadamard(wires=w)
        for a, b in zip(range(n - 1), range(1, n)):
            qml.CZ(wires=[a, b])
        qml.RY(params[0], wires=0)
        qml.RY(params[1], wires=14)
        qml.RY(params[2], wires=29)
        return qml.expval(qml.PauliZ(0) @ qml.PauliX(1) @ qml.PauliZ(2))

    x = jnp.array([0.11, -0.07, 0.05], dtype=jnp.float32)
    t0 = time.time()
    val = cost_fn(x)
    grad = jax.grad(cost_fn)(x)
    elapsed = time.time() - t0

    return {
        'ok': True,
        'case': 'jax_parameter_shift_30w',
        'elapsed_s': elapsed,
        'value': float(val),
        'grad': np.asarray(grad).tolist(),
    }


def case_jax_adjoint_small():
    import numpy as np
    import jax
    import jax.numpy as jnp
    import pennylane as qml

    jax.config.update('jax_enable_x64', False)

    dev = qml.device(
        'default.tensor',
        wires=4,
        method='mps',
        max_bond_dim=4,
        c_dtype=np.complex64,
    )

    @qml.qnode(dev, interface='jax-python', diff_method='adjoint')
    def cost_fn(x):
        qml.RY(x, wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(1))

    x = jnp.array(0.3, dtype=jnp.float32)
    t0 = time.time()
    try:
        val = cost_fn(x)
        grad = jax.grad(cost_fn)(x)
        return {
            'ok': True,
            'case': 'jax_adjoint_small',
            'elapsed_s': time.time() - t0,
            'value': float(val),
            'grad': float(grad),
        }
    except Exception as exc:
        return {
            'ok': False,
            'case': 'jax_adjoint_small',
            'elapsed_s': time.time() - t0,
            'error': repr(exc),
        }


def case_catalyst_qjit_small():
    import numpy as np
    import jax.numpy as jnp
    import pennylane as qml

    dev = qml.device(
        'default.tensor',
        wires=2,
        method='mps',
        max_bond_dim=4,
        c_dtype=np.complex64,
    )

    @qml.qnode(dev, interface='jax', diff_method='parameter-shift')
    def circuit(x):
        qml.RY(x, wires=0)
        return qml.expval(qml.PauliZ(0))

    try:
        @qml.qjit
        def wrapped(x):
            return circuit(x)

        t0 = time.time()
        val = wrapped(jnp.array(0.2, dtype=jnp.float32))
        return {
            'ok': True,
            'case': 'catalyst_qjit_small',
            'elapsed_s': time.time() - t0,
            'value': float(val),
        }
    except Exception as exc:
        return {
            'ok': False,
            'case': 'catalyst_qjit_small',
            'error': repr(exc),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--timeout_s', type=int, default=120)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import pennylane as qml
    import numpy as np
    report = {
        'python_pid': os.getpid(),
        'python_executable': os.environ.get('CONDA_PREFIX', '') + '/bin/python' if os.environ.get('CONDA_PREFIX') else None,
        'timeout_s_per_case': int(args.timeout_s),
        'pennylane_version': qml.__version__,
        'notes': [
            'All cases use default.tensor with method=mps.',
            'All cases request c_dtype=numpy.complex64.',
            'All cases request max_bond_dim=4.',
        ],
        'results': [],
    }

    cases = [
        ('jax_python_parameter_shift_30w', case_jax_python_parameter_shift_30w),
        ('jax_parameter_shift_30w', case_jax_parameter_shift_30w),
        ('jax_adjoint_small', case_jax_adjoint_small),
        ('catalyst_qjit_small', case_catalyst_qjit_small),
    ]

    for name, fn in cases:
        t0 = time.time()
        result = _run_with_timeout(fn, args.timeout_s)
        result.setdefault('case', name)
        result['wall_s_parent'] = time.time() - t0
        report['results'].append(result)

    out_json = out_dir / 'mps_gradient_stack_probe_report.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f'report written to: {out_json}')


if __name__ == '__main__':
    main()
