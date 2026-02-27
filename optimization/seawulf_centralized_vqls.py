import argparse
import itertools
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax
import pennylane as qml

import importlib
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.reporting import JsonlWriter, make_run_dir, setup_logger


PAULI_MATS = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def load_static_ops(module_name: str):
    return importlib.import_module(module_name)


def init_ansatz_weights(n_qubits: int, layers: int, seed: int):
    key = jax.random.PRNGKey(seed)
    return jax.random.uniform(
        key,
        shape=(layers, n_qubits),
        minval=-math.pi,
        maxval=math.pi,
        dtype=jnp.float64,
    )


def _op_to_data_pauli_word(op, data_wires: List[int]) -> str:
    wire_to_pos = {int(w): i for i, w in enumerate(data_wires)}
    word = ["I"] * len(data_wires)

    def visit(node):
        # qml.prod(...) returns an op-math tree with operands.
        if hasattr(node, "operands"):
            for child in node.operands:
                visit(child)
            return

        name = node.name
        if name == "Identity":
            return
        if name in ("PauliX", "PauliY", "PauliZ"):
            w = int(node.wires[0])
            pos = wire_to_pos[w]
            label = name[-1]  # X / Y / Z
            if word[pos] != "I" and word[pos] != label:
                raise ValueError(f"Unsupported non-Pauli product on wire {w}.")
            word[pos] = label
            return
        raise ValueError(f"Unsupported gate in distributed block decomposition: {name}")

    visit(op)
    return "".join(word)


def _kron_pauli_word(word: str) -> np.ndarray:
    mats = [PAULI_MATS[ch] for ch in word]
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def build_global_lcu_terms_from_distributed(system, data_wires: List[int], logger):
    """
    Convert distributed block decomposition A_ij = sum_m c_ijm G_ijm
    to centralized A = sum_l c_l A_l with global Pauli-string unitaries A_l.
    """
    n_agents = int(system.n)
    n_index = int(round(math.log2(n_agents)))
    if 2**n_index != n_agents:
        raise ValueError(f"system.n={n_agents} is not a power of 2; cannot map to index qubits.")

    # Aggregate coefficient matrices per unique data Pauli word.
    coeff_mats: Dict[str, np.ndarray] = {}
    for i in range(n_agents):
        for j in range(n_agents):
            gates = system.gates_grid[i][j]
            coeffs = system.coeffs[i][j]
            for gate_fn, c in zip(gates, coeffs):
                op = gate_fn()
                data_word = _op_to_data_pauli_word(op, data_wires)
                if data_word not in coeff_mats:
                    coeff_mats[data_word] = np.zeros((n_agents, n_agents), dtype=complex)
                coeff_mats[data_word][i, j] += complex(c)

    # Decompose each index-space coeff matrix on n_index-qubit Pauli basis.
    index_basis = list(itertools.product("IXYZ", repeat=n_index))
    index_pauli_mats = {
        basis: _kron_pauli_word("".join(basis)) for basis in index_basis
    }

    global_coeffs = defaultdict(complex)  # key: full Pauli word over (index+data)
    for data_word, cmat in coeff_mats.items():
        for basis, pmat in index_pauli_mats.items():
            alpha = np.trace(pmat.conj().T @ cmat) / n_agents
            if abs(alpha) > 1e-12:
                full_word = "".join(basis) + data_word
                global_coeffs[full_word] += alpha

    # Pack terms
    terms = []
    for word in sorted(global_coeffs.keys()):
        c = global_coeffs[word]
        if abs(c) <= 1e-12:
            continue
        terms.append({"word": word, "coeff": c})

    # Verify reconstruction against explicit global matrix.
    a_target = np.array(system.get_global_matrix(), dtype=complex)
    a_recon = np.zeros_like(a_target)
    for term in terms:
        a_recon += term["coeff"] * _kron_pauli_word(term["word"])
    rel_err = np.linalg.norm(a_recon - a_target) / max(np.linalg.norm(a_target), 1e-16)
    logger.info(f"LCU translation: L={len(terms)} global terms, relative matrix reconstruction err={rel_err:.3e}")
    if rel_err > 1e-10:
        raise ValueError("Global A_l translation failed verification against distributed global matrix.")

    return terms, n_index


def is_uniform_superposition(state: np.ndarray, atol: float = 1e-10) -> bool:
    dim = state.shape[0]
    idx = int(np.argmax(np.abs(state)))
    if abs(state[idx]) < 1e-14:
        return False
    phased = state * np.conj(state[idx]) / abs(state[idx])
    target = np.ones(dim, dtype=complex) / np.sqrt(dim)
    return np.allclose(phased, target, atol=atol, rtol=0.0)


def apply_pauli_word(word: str, wires: List[int]):
    for ch, w in zip(word, wires):
        if ch == "I":
            continue
        if ch == "X":
            qml.PauliX(wires=w)
        elif ch == "Y":
            qml.PauliY(wires=w)
        elif ch == "Z":
            qml.PauliZ(wires=w)
        else:
            raise ValueError(f"Unknown Pauli label: {ch}")


def recover_solution_vector(x_state: np.ndarray, psi: np.ndarray, b_norm: float):
    psi_norm = np.linalg.norm(psi)
    if psi_norm < 1e-14:
        return np.zeros_like(x_state), 0.0
    scale = b_norm / psi_norm
    return scale * x_state, scale


def l2_errors(x_est: np.ndarray, x_true: np.ndarray):
    overlap = np.vdot(x_est, x_true)
    if abs(overlap) > 1e-14:
        x_est = x_est * np.exp(-1j * np.angle(overlap))
    abs_l2 = np.linalg.norm(x_true - x_est)
    rel_l2 = abs_l2 / max(np.linalg.norm(x_true), 1e-14)
    return abs_l2, rel_l2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static_ops", required=True, help="e.g. problems.static_ops_16agents_Ising")
    ap.add_argument("--out", required=True)
    ap.add_argument("--topology", type=str, default="line")  # compatibility
    ap.add_argument("--system_id", type=int, default=0)      # compatibility
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--decay", type=float, default=0.9999)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument(
        "--ht_diff_method",
        type=str,
        default="best",
        choices=["best", "adjoint", "parameter-shift", "finite-diff", "spsa"],
        help="Differentiation method for Hadamard-test qnodes.",
    )
    args = ap.parse_args()

    jax.config.update("jax_enable_x64", True)
    np.random.seed(args.seed)

    ops = load_static_ops(args.static_ops)
    system = ops.SYSTEM
    data_wires = list(ops.DATA_WIRES)

    paths = make_run_dir(args.out)
    logger = setup_logger(paths.report_txt)
    metrics = JsonlWriter(paths.metrics_jsonl)

    # Build centralized A = sum_l c_l A_l from distributed block data and verify exactness.
    terms, n_index = build_global_lcu_terms_from_distributed(system, data_wires, logger)
    coeffs = np.array([t["coeff"] for t in terms], dtype=complex)
    if np.max(np.abs(np.imag(coeffs))) > 1e-10:
        raise ValueError("Current implementation expects real LCU coefficients for Eq(14-18) real-part estimator.")
    coeffs = np.real(coeffs)

    n_data = len(data_wires)
    n_qubits = n_index + n_data
    sys_wires = list(range(n_qubits))
    anc = n_qubits

    # Global linear system for reference solution and recovery scaling.
    a_global = np.array(system.get_global_matrix(), dtype=complex)
    b_global = np.array(system.get_global_b_vector(), dtype=complex)
    b_norm = float(np.linalg.norm(b_global))
    if b_norm < 1e-14:
        raise ValueError("Global b vector has zero norm.")
    b_state = b_global / b_norm

    # For this Ising distributed problem, global |b> must be uniform superposition,
    # so U is a global Hadamard layer (and U^\dagger = U).
    if not is_uniform_superposition(b_state):
        raise ValueError(
            "This Hadamard-test centralized implementation currently expects uniform |b>, "
            "which is satisfied by problems.static_ops_16agents_Ising."
        )

    true_sol = np.linalg.solve(a_global, b_global)

    logger.info(f"n_index={n_index}, n_data={n_data}, n_global_qubits={n_qubits}, L={len(terms)}")
    logger.info(f"Condition number of A: {np.linalg.cond(a_global):.8e}")
    logger.info(f"||b||_2: {b_norm:.8e}")
    logger.info("Cost is evaluated via Hadamard-test estimators for Eq.(14),(18),(37) and optimized with local C_L.")
    logger.info(f"Hadamard-test diff method: {args.ht_diff_method}")

    def apply_v(weights):
        qml.BasicEntanglerLayers(weights=weights, wires=sys_wires, rotation=qml.RY)

    term_words = [t["word"] for t in terms]

    def apply_a_l(l: int):
        apply_pauli_word(term_words[l], sys_wires)

    # Device for Hadamard tests.
    dev_ht = qml.device("lightning.qubit", wires=n_qubits + 1)

    # State qnode only for post-optimization recovery metrics (not used to build cost).
    dev_state = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev_state, interface="jax")
    def ansatz_state(weights):
        apply_v(weights)
        return qml.state()

    # Prebuild all upper-triangle qnodes: only Re parts are needed for real-coeff CL.
    pair_indices = [(l, lp) for l in range(len(terms)) for lp in range(l, len(terms))]

    beta_re_qnodes = {}
    for l, lp in pair_indices:
        def make_beta_qnode(l=l, lp=lp):
            @qml.qnode(dev_ht, interface="jax", diff_method=args.ht_diff_method)
            def qnode(weights):
                qml.Hadamard(wires=anc)
                apply_v(weights)
                qml.ctrl(lambda: apply_a_l(l), control=anc)()
                qml.ctrl(lambda: apply_a_l(lp), control=anc)()
                qml.Hadamard(wires=anc)
                return qml.expval(qml.PauliZ(anc))

            return qnode

        beta_re_qnodes[(l, lp)] = make_beta_qnode()

    zeta_re_qnodes = {}
    for j in range(n_qubits):
        for l, lp in pair_indices:
            def make_zeta_qnode(j=j, l=l, lp=lp):
                @qml.qnode(dev_ht, interface="jax", diff_method=args.ht_diff_method)
                def qnode(weights):
                    qml.Hadamard(wires=anc)
                    apply_v(weights)
                    qml.ctrl(lambda: apply_a_l(l), control=anc)()
                    # For this problem, U = H^{⊗n}; hence U Z_j U^\dagger = X_j.
                    qml.ctrl(lambda: qml.PauliX(wires=sys_wires[j]), control=anc)()
                    qml.ctrl(lambda: apply_a_l(lp), control=anc)()
                    qml.Hadamard(wires=anc)
                    return qml.expval(qml.PauliZ(anc))

                return qnode

            zeta_re_qnodes[(j, l, lp)] = make_zeta_qnode()

    # Precompute scalar pair weights from Eq.(14),(18) for real coefficients and Re parts.
    # sum_{l,l'} c_l c_{l'} Re[T_ll'] = sum_diag + 2*sum_offdiag
    pair_weights = {}
    for l, lp in pair_indices:
        w = coeffs[l] * coeffs[lp]
        if l != lp:
            w = 2.0 * w
        pair_weights[(l, lp)] = float(w)

    def local_cost_cl(weights):
        # Eq.(14): mu = <psi|psi>
        mu = jnp.array(0.0, dtype=jnp.float64)
        for l, lp in pair_indices:
            b_re = beta_re_qnodes[(l, lp)](weights)
            mu = mu + pair_weights[(l, lp)] * b_re

        mu_safe = jnp.maximum(mu, 1e-18)

        # Eq.(37): delta = (beta + zeta)/2  -> Eq.(18) numerator omega.
        omega = jnp.array(0.0, dtype=jnp.float64)
        for j in range(n_qubits):
            zsum_j = jnp.array(0.0, dtype=jnp.float64)
            for l, lp in pair_indices:
                z_re = zeta_re_qnodes[(j, l, lp)](weights)
                zsum_j = zsum_j + pair_weights[(l, lp)] * z_re
            delta_sum_j = 0.5 * (mu + zsum_j)
            omega = omega + delta_sum_j
        omega = omega / float(n_qubits)

        # Eq.(6): C_L = C_L_tilde / <psi|psi> = 1 - omega/mu
        cl = 1.0 - (omega / mu_safe)
        return jnp.real(cl)

    loss_and_grad = jax.value_and_grad(local_cost_cl)

    layers = n_qubits
    params = init_ansatz_weights(n_qubits=n_qubits, layers=layers, seed=args.seed)

    lr_schedule = optax.exponential_decay(
        init_value=args.lr,
        transition_steps=1,
        decay_rate=args.decay,
        staircase=False,
    )
    optimizer = optax.adam(lr_schedule)
    opt_state = optimizer.init(params)

    loss_hist = []
    rel_l2_hist = []
    residual_hist = []
    t0 = time.time()

    def evaluate(weights):
        cl_val = float(np.array(local_cost_cl(weights)))
        x_state = np.array(ansatz_state(weights))
        psi = a_global @ x_state
        x_rec, scale = recover_solution_vector(x_state, psi, b_norm)
        abs_l2, rel_l2 = l2_errors(x_rec, true_sol)
        residual = np.linalg.norm(a_global @ x_rec - b_global) / b_norm
        return {
            "cl": cl_val,
            "scale": float(scale),
            "l2_abs": float(abs_l2),
            "l2_rel": float(rel_l2),
            "residual": float(residual),
            "x_recovered": x_rec,
        }

    init_ev = evaluate(params)
    logger.info(
        "[Init] C_L=%.5e | relL2=%.5e | resid=%.5e"
        % (init_ev["cl"], init_ev["l2_rel"], init_ev["residual"])
    )

    best_rel_l2 = init_ev["l2_rel"]
    best_x = init_ev["x_recovered"]
    best_epoch = 0

    loss_hist.append(init_ev["cl"])
    rel_l2_hist.append(init_ev["l2_rel"])
    residual_hist.append(init_ev["residual"])

    for ep in range(1, args.epochs + 1):
        _, grads = loss_and_grad(params)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)

        ev = evaluate(params)
        loss_hist.append(ev["cl"])
        rel_l2_hist.append(ev["l2_rel"])
        residual_hist.append(ev["residual"])

        if ev["l2_rel"] < best_rel_l2:
            best_rel_l2 = ev["l2_rel"]
            best_x = ev["x_recovered"]
            best_epoch = ep

        if (ep % args.log_every) == 0 or ep == 1:
            metrics.write(
                {
                    "epoch": ep,
                    "loss_local_cl": ev["cl"],
                    "recover_scale": ev["scale"],
                    "l2_abs": ev["l2_abs"],
                    "l2_rel": ev["l2_rel"],
                    "residual_rel": ev["residual"],
                    "lr": float(args.lr),
                    "wall_s": time.time() - t0,
                }
            )
            logger.info(
                "[Epoch %05d] C_L=%.5e | relL2=%.5e | resid=%.5e"
                % (ep, ev["cl"], ev["l2_rel"], ev["residual"])
            )

    metrics.close()

    analysis_path = paths.run_dir / "analysis.txt"
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write("Centralized VQLS (Hadamard-test CL) final analysis\n")
        f.write(f"Best epoch (relL2): {best_epoch}\n")
        f.write(f"Best relative L2 error: {best_rel_l2:.8e}\n")
        f.write(f"Final relative L2 error: {rel_l2_hist[-1]:.8e}\n")
        f.write(f"Final relative residual ||Ax-b||/||b||: {residual_hist[-1]:.8e}\n")
        f.write("\nRecovered solution (best):\n")
        f.write(np.array2string(best_x, precision=6, suppress_small=False))
        f.write("\n\nTrue linear-algebra solution:\n")
        f.write(np.array2string(true_sol, precision=6, suppress_small=False))

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kappa_int = int(np.rint(np.linalg.cond(a_global)))
    title = f"kappa≈{kappa_int} | lr0={args.lr:g} | centralized_hadamard_CL"

    xs = np.arange(len(rel_l2_hist))
    plt.figure()
    plt.plot(xs, rel_l2_hist, label="relative L2 error")
    plt.plot(xs, residual_hist, label="relative residual")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(paths.fig_diff, dpi=200, bbox_inches="tight")
    plt.close()

    xs2 = np.arange(len(loss_hist))
    plt.figure()
    plt.plot(xs2, loss_hist, label="C_L (Hadamard-test)")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(paths.fig_loss, dpi=200, bbox_inches="tight")
    plt.close()

    np.savez(
        paths.artifacts_npz,
        cl=np.array(loss_hist),
        rel_l2=np.array(rel_l2_hist),
        residual=np.array(residual_hist),
    )

    logger.info(f"Post-analysis written to: {analysis_path}")
    logger.info("Finished. Outputs written to: " + str(paths.run_dir))


if __name__ == "__main__":
    main()
