from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pennylane as qml
import pennylane.numpy as pnp
import jax.numpy as jnp

from .circuits_cat_nodispatch import make_term_bundle


_DEV: Dict[Tuple[str, int], qml.Device] = {}


def dev_cpu(nwires: int, device_name: str = "lightning.qubit"):
    """Reuse a single device per (name, wire-count)."""
    key = (str(device_name), int(nwires))
    if key not in _DEV:
        _DEV[key] = qml.device(device_name, wires=int(nwires))
    return _DEV[key]


# ----------------------------------------------------------------------
# Flat prebuild tables (no LOCAL_SPECS object dispatch)
# ----------------------------------------------------------------------
_ENTRY_SYS: List[int] = []
_ENTRY_AGENT: List[int] = []
_ENTRY_NEIGHBORS: List[Tuple[int, ...]] = []
_ENTRY_DEGREE: List[int] = []
_ENTRY_COEFF: List[Any] = []
_ENTRY_CVEC: List[Any] = []
_ENTRY_TERMS: List[Dict[str, Any]] = []
_ENTRY_L: List[int] = []


def _as_float_array(x, interface: str):
    if interface == "jax":
        return jnp.asarray(x, dtype=jnp.float64)
    return pnp.asarray(x, dtype=float)


def _zeros_1d(size: int, interface: str):
    if interface == "jax":
        return jnp.zeros((int(size),), dtype=jnp.float64)
    return pnp.zeros((int(size),), dtype=float)


def _zeros_2d(n0: int, n1: int, interface: str):
    if interface == "jax":
        return jnp.zeros((int(n0), int(n1)), dtype=jnp.float64)
    return pnp.zeros((int(n0), int(n1)), dtype=float)


def _build_agent_params(entry_idx: int, current_params, *, interface="jax") -> Dict[str, Any]:
    sys_id = _ENTRY_SYS[entry_idx]
    agent_id = _ENTRY_AGENT[entry_idx]
    neigh = _ENTRY_NEIGHBORS[entry_idx]

    alpha_self = current_params["alpha"][sys_id][agent_id]
    beta_self = current_params["beta"][sys_id][agent_id]
    sigma_self = current_params["sigma"][sys_id][agent_id]
    lam_self = current_params["lambda"][sys_id][agent_id]
    b_norm = current_params["b_norm"][sys_id][agent_id]

    betas = [beta_self]
    lams = [lam_self]
    for n_col in neigh:
        betas.append(current_params["beta"][sys_id][n_col])
        lams.append(current_params["lambda"][sys_id][n_col])

    if interface == "jax":
        beta_vec = jnp.stack([jnp.asarray(x) for x in betas], axis=0)
        lam_vec = jnp.stack([jnp.asarray(x) for x in lams], axis=0)
    else:
        beta_vec = pnp.stack(betas, axis=0)
        lam_vec = pnp.stack(lams, axis=0)

    return {
        "alpha": alpha_self,
        "beta_vec": beta_vec,
        "lam_vec": lam_vec,
        "sigma": sigma_self,
        "b_norm": b_norm,
    }


def _eval_terms(entry_idx: int, alphas, beta_vec, *, interface="jax"):
    coeff = _ENTRY_COEFF[entry_idx]
    terms = _ENTRY_TERMS[entry_idx]
    L = int(_ENTRY_L[entry_idx])
    m = int(beta_vec.shape[0])

    omega = terms["OMEGA"]
    delta = terms["DELTA"]
    zeta = terms["ZETA"]
    tau = terms["TAU"]
    beta = terms["BETA"]

    tau_re = _as_float_array(0.0, interface)
    beta_re = _as_float_array(0.0, interface)

    for l in range(L):
        tau_re = tau_re + coeff[l] * tau.eval_re(alphas, l)

    for li in range(L):
        wi = coeff[li]
        for lj in range(L):
            beta_re = beta_re + (wi * coeff[lj]) * beta.eval_re(alphas, li, lj)

    zeta_vec = _zeros_1d(m, interface)
    delta_re_vec = _zeros_1d(m, interface)

    for idx in range(m):
        bk = beta_vec[idx]
        z_sum = _as_float_array(0.0, interface)
        for l in range(L):
            z_sum = z_sum + coeff[l] * zeta.eval_re(alphas, bk, l)

        d_re = delta.eval_re(bk)
        if interface == "jax":
            zeta_vec = zeta_vec.at[idx].set(z_sum)
            delta_re_vec = delta_re_vec.at[idx].set(d_re)
        else:
            zeta_vec[idx] = z_sum
            delta_re_vec[idx] = d_re

    omega_mat = _zeros_2d(m, m, interface)
    for i in range(m):
        for j in range(i + 1, m):
            w_ij = omega.eval_re(beta_vec[i], beta_vec[j])
            if interface == "jax":
                omega_mat = omega_mat.at[i, j].set(w_ij)
                omega_mat = omega_mat.at[j, i].set(w_ij)
            else:
                omega_mat[i, j] = w_ij
                omega_mat[j, i] = w_ij

    return tau_re, beta_re, zeta_vec, delta_re_vec, omega_mat


def _combine_local_loss(entry_idx: int, params: Dict[str, Any], *, interface="jax"):
    alphas = params["alpha"]
    beta_vec = params["beta_vec"]
    lam_vec = params["lam_vec"]
    sigma = params["sigma"]
    b_norm = params["b_norm"]

    if interface == "jax":
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        b_norm = jnp.asarray(b_norm, dtype=jnp.float64)
        lam_vec = jnp.asarray(lam_vec, dtype=jnp.float64)
    else:
        sigma = pnp.asarray(sigma)
        b_norm = pnp.asarray(b_norm)

    tau_re, beta_re, zeta_vec, delta_re_vec, omega_mat = _eval_terms(
        entry_idx, alphas, beta_vec, interface=interface
    )

    c = _ENTRY_CVEC[entry_idx]
    t = c * lam_vec

    # Unified expression for both isolated and connected nodes.
    s_norm_sq = (sigma**2) * beta_re
    s_norm_sq = s_norm_sq + (jnp.sum(t**2) if interface == "jax" else pnp.sum(t**2))
    s_norm_sq = s_norm_sq + 2.0 * sigma * (
        jnp.sum(t * zeta_vec) if interface == "jax" else pnp.sum(t * zeta_vec)
    )

    cross = (t[:, None] * t[None, :]) * omega_mat
    s_norm_sq = s_norm_sq + (jnp.sum(cross) if interface == "jax" else pnp.sum(cross))

    overlap_s_b = sigma * tau_re
    overlap_s_b = overlap_s_b + (
        jnp.sum(t * delta_re_vec) if interface == "jax" else pnp.sum(t * delta_re_vec)
    )

    return s_norm_sq + b_norm**2 - 2.0 * overlap_s_b * b_norm


def prebuild_local_evals(
    SYSTEM,
    ROW_TOPOLOGY,
    n_input_qubit: int,
    diff_method="adjoint",
    interface="jax",
):
    """
    Build flat, index-driven tables for all local losses.

    Key differences vs builder_cat.py:
    - No `LOCAL_SPECS` object list.
    - No `A_op(l)` wrapper dispatch: term builders receive explicit `A_gates` directly.
    - Local isolated/connected branch is merged into one algebraic expression.
    """
    _ENTRY_SYS.clear()
    _ENTRY_AGENT.clear()
    _ENTRY_NEIGHBORS.clear()
    _ENTRY_DEGREE.clear()
    _ENTRY_COEFF.clear()
    _ENTRY_CVEC.clear()
    _ENTRY_TERMS.clear()
    _ENTRY_L.clear()

    dev = dev_cpu(int(n_input_qubit) + 1, device_name="lightning.qubit")

    N = int(SYSTEM.n)
    for sys_id in range(N):
        for agent_id in range(N):
            neighbor_ids = tuple(int(x) for x in ROW_TOPOLOGY[agent_id])
            degree = int(len(neighbor_ids))

            U = SYSTEM.b_gates[sys_id][agent_id]
            A_gates = tuple(SYSTEM.gates_grid[sys_id][agent_id])
            coeff = _as_float_array(SYSTEM.coeffs[sys_id][agent_id], interface)

            terms = make_term_bundle(
                n_input_qubit=n_input_qubit,
                U_op=U,
                A_gates=A_gates,
                dev=dev,
                interface=interface,
                diff_method=diff_method,
            )

            c_list = [-1.0 * float(degree)] + [1.0] * degree
            c_vec = _as_float_array(c_list, interface)

            _ENTRY_SYS.append(sys_id)
            _ENTRY_AGENT.append(agent_id)
            _ENTRY_NEIGHBORS.append(neighbor_ids)
            _ENTRY_DEGREE.append(degree)
            _ENTRY_COEFF.append(coeff)
            _ENTRY_CVEC.append(c_vec)
            _ENTRY_TERMS.append(terms)
            _ENTRY_L.append(int(len(A_gates)))


@qml.qjit
def eval_total_loss(current_params, *, SYSTEM=None, interface="jax"):
    """Global loss reducer over the flat prebuilt tables."""
    del SYSTEM

    total = 0.0
    n_entries = len(_ENTRY_SYS)
    for entry_idx in range(n_entries):
        agent_params = _build_agent_params(entry_idx, current_params, interface=interface)
        total = total + _combine_local_loss(entry_idx, agent_params, interface=interface)
    return total
