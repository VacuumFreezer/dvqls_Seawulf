# incomplete_builder.py
from __future__ import annotations

from typing import List, Tuple, Dict, Any, Callable, Optional
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
import jax.numpy as jnp


from .circuits_cat import make_term_bundle


# ==========================================================
# Device cache
# ==========================================================
_DEV: Dict[Tuple[str, int], qml.Device] = {}


def dev_cpu(nwires: int, device_name: str = "lightning.qubit"):
    """Reuse a single device per (name, wire-count)."""
    key = (str(device_name), int(nwires))
    if key not in _DEV:
        _DEV[key] = qml.device(device_name, wires=nwires)
    return _DEV[key]


# ==========================================================
# Optional profiler (helpful when you later want to time terms)
# ==========================================================
class TermProfiler:
    def __init__(self):
        self.t_sum = defaultdict(float)
        self.n_call = defaultdict(int)
        self._stack = []

    @contextmanager
    def section(self, name: str):
        t0 = time.perf_counter()
        self._stack.append(name)
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self._stack.pop()
            self.t_sum[name] += dt
            self.n_call[name] += 1

    def reset(self):
        self.t_sum.clear()
        self.n_call.clear()

    def report(self, top=30, sort_by="total"):
        rows = []
        for k in self.t_sum:
            rows.append((k, self.t_sum[k], self.n_call[k], self.t_sum[k] / max(1, self.n_call[k])))
        if sort_by == "avg":
            rows.sort(key=lambda x: x[3], reverse=True)
        else:
            rows.sort(key=lambda x: x[1], reverse=True)

        print(f"{'name':60s} {'total(s)':>10s} {'calls':>10s} {'avg(ms)':>10s}")
        print("-" * 95)
        for name, ttot, n, avg in rows[:top]:
            print(f"{name:60s} {ttot:10.4f} {n:10d} {avg*1e3:10.3f}")


# ==========================================================
# Local cost builder (Scheme-1 ready)
# ==========================================================
class LocalCostBuilder:
    """
    重要：本类 **不** 使用动态 dict key（"W{k}" / "L{k}"），而是使用固定顺序向量：
      involved_indices = (agent_id,) + tuple(neighbor_ids)
      beta_vec[i] 对应 involved_indices[i] 的 beta
      lam_vec[i]  对应 involved_indices[i] 的 lambda
    """

    def __init__(
        self,
        agent_id: int,
        neighbor_ids: List[int],
        term_bundle: Dict[str, Any],
        n_input_qubit: int,
        *,
        dev=None,
        interface="jax",
        diff_method="adjoint",
        show_expression=False,
    ):
        self.agent_id = int(agent_id)
        self.neighbor_ids = tuple(int(x) for x in neighbor_ids)

        # 顺序固定：self 在第 0 位，后面依次是邻居
        self.involved_indices: Tuple[int, ...] = (self.agent_id,) + self.neighbor_ids
        self.m = len(self.involved_indices)

        self.degree = len(self.neighbor_ids)
        self.is_isolated = (self.degree == 0)

        self.term_bundle = term_bundle
        self.interface = interface
        self.diff_method = diff_method

        # IMPORTANT: 你目前用 lightning.qubit + adjoint
        self.dev = dev if dev is not None else dev_cpu(n_input_qubit + 1, device_name="lightning.qubit")

        # coeff 现在固定在 builder 上（prebuild 时设置）
        self.coeff = None  # shape (L,)
        self.L = None

        # 预计算 c_vec：c_k = -deg (self) else 1
        # 顺序与 involved_indices 一致
        c0 = -1.0 * float(self.degree)
        c_rest = [1.0] * self.degree
        c_list = [c0] + c_rest
        if self.interface == "jax":
            self.c_vec = jnp.asarray(c_list, dtype=jnp.float64)  # (m,)
        else:
            self.c_vec = pnp.asarray(c_list, dtype=float)

        self.show_expression = show_expression
        if show_expression:
            self.print_expression()

    def set_coeff(self, coeff):
        """Bind coeff once; coeff is treated as constant for this builder."""
        if self.interface == "jax":
            self.coeff = jnp.asarray(coeff, dtype=jnp.float64)
        else:
            self.coeff = pnp.asarray(coeff, dtype=float)
        # L is static w.r.t. optimization
        self.L = int(self.coeff.shape[0]) if hasattr(self.coeff, "shape") else int(len(self.coeff))
        return self

    def _zero(self):
        return jnp.array(0.0, dtype=jnp.float64) if self.interface == "jax" else pnp.array(0.0)

    # ---------------------------
    # term evaluation (no dict output)
    # ---------------------------
    def _eval_terms(self, alphas, beta_vec):
        """
        Returns:
          tau_re, tau_im,
          beta_re, beta_im,
          zeta_vec (m,),
          delta_re_vec (m,),
          delta_im_vec (m,),
          omega_mat (m,m) with zeros diagonal
        """
        if self.coeff is None:
            raise RuntimeError("builder.coeff is not set. Call builder.set_coeff(C) in prebuild.")

        coeff = self.coeff
        L = self.L
        omega = self.term_bundle["OMEGA"]
        delta = self.term_bundle["DELTA"]
        zeta  = self.term_bundle["ZETA"]
        tau   = self.term_bundle["TAU"]
        beta  = self.term_bundle["BETA"]

        # ---------- tau_sum ----------
        tau_re = self._zero()
        tau_im = self._zero()
        for l in range(L):
            w = coeff[l]
            tau_re = tau_re + w * tau.eval_re(alphas, l)
            tau_im = tau_im + w * tau.eval_im(alphas, l)

        # ---------- beta_sum ----------
        beta_re = self._zero()
        beta_im = self._zero()
        for li in range(L):
            wi = coeff[li]
            for lj in range(L):
                w = wi * coeff[lj]
                beta_re = beta_re + w * beta.eval_re(alphas, li, lj)
                # print(beta.eval_re(alphas, li, lj))
                beta_im = beta_im + w * beta.eval_im(alphas, li, lj)

        # ---------- zeta/delta for involved nodes ----------
        if self.interface == "jax":
            zeta_vec = jnp.zeros((self.m,), dtype=jnp.float64)
            delta_re_vec = jnp.zeros((self.m,), dtype=jnp.float64)
            delta_im_vec = jnp.zeros((self.m,), dtype=jnp.float64)
        else:
            zeta_vec = pnp.zeros((self.m,), dtype=float)
            delta_re_vec = pnp.zeros((self.m,), dtype=float)
            delta_im_vec = pnp.zeros((self.m,), dtype=float)

        for idx in range(self.m):
            bk = beta_vec[idx]

            # zeta sum over l
            z_sum = self._zero()
            for l in range(L):
                z_sum = z_sum + coeff[l] * zeta.eval_re(alphas, bk, l)
            zeta_vec = zeta_vec.at[idx].set(z_sum) if self.interface == "jax" else zeta_vec.__setitem__(idx, z_sum)
            # print(f"zeta_vec[{idx}]={z_sum}")
            # delta
            d_re = delta.eval_re(bk)
            # print(f"d_re={d_re}")
            d_im = delta.eval_im(bk)
            if self.interface == "jax":
                delta_re_vec = delta_re_vec.at[idx].set(d_re)
                delta_im_vec = delta_im_vec.at[idx].set(d_im)
            else:
                delta_re_vec[idx] = d_re
                delta_im_vec[idx] = d_im

        # ---------- omega matrix ----------
        if self.interface == "jax":
            omega_mat = jnp.zeros((self.m, self.m), dtype=jnp.float64)
        else:
            omega_mat = pnp.zeros((self.m, self.m), dtype=float)

        for i in range(self.m):
            for j in range(i + 1, self.m):
                w_ij = omega.eval_re(beta_vec[i], beta_vec[j])
                if self.interface == "jax":
                    omega_mat = omega_mat.at[i, j].set(w_ij)
                    omega_mat = omega_mat.at[j, i].set(w_ij)
                else:
                    omega_mat[i, j] = w_ij
                    omega_mat[j, i] = w_ij
        # print(beta_re)
        # print(tau_re)
        # print(zeta_vec)
        # print(omega_mat)
        return tau_re, tau_im, beta_re, beta_im, zeta_vec, delta_re_vec, delta_im_vec, omega_mat

    # ---------------------------
    # loss assembly (vector form)
    # ---------------------------
    def combine_to_loss(self, params: Dict[str, Any]):
        """
        params must contain:
          alpha: (layers, nq)
          beta_vec: (m, layers, nq)
          lam_vec: (m,)  (scalars per involved node)
          sigma: scalar
          b_norm: scalar
        """
        alphas = params["alpha"]
        beta_vec = params["beta_vec"]
        lam_vec = params["lam_vec"]
        sigma = params["sigma"]
        b_norm = params["b_norm"]

        if self.interface == "jax":
            sigma = jnp.asarray(sigma, dtype=jnp.float64)
            b_norm = jnp.asarray(b_norm, dtype=jnp.float64)
            lam_vec = jnp.asarray(lam_vec, dtype=jnp.float64)
        else:
            sigma = pnp.asarray(sigma)
            b_norm = pnp.asarray(b_norm)

        tau_re, tau_im, beta_re, beta_im, zeta_vec, delta_re_vec, delta_im_vec, omega_mat = \
            self._eval_terms(alphas, beta_vec)

        # ---------- ISOLATED ----------
        if self.is_isolated:
            # L = ||σAx - b||^2 = σ^2 * Beta + b^2 - 2σ Re(Tau)*b
            s_norm_sq = (sigma ** 2) * beta_re
            overlap_s_b = sigma * tau_re
            loss = s_norm_sq + b_norm ** 2 - 2.0 * overlap_s_b * b_norm
            return loss

        # ---------- CONNECTED ----------
        # precomputed c_vec
        c = self.c_vec  # (m,)
        # t_i = c_i * lam_i
        t = c * lam_vec  # (m,)

        # 1) ||s||^2
        s_norm_sq = (sigma ** 2) * beta_re
        # sum_i t_i^2
        s_norm_sq = s_norm_sq + jnp.sum(t ** 2) if self.interface == "jax" else s_norm_sq + pnp.sum(t ** 2)
        # 2 sigma sum_i t_i zeta_i
        s_norm_sq = s_norm_sq + 2.0 * sigma * (jnp.sum(t * zeta_vec) if self.interface == "jax" else pnp.sum(t * zeta_vec))
        # 2 sum_{i<j} t_i t_j omega_ij  ==  sum_{i!=j} t_i t_j omega_ij
        # since omega diag=0 and symmetric, we can do:
        cross = (t[:, None] * t[None, :]) * omega_mat
        if self.interface == "jax":
            s_norm_sq = s_norm_sq + jnp.sum(cross)
        else:
            s_norm_sq = s_norm_sq + pnp.sum(cross)

        # 2) <s|b>
        overlap_s_b = sigma * tau_re
        overlap_s_b = overlap_s_b + (jnp.sum(t * delta_re_vec) if self.interface == "jax" else pnp.sum(t * delta_re_vec))
        # print(delta_re_vec)
        # 3) loss
        # print(f"overlap_s_b={overlap_s_b}")
        loss = s_norm_sq + b_norm ** 2 - 2.0 * overlap_s_b * b_norm
        return loss

    # optional (not used now)
    def print_expression(self):
        print(f"[LocalCostBuilder] agent={self.agent_id}, neigh={self.neighbor_ids}, m={self.m}, deg={self.degree}")


def make_local_costbuilder(agent_id, neighbor_ids, term_bundle, n_input_qubit, *,
                           dev=None, diff_method="adjoint", show=False, interface="jax"):
    return LocalCostBuilder(
        agent_id=agent_id,
        neighbor_ids=neighbor_ids,
        term_bundle=term_bundle,
        n_input_qubit=n_input_qubit,
        dev=dev,
        interface=interface,
        diff_method=diff_method,
        show_expression=show,
    )


# ==========================================================
# Agent param extractor
#   旧 global_params(dict of list-of-lists) 也能用
#   但返回的是 array-friendly 的 beta_vec/lam_vec（固定顺序）
# ==========================================================
def build_params_agent(
    sys_id: int,
    agent_id: int,
    row_neighbor_ids: List[int],
    global_params,
    *,
    interface="jax",
) -> Dict[str, Any]:
    """
    Returns:
      {
        "alpha": alpha_self,
        "beta_vec": stack([beta_self] + [beta_nei ...]),
        "lam_vec":  stack([lam_self]  + [lam_nei  ...]),
        "sigma": sigma_self,
        "b_norm": b_norm_self
      }
    Order is fixed: self first, then neighbors in row_neighbor_ids order.
    """
    sys_id = int(sys_id)
    agent_id = int(agent_id)
    neigh = [int(x) for x in row_neighbor_ids]

    alpha_self = global_params["alpha"][sys_id][agent_id]
    beta_self  = global_params["beta"][sys_id][agent_id]
    sigma_self = global_params["sigma"][sys_id][agent_id]
    lam_self   = global_params["lambda"][sys_id][agent_id]
    b_norm     = global_params["b_norm"][sys_id][agent_id]

    betas = [beta_self]
    lams = [lam_self]
    for n_col in neigh:
        betas.append(global_params["beta"][sys_id][n_col])
        lams.append(global_params["lambda"][sys_id][n_col])

    if interface == "jax":
        beta_vec = jnp.stack([jnp.asarray(x) for x in betas], axis=0)  # (m, layers, nq)
        lam_vec  = jnp.stack([jnp.asarray(x) for x in lams], axis=0)   # (m,)
    else:
        beta_vec = pnp.stack(betas, axis=0)
        lam_vec  = pnp.stack(lams, axis=0)

    return {
        "alpha": alpha_self,
        "beta_vec": beta_vec,
        "lam_vec": lam_vec,
        "sigma": sigma_self,
        "b_norm": b_norm,
    }


# ==========================================================
# Prebuild: build all builders + store specs
#   注意：这里不加 @qml.qjit
# ==========================================================
LOCAL_SPECS: List[Tuple[int, int, Tuple[int, ...], LocalCostBuilder]] = []


def prebuild_local_evals(
    SYSTEM,
    ROW_TOPOLOGY,
    n_input_qubit: int,
    diff_method="adjoint",
    interface="jax",
):
    """
    只调用一次：
      - 为每个 (sys_id, agent_id) 创建 term_bundle + builder
      - builder 内绑定 coeff（常量）
      - 缓存到 LOCAL_SPECS，后续 eval_total_loss 遍历它计算总 loss

    重要：这里不返回 closure 列表，避免后续 qjit/autograph 进来搞 closure mismatch。
    """
    LOCAL_SPECS.clear()

    dev = dev_cpu(int(n_input_qubit) + 1, device_name="lightning.qubit")

    N = int(SYSTEM.n)
    for sys_id in range(N):
        for agent_id in range(N):
            neighbor_ids = tuple(int(x) for x in ROW_TOPOLOGY[agent_id])

            U = SYSTEM.b_gates[sys_id][agent_id]
            A = SYSTEM.ops[sys_id][agent_id]
            C = SYSTEM.coeffs[sys_id][agent_id]

            term_bundle = make_term_bundle(
                n_input_qubit=n_input_qubit,
                U_op=U,
                A_op=A,
                dev=dev,
                interface=interface,
                diff_method=diff_method,
            )

            builder = make_local_costbuilder(
                agent_id=agent_id,
                neighbor_ids=list(neighbor_ids),
                term_bundle=term_bundle,
                n_input_qubit=n_input_qubit,
                dev=dev,
                diff_method=diff_method,
                show=False,
                interface=interface,
            ).set_coeff(C)

            LOCAL_SPECS.append((sys_id, agent_id, neighbor_ids, builder))

    b0 = LOCAL_SPECS[0][3]   # 例如 sys0 agent0
    b1 = LOCAL_SPECS[1][3]   # 例如 sys0 agent1

    d0 = b0.term_bundle["DELTA"]
    d1 = b1.term_bundle["DELTA"]

@qml.qjit
def eval_total_loss(current_params, *, SYSTEM=None, interface="jax"):
    """
    总 loss（仍是 Python 循环版本，给你现在先跑通/对比用）。
    Scheme-1 真正提速会在下一步：把 update_step/optimize 用 qjit + for_loop 编译。
    """
    total = 0.0
    for sys_id, agent_id, neighbor_ids, builder in LOCAL_SPECS:
        agent_params = build_params_agent(
            sys_id=sys_id,
            agent_id=agent_id,
            row_neighbor_ids=list(neighbor_ids),
            global_params=current_params,
            interface=interface,
        )
        total = total + builder.combine_to_loss(agent_params)
    return total


# ==========================================================
# 关于 @qml.qjit：本文件里不加（很重要）
# ==========================================================
"""
为什么这里不加 @qml.qjit

1) qml.qjit 不能安全地装饰类方法（会破坏 self 绑定，导致你之前的 missing params 错误）。
2) 目前 total loss 仍有 Python 侧的循环和对象列表（LOCAL_SPECS），它应该由“外层 cost/update_step”
   在下一步用 qml.for_loop + @qml.qjit 来编译，而不是在这里对这些方法打补丁式 qjit。
"""
