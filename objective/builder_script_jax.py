from typing import List, Tuple, Dict, Any
import pennylane as qml
import pennylane.numpy as pnp
import jax.numpy as jnp

# Import the Global Circuit Objects
from .circuit_script import make_term_bundle
# ==========================================================
# Term interface and concrete terms and term sums
# These accept python callables that return numpy scalars
# ==========================================================
import time
from collections import defaultdict
from contextlib import contextmanager

_DEV = {}
def dev_cpu(nwires: int):
    """Reuse a single default.qubit device per wire-count."""
    if nwires not in _DEV:
        _DEV[nwires] = qml.device("default.qubit", wires=nwires)
    return _DEV[nwires]

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
        print("-"*95)
        for name, ttot, n, avg in rows[:top]:
            print(f"{name:60s} {ttot:10.4f} {n:10d} {avg*1e3:10.3f}")



class LocalCostBuilder:
    def __init__(self, agent_id, neighbor_ids, term_bundle, n_input_qubit, 
                 interface="jax", diff_method="best", show_expression=False):
        self.agent_id = agent_id
        self.neighbor_ids = neighbor_ids
        self.involved_indices = sorted([agent_id] + neighbor_ids)
        self.degree = len(neighbor_ids)
        self.is_isolated = (self.degree == 0)

        self.term_bundle = term_bundle
        self.interface = interface
        self.diff_method = diff_method
        self.dev = dev_cpu(n_input_qubit + 1)

        self.show_expression = show_expression
        if show_expression:
            self.print_expression()

    def _zero(self):
        if self.interface == "jax":
            return jnp.array(0.0)
        return pnp.array(0.0)
    
    def _get_coeff(self, k: int) -> float:
        """
        Returns the coefficient for lambda_k.
        Rule: 
          - If k is self: coeff = -1 * degree
          - If k is neighbor: coeff = +1
        """
        if k == self.agent_id:
            return -1.0 * self.degree
        return 1.0


    def print_expression(self):
        print(f"\n=== Local Cost Function for Agent {self.agent_id} ===")
        
        if self.is_isolated:
            print("Topology: ISOLATED (No Neighbors)")
            print(f"State: |s_{self.agent_id}> = σ_{self.agent_id} A_{self.agent_id} |x_{self.agent_id}>")
            print("L = ||σ A x - b||^2")
            print("  = σ^2 <ψ|β|ψ> + <b|b> - 2σ Re(<ψ|τ|b>)")
        else:
            print(f"Topology: Connected to {self.neighbor_ids} (Degree {self.degree})")
            
            # 1. State Definition
            s_str = f"|s_{self.agent_id}> = σ_{self.agent_id} A_{self.agent_id} |x_{self.agent_id}>"
            
            # Self term (with degree coeff)
            c_self = self._get_coeff(self.agent_id)
            s_str += f" {c_self:.0f} λ_{self.agent_id} |z_{self.agent_id}>"
            
            # Neighbor terms
            for n_id in self.neighbor_ids:
                s_str += f" + λ_{n_id} |z_{n_id}>"
            
            print(f"State Definition:\n  {s_str}\n")
            
            # 2. Norm Squared
            norm_str = f"||s_{self.agent_id}||^2 = \n"
            norm_str += f"    σ^2 * Beta_Sum \n"
            
            # Loop for Lambda^2 and Zeta
            for k in self.involved_indices:
                c = self._get_coeff(k)
                # Squared term is always positive coefficient (c^2)
                norm_str += f"  + {c**2:.0f} λ_{k}^2\n"
                # Zeta term: 2 * sigma * coeff * zeta
                norm_str += f"  + 2σ ({c:.0f} λ_{k}) Re(ζ_{k})\n"

            # Loop for Omega
            for i in range(len(self.involved_indices)):
                for j in range(i + 1, len(self.involved_indices)):
                    k1, k2 = self.involved_indices[i], self.involved_indices[j]
                    c1 = self._get_coeff(k1)
                    c2 = self._get_coeff(k2)
                    norm_str += f"  + 2 ({c1:.0f} λ_{k1}) ({c2:.0f} λ_{k2}) Re(ω_{k1},{k2})\n"
            
            print(norm_str)
            
            # 3. Overlap
            over_str = f"<s_{self.agent_id}|b> = \n"
            over_str += "    σ * τ_sum \n"
            for k in self.involved_indices:
                c = self._get_coeff(k)
                over_str += f"  + ({c:.0f} λ_{k}) δ_{k}\n"
            
            print(over_str)
            
        print("==========================================\n")


    def _eval_terms(self, params):
        alphas = params["alpha"]
        betas  = params["beta"]
        coeff  = params["coeff"]
        if self.interface == "jax":
            
            coeff = jnp.asarray(coeff)

        L = len(coeff)

        omega = self.term_bundle["OMEGA"]
        delta = self.term_bundle["DELTA"]
        zeta  = self.term_bundle["ZETA"]
        tau   = self.term_bundle["TAU"]
        beta  = self.term_bundle["BETA"]

        tapes = []
        sl = {}  # name -> slice

        # ---------- tau_sum: 2*L tapes ----------
        t0 = len(tapes)
        for l in range(L):
            tapes.extend(tau.make_tapes_for_l(alphas, l))  # [re, im]
        sl["tau"] = slice(t0, len(tapes))

        # ---------- beta_sum: 2*L*L tapes ----------
        t0 = len(tapes)
        for li in range(L):
            for lj in range(L):
                tapes.extend(beta.make_tapes_for_pair(alphas, li, lj))  # [re, im]
        sl["beta"] = slice(t0, len(tapes))

        # ---------- zeta{k}: L tapes each ----------
        for k in self.involved_indices:
            t0 = len(tapes)
            bk = betas[f"W{k}"]
            for l in range(L):
                tapes.append(zeta.make_tape(alphas, bk, l))  # real
            sl[f"zeta{k}"] = slice(t0, len(tapes))

        # ---------- delta{k}: 2 tapes each ----------
        for k in self.involved_indices:
            t0 = len(tapes)
            bk = betas[f"W{k}"]
            tapes.extend(delta.make_tapes(bk))  # [re, im]
            sl[f"delta{k}"] = slice(t0, len(tapes))

        # ---------- omega_{k1,k2}: 1 tape each ----------
        for i in range(len(self.involved_indices)):
            for j in range(i + 1, len(self.involved_indices)):
                k1 = self.involved_indices[i]
                k2 = self.involved_indices[j]
                t0 = len(tapes)
                tapes.append(omega.make_tape(betas[f"W{k1}"], betas[f"W{k2}"]))  # real
                sl[f"omega_{k1}_{k2}"] = slice(t0, len(tapes))

        # ---------- one-shot execute ----------
        # 你可以二选一：
        # 1) gradient_fn=None: 先只测 forward 是否变快
        # 2) gradient_fn=... : 让它支持 jax 求导（版本相关）
        # exec_kwargs = dict(interface=self.interface)
        # if self.gradient_fn is not None:
        #     exec_kwargs["gradient_fn"] = self.gradient_fn

        out = qml.execute(tapes, self.dev, diff_method=self.diff_method,  
                          interface=self.interface)

        # ---------- unpack & postprocess ----------
        vals = {}

        # tau_sum
        tau_re = self._zero()
        tau_im = self._zero()

        tau_out = out[sl["tau"]]  # [re0, im0, re1, im1, ...]
        for l in range(L):
            tau_re = tau_re + coeff[l] * tau_out[2*l]
            tau_im = tau_im + coeff[l] * tau_out[2*l+1]

        vals["tau_sum_re"] = tau_re
        vals["tau_sum_im"] = tau_im


        # beta_sum
        beta_re = self._zero()
        beta_im = self._zero()

        beta_out = out[sl["beta"]]  # length 2*L*L
        idx = 0
        for li in range(L):
            for lj in range(L):
                w = coeff[li]*coeff[lj]
                beta_re = beta_re + w * beta_out[idx]
                beta_im = beta_im + w * beta_out[idx+1]
                idx += 2

        vals["beta_sum_re"] = beta_re
        vals["beta_sum_im"] = beta_im


        # zeta{k}
        for k in self.involved_indices:
            z_out = out[sl[f"zeta{k}"]]  # length L, real
            z_sum = self._zero()
            for l in range(L):
                z_sum = z_sum + coeff[l] * z_out[l]
            vals[f"zeta{k}"] = z_sum

        # delta{k}
        for k in self.involved_indices:
            d_out = out[sl[f"delta{k}"]]  # [re, im]
            vals[f"delta{k}_re"] = d_out[0]
            vals[f"delta{k}_im"] = d_out[1]

        # omega_{k1,k2}
        for i in range(len(self.involved_indices)):
            for j in range(i + 1, len(self.involved_indices)):
                k1 = self.involved_indices[i]
                k2 = self.involved_indices[j]
                vals[f"omega_{k1}_{k2}"] = out[sl[f"omega_{k1}_{k2}"]][0]
  
        return vals

    def combine_to_loss(self, params: Dict[str, Any]):
        vals = self._eval_terms(params)
        
        sigma = params["sigma"]
        b_norm = params["b_norm"]
        
        # --- CASE 1: ISOLATED AGENT ---
        if self.is_isolated:
            # L = ||σAx - b||^2 = σ^2 Beta + b^2 - 2σ Re(Tau)
            s_norm_sq = (sigma**2) * vals["beta_sum"]
            overlap_s_b = sigma * vals["tau_sum"]
            
            loss = pnp.real(s_norm_sq) + b_norm**2 - 2 * pnp.real(overlap_s_b) * b_norm
            return loss, pnp.real(s_norm_sq)

        # --- CASE 2: CONNECTED AGENT ---
        lam_dict = params["lambda"] 

        # 1. Calculate ||s||^2
        s_norm_sq = (sigma**2) * vals["beta_sum_re"]
        
        # Single loop terms (Lambda^2 and Zeta)
        for k in self.involved_indices:
            lam_k = lam_dict[f"L{k}"]
            c_k = self._get_coeff(k) # Gets -degree or +1
            
            # Term: (c_k * λ_k)^2
            s_norm_sq = s_norm_sq + (c_k * lam_k)**2
            
            # Term: 2 * σ * (c_k * λ_k) * ζ_k
            s_norm_sq = s_norm_sq + 2 * sigma * (c_k * lam_k) * vals[f"zeta{k}"]

        # Cross terms (Omega)
        # Term: 2 * (c_j * λ_j) * (c_k * λ_k) * ω_jk
        for i in range(len(self.involved_indices)):
            for j in range(i + 1, len(self.involved_indices)):
                k1 = self.involved_indices[i]
                k2 = self.involved_indices[j]
                
                lam_1 = lam_dict[f"L{k1}"]
                lam_2 = lam_dict[f"L{k2}"]
                c_1 = self._get_coeff(k1)
                c_2 = self._get_coeff(k2)
                
                s_norm_sq = s_norm_sq + 2 * (c_1 * lam_1) * (c_2 * lam_2) * vals[f"omega_{k1}_{k2}"]

        # 2. Calculate Overlap <s|b>
        # Term: σ * τ
        overlap_s_b = sigma * vals["tau_sum_re"]
        
        # Term: (c_k * λ_k) * δ_k
        for k in self.involved_indices:
            lam_k = lam_dict[f"L{k}"]
            c_k = self._get_coeff(k)
            
            overlap_s_b = overlap_s_b + (c_k * lam_k) * vals[f"delta{k}_re"]

        # # 3. Final Loss
        # if qml.math.toarray(vals["beta_sum_im"]) >= 1e-7 or qml.math.toarray(vals["tau_sum_im"]) >= 1e-7:
        #     raise ValueError("Imaginary part of complex term sum is non-zero.")

        loss = s_norm_sq + b_norm**2 - 2 * overlap_s_b * b_norm
        return loss

def make_local_costbuilder(agent_id, neighbor_ids, term_bundle, n_input_qubit, diff_method="best", show=False, interface="jax"):
    return LocalCostBuilder(
        agent_id=agent_id,
        neighbor_ids=neighbor_ids,
        term_bundle=term_bundle,
        n_input_qubit=n_input_qubit,
        interface=interface,
        diff_method=diff_method,
        show_expression=show,
    )


def static_builder(params, builder, coeff):
    p = dict(params)
    p["coeff"] = coeff
    return builder.combine_to_loss(p)

# ===== Agent helpers (unchanged shape/API, just add system selector) =====
"""
nei means the neighbour of this agent
"""
# GlobalParams = Dict[str, Any] 

def build_params_agent(
    sys_id: int,                 # Row index (0-based)
    agent_id: int,               # Column index (0-based)
    row_neighbor_ids: List[int], # Neighbor column indices (0-based)
    global_params,  # Global data structure
) -> Dict[str, Any]:
    """
    Constructs the parameter dictionary using 0-based indexing keys.
    Keys will be: "W0", "W1", "lambda0", "lambda1", etc.
    """
    
    # 1. Retrieve Local Parameters (Self)
    # -----------------------------------
    alpha_self = global_params["alpha"][sys_id][agent_id]
    beta_self  = global_params["beta"][sys_id][agent_id]
    sigma_self = global_params["sigma"][sys_id][agent_id]
    lam_self   = global_params["lambda"][sys_id][agent_id]
    b_norm     = global_params["b_norm"][sys_id][agent_id]

# 2. Initialize Sub-Dictionaries
    # ------------------------------
    beta_dict = {}
    lam_dict = {}
    
    # 3. Add Self Parameters
    # ----------------------
    # Using "W" for weights (beta) and "L" for lambdas
    beta_dict[f"W{agent_id}"] = beta_self
    lam_dict[f"L{agent_id}"] = lam_self
    
    # 4. Add Neighbor Parameters
    # --------------------------
    for n_col in row_neighbor_ids:
        # Retrieve from global store
        beta_nei = global_params["beta"][sys_id][n_col]
        lam_nei  = global_params["lambda"][sys_id][n_col]
        
        # Store with neighbor's ID
        beta_dict[f"W{n_col}"] = beta_nei
        lam_dict[f"L{n_col}"] = lam_nei
    
    # 5. Construct Final Dictionary
    # -----------------------------
    params = {
        "alpha": alpha_self,
        "beta": beta_dict,   # Now contains all W terms
        "lambda": lam_dict,  # Now contains all L terms
        "sigma": sigma_self,
        "b_norm": b_norm,
    }

    return params

LOCAL_EVALS = []

def prebuild_local_evals(SYSTEM, ROW_TOPOLOGY, n_input_qubit, diff_method="adjoint", interface="jax"):
    """
    只调用一次：为每个 (sys_id, agent_id) 创建 term_bundle + builder，并缓存 evaluator 闭包
    """
    LOCAL_EVALS.clear()

    for sys_id in range(SYSTEM.n):
        for agent_id in range(SYSTEM.n):
            neighbor_ids = ROW_TOPOLOGY[agent_id]

            U = SYSTEM.b_gates[sys_id][agent_id]
            A = SYSTEM.ops[sys_id][agent_id]
            C = SYSTEM.coeffs[sys_id][agent_id]

            term_bundle = make_term_bundle(
                n_input_qubit=n_input_qubit,
                U_op=U,
                A_op=A,
            )

            builder = make_local_costbuilder(
                agent_id=agent_id,
                neighbor_ids=neighbor_ids,
                term_bundle=term_bundle,
                n_input_qubit=n_input_qubit,
                diff_method=diff_method,
                show=False,
                interface=interface
            )

            def _eval_one(current_params, _sys=sys_id, _aid=agent_id, _neigh=neighbor_ids, _C=C, _b=builder):
                agent_params = build_params_agent(
                    sys_id=_sys,
                    agent_id=_aid,
                    row_neighbor_ids=_neigh,
                    global_params=current_params,
                )
                return static_builder(agent_params, _b, coeff=_C)

            LOCAL_EVALS.append(_eval_one)


def eval_total_loss(current_params):
    total = 0.0
    for f in LOCAL_EVALS:
        total = total + f(current_params)
    return total
