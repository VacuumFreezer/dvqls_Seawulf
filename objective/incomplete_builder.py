from typing import List, Tuple, Dict, Any
import pennylane.numpy as pnp

# Import the Global Circuit Objects
from .circuits_sys import OMEGA, DELTA, ZETA, BETA, TAU
# ==========================================================
# Term interface and concrete terms and term sums
# These accept python callables that return numpy scalars
# ==========================================================

class Term:
    """Interface for a cost term (numpy)."""
    def __init__(self, name: str):
        self.name = name
    def eval(self, alphas, betas, coeff):
        raise NotImplementedError

class OmegaWrapper(Term):
    def __init__(self, name, k1, k2):
        super().__init__(name)
        self.k1, self.k2 = k1, k2
    def eval(self, alphas, betas, coeff):
        return OMEGA.compute(betas, self.k1, self.k2)

class DeltaWrapper(Term):
    def __init__(self, name, k):
        super().__init__(name)
        self.k = k
    def eval(self, alphas, betas, coeff):
        return DELTA.compute(betas, self.k)

class ZetaSumWrapper(Term):
    def __init__(self, name, k):
        super().__init__(name)
        self.k = k
    def eval(self, alphas, betas, coeff):
        s = 0.0
        for l in range(len(coeff)):
            s += coeff[l] * ZETA.compute(alphas, betas, l, self.k)
        return s

class TauSumWrapper(Term):
    def __init__(self, name):
        super().__init__(name)
    def eval(self, alphas, betas, coeff):
        s = 0.0
        for l in range(len(coeff)):
            s += coeff[l] * TAU.compute(alphas, l)
        return s

class BetaDoubleSumWrapper(Term):
    def __init__(self, name):
        super().__init__(name)
    def eval(self, alphas, betas, coeff):
        s = 0.0
        L = len(coeff)
        for li in range(L):
            for lj in range(L):
                s += coeff[li] * coeff[lj] * BETA.compute(alphas, li, lj)
        return s

class LocalCostBuilder:
    def __init__(
        self, 
        terms: List[Term], 
        agent_id: int, 
        neighbor_ids: List[int],
        show_expression: bool = False
    ):
        self.terms = terms
        self.agent_id = agent_id
        self.neighbor_ids = neighbor_ids
        self.show_expression = show_expression

        # Calculate degree for the coefficient rule
        self.degree = len(neighbor_ids)
        self.is_isolated = (self.degree == 0)
        
        # Combined indices (Self + Neighbors)
        self.involved_indices = sorted([agent_id] + neighbor_ids)

        if self.show_expression:
            # Print the mathematical expression
            self.print_expression()

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

    def _eval_terms(self, params: Dict[str, Any]) -> Dict[str, pnp.tensor]:
        alphas = params["alpha"]
        betas = params["beta"] 
        coeff = params["coeff"]
        return {t.name: t.eval(alphas, betas, coeff) for t in self.terms}

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
        s_norm_sq = (sigma**2) * vals["beta_sum"]
        
        # Single loop terms (Lambda^2 and Zeta)
        for k in self.involved_indices:
            lam_k = lam_dict[f"L{k}"]
            c_k = self._get_coeff(k) # Gets -degree or +1
            
            # Term: (c_k * λ_k)^2
            s_norm_sq = s_norm_sq + (c_k * lam_k)**2
            
            # Term: 2 * σ * (c_k * λ_k) * ζ_k
            zeta_val = vals[f"zeta{k}"]
            s_norm_sq = s_norm_sq + 2 * sigma * (c_k * lam_k) * zeta_val

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
                
                omega_val = vals[f"omega_{k1}_{k2}"]
                
                s_norm_sq = s_norm_sq + 2 * (c_1 * lam_1) * (c_2 * lam_2) * omega_val

        # 2. Calculate Overlap <s|b>
        # Term: σ * τ
        overlap_s_b = sigma * vals["tau_sum"]
        
        # Term: (c_k * λ_k) * δ_k
        for k in self.involved_indices:
            lam_k = lam_dict[f"L{k}"]
            c_k = self._get_coeff(k)
            delta_val = vals[f"delta{k}"]
            
            overlap_s_b = overlap_s_b + (c_k * lam_k) * delta_val

        # 3. Final Loss
        if pnp.imag(s_norm_sq) >= 1e-7 or pnp.imag(overlap_s_b) >= 1e-7:
            raise ValueError("Imaginary part of complex term sum is non-zero.")

        loss = pnp.real(s_norm_sq) + b_norm**2 - 2 * pnp.real(overlap_s_b) * b_norm
        return loss

def make_local_costbuilder(agent_id: int, neighbor_ids: List[int], show: bool = False) -> LocalCostBuilder:
    """
    Dynamically creates a CostBuilder specific to one agent's topology.
    """
    terms: List[Term] = []

    # 1. Always include Local System Terms
    terms.append(BetaDoubleSumWrapper("beta_sum"))
    terms.append(TauSumWrapper("tau_sum"))
    if len(neighbor_ids) > 0:
        # Define the full set of indices involved in this local loss
        involved_indices = sorted([agent_id] + neighbor_ids)

        # 2. Per-Agent terms (Zeta, Delta) for EVERY involved index
        for k in involved_indices:
            terms.append(ZetaSumWrapper(f"zeta{k}", k=k))
            terms.append(DeltaWrapper(f"delta{k}", k=k))
            
        # 3. Cross-Agent terms (Omega) for every pair in involved indices
        # We loop carefully to avoid duplicates and ensure k1 < k2 convention for names
        for i in range(len(involved_indices)):
            for j in range(i + 1, len(involved_indices)):
                k1 = involved_indices[i]
                k2 = involved_indices[j]

                terms.append(OmegaWrapper(f"omega_{k1}_{k2}", k1=k1, k2=k2))

    return LocalCostBuilder(terms, agent_id, neighbor_ids, show_expression=show)

    
# ==========================================================
SYSTEM = None  # will be injected

def bind_static_ops(static_ops_module):
    """
      import importlib
      ops = importlib.import_module("problems.static_ops_equation1_kappa196")
      ib.bind_static_ops(ops)
    """
    global SYSTEM
    SYSTEM = static_ops_module.SYSTEM

def _require_system():
    if SYSTEM is None:
        raise RuntimeError(
            "SYSTEM is not bound. Call objective.incomplete_builder.bind_static_ops(static_ops_module) first."
        )

def static_builder(params: Dict[str, Any], builder: LocalCostBuilder, sys_id: int, agent_id: int):
    """
    Injects the static operators (U, A) and coefficients (C) 
    using the 2D system structure.
    """
    _require_system()
    # 1. Retrieve Operators and Coefficients
    # --------------------------------------
    # U: Single unitary function for this grid point (no 'l' index)
    U = SYSTEM.b_gates[sys_id][agent_id]
    
    # A: Wrapper function for this grid point (requires 'l' index)
    A = SYSTEM.ops[sys_id][agent_id]
    
    # C: List of coefficients for A
    C = SYSTEM.coeffs[sys_id][agent_id]

    # 2. Inject into Static Term Definitions
    # --------------------------------------
    # The internal Term classes will handle U as a direct call U() 
    # and A as an indexed call A(l).
    for T in (OMEGA, DELTA, ZETA, BETA, TAU):
        T.set_static_ops(U_op=U, A_op=A)

    # 3. Prepare Parameters and Compute Loss
    # --------------------------------------
    p = dict(params)
    p["coeff"] = C  # Pass the coeff vector specific to A
    
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