from pennylane import numpy as pnp
import numpy as np
from typing import Callable, Any
import pennylane as qml
from functools import lru_cache

# ======================================================================
# NOTE:
#  k is the index for agents
#  l is the index for A unitaries
# ======================================================================

# ======================================================================
# Device cache 
# ======================================================================
_DEV = {}
def dev_cpu(nwires: int):
    """Reuse a single lightning.qubit device per wire-count."""
    if nwires not in _DEV:
        _DEV[nwires] = qml.device("default.qubit", wires=nwires)
    return _DEV[nwires]

# ======================================================================
# Father class: setting up variational blocks
# ======================================================================

class HadamardTest:
    """
    Father class for Hadamard-test style terms (δ, χ, ζ, ω).

    Static gates:
      - U_op(): callable that applies U (no trainable params)
      - A_op(l): callable that applies A_{l} (no trainable params)

    Shared variational blocks (same V and W used by all terms):
      - W_var_block(weights)
      - V_var_block(alphas)

    Notes:
      All indice are representing the index of agent
    """
    def __init__(
        self,
        n_input_qubit: int,
        U_op: Callable[[], None] = None,
        A_op: Callable[[int], None] = None,
    ):
        self.n_input_qubit = n_input_qubit
        self.U_op = U_op
        self.A_op = A_op

    def set_static_ops(
        self,
        U_op: Callable[[], None],
        A_op: Callable[[int], None],
    ):
        
        self.U_op = U_op
        self.A_op = A_op
        # self.coeff = coeff
        # return self

    # Variational blocks
    def W_var_block(self, betas: pnp.tensor):
        """Trainable W(beta) block"""
        n = self.n_input_qubit
        # qml.AngleEmbedding(features=betas, wires=range(n), rotation="Y")
        qml.BasicEntanglerLayers(weights=betas, wires=range(n), rotation=qml.RY)
        # q0, q1 = 0, 1
        # # Layer 0
        # qml.RY(betas[0, 0], wires=q0)
        # qml.RY(betas[0, 1], wires=q1)
        # qml.CZ(wires=[q0, q1])

        # # Layer 1
        # qml.RY(betas[1, 0], wires=q0)
        # qml.RY(betas[1, 1], wires=q1)
        # qml.CZ(wires=[q0, q1])  # CZ is symmetric; reversing is identical for 2 qubits

        # # Layer 2
        # qml.RY(betas[2, 0], wires=q0)
        # qml.RY(betas[2, 1], wires=q1)
   

    def V_var_block(self, alphas: pnp.tensor):
        """Trainable V(alpha) block."""
        n = self.n_input_qubit 
        # qml.AngleEmbedding(features=alphas, wires=range(n), rotation="Y")
        qml.BasicEntanglerLayers(weights=alphas, wires=range(n), rotation=qml.RY)
        # q0, q1 = 0, 1
        # # Layer 0
        # qml.RY(alphas[0, 0], wires=q0)
        # qml.RY(alphas[0, 1], wires=q1)
        # qml.CZ(wires=[q0, q1])

        # # Layer 1
        # qml.RY(alphas[1, 0], wires=q0)
        # qml.RY(alphas[1, 1], wires=q1)
        # qml.CZ(wires=[q0, q1])  # CZ is symmetric; reversing is identical for 2 qubits

        # # Layer 2
        # qml.RY(alphas[2, 0], wires=q0)
        # qml.RY(alphas[2, 1], wires=q1)

# ======================================================================
# Child terms (system-aware versions)
# NOTE:
#  - These terms are bound to a `system` or a column partition problem.
# ======================================================================

class OmegaTerm(HadamardTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n = self.n_input_qubit
        self.anc = n  # ancilla wire index

    @lru_cache(None)
    def _script_maker(self):
        anc = self.anc

        def qfunc(b1, b2):
            qml.Hadamard(wires=anc)
            qml.ctrl(self.W_var_block, control=anc)(b2)
            qml.ctrl(qml.adjoint(self.W_var_block), control=anc)(b1)
            qml.Hadamard(wires=anc)
            return qml.expval(qml.PauliZ(anc))

        return qml.tape.make_qscript(qfunc)

    def make_tape(self, b1, b2):
        return self._script_maker()(b1, b2)



class DeltaTerm(HadamardTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anc = self.n_input_qubit

    @lru_cache(None)
    def _script_maker(self):
        anc = self.anc

        def U_dag():
            qml.adjoint(self.U_op)()

        def qfunc(bk, phase):
            qml.Hadamard(wires=anc)
            qml.PhaseShift(phase, wires=anc)
            qml.ctrl(self.W_var_block, control=anc)(bk)
            qml.ctrl(U_dag, control=anc)()
            qml.Hadamard(wires=anc)
            return qml.expval(qml.PauliZ(anc))

        return qml.tape.make_qscript(qfunc)

    def make_tapes(self, bk):
        mk = self._script_maker()
        return [mk(bk, 0.0), mk(bk, -np.pi / 2)]



class ZetaTerm(HadamardTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anc = self.n_input_qubit

    @lru_cache(None)
    def _script_maker_for_l(self, l: int):
        anc = self.anc

        def A_l_dag():
            qml.adjoint(self.A_op)(l)

        def qfunc(bk, a):
            qml.Hadamard(wires=anc)
            qml.ctrl(self.W_var_block, control=anc)(bk)
            qml.ctrl(A_l_dag, control=anc)()
            qml.ctrl(qml.adjoint(self.V_var_block), control=anc)(a)
            qml.Hadamard(wires=anc)
            return qml.expval(qml.PauliZ(anc))

        return qml.tape.make_qscript(qfunc)

    def make_tape(self, alphas, bk, l: int):
        return self._script_maker_for_l(l)(bk, alphas)



class TauTerm(HadamardTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anc = self.n_input_qubit

    @lru_cache(None)
    def _script_maker_for_l(self, l: int):
        anc = self.anc

        def op_chain():
            self.U_op()
            qml.adjoint(self.A_op)(l)

        def qfunc(alphas, phase):
            qml.Hadamard(wires=anc)
            qml.PhaseShift(phase, wires=anc)
            qml.ctrl(op_chain, control=anc)()
            qml.ctrl(qml.adjoint(self.V_var_block), control=anc)(alphas)
            qml.Hadamard(wires=anc)
            return qml.expval(qml.PauliZ(anc))

        return qml.tape.make_qscript(qfunc)

    def make_tapes_for_l(self, alphas, l: int):
        mk = self._script_maker_for_l(l)
        return [mk(alphas, 0.0), mk(alphas, -np.pi / 2)]



class BetaTerm(HadamardTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anc = self.n_input_qubit

    @lru_cache(None)
    def _script_maker_for_pair(self, l: int, lp: int):
        anc = self.anc

        def op_chain():
            self.A_op(l)
            qml.adjoint(self.A_op)(lp)

        def qfunc(alphas, phase):
            qml.Hadamard(wires=anc)
            qml.PhaseShift(phase, wires=anc)
            self.V_var_block(alphas)
            qml.ctrl(op_chain, control=anc)()
            qml.Hadamard(wires=anc)
            return qml.expval(qml.PauliZ(anc))

        return qml.tape.make_qscript(qfunc)

    def make_tapes_for_pair(self, alphas, l: int, lp: int):
        mk = self._script_maker_for_pair(l, lp)
        return [mk(alphas, 0.0), mk(alphas, -np.pi / 2)]



# Collect all term instances into a bundle. Set static ops here.
def make_term_bundle(
    n_input_qubit: int,
    U_op,
    A_op,
):
    omega = OmegaTerm(n_input_qubit=n_input_qubit, U_op=U_op, A_op=A_op)
    delta = DeltaTerm(n_input_qubit=n_input_qubit, U_op=U_op, A_op=A_op)
    zeta  = ZetaTerm(n_input_qubit=n_input_qubit, U_op=U_op, A_op=A_op)
    tau   = TauTerm(n_input_qubit=n_input_qubit, U_op=U_op, A_op=A_op)
    beta  = BetaTerm(n_input_qubit=n_input_qubit, U_op=U_op, A_op=A_op)

    return {"OMEGA": omega, "DELTA": delta, "ZETA": zeta, "TAU": tau, "BETA": beta}
# # ========== Term instances ==========
# OMEGA = OmegaTerm(n_input_qubit=2)
# DELTA = DeltaTerm(n_input_qubit=2)
# ZETA  = ZetaTerm(n_input_qubit=2)
# TAU   = TauTerm(n_input_qubit=2)
# BETA  = BetaTerm(n_input_qubit=2)

# # ========== Public wrappers with the SAME names/signatures as one-system ==========

# def omega_k1k2_re(betas, k1: int, k2: int) -> float:
#     return OMEGA.compute(betas, k1, k2)

# def delta_k(betas, k: int) -> complex:
#     return DELTA.compute(betas, k)

# def zeta_lk_re(alphas, betas, l: int, k: int) -> float:
#     return ZETA.compute(alphas, betas, l, k)

# def tau_l(alphas, l: int) -> complex:
#     return TAU.compute(alphas, l)

# def beta_llp(alphas, l: int, lp: int) -> complex:
#     return BETA.compute(alphas, l, lp)
