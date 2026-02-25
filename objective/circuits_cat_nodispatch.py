from pennylane import numpy as pnp
import numpy as np
from typing import Callable, Any, Sequence
import pennylane as qml
# from functools import lru_cache

import jax
import jax.numpy as jnp

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
    """Reuse a single default.qubit device per wire-count."""
    if nwires not in _DEV:
        _DEV[nwires] = qml.device("lightning.qubit", wires=nwires)
    return _DEV[nwires]

# ======================================================================
# Father class: setting up variational blocks
# ======================================================================

class HadamardTest:
    """
    Father class for Hadamard-test style terms (δ, χ, ζ, ω).

    Static gates (no trainable params):
      - U_op(): callable that applies U 
      - A_gates[l]: explicit local gate list for A_{l}

    Shared variational blocks (same V and W used by all terms):
      - W_var_block(betas)
      - V_var_block(alphas)
    """
    def __init__(
        self,
        n_input_qubit: int,
        U_op: Callable[[], None] = None,
        A_gates: Sequence[Callable[[], None]] = (),
    ):
        self.n_input_qubit = n_input_qubit
        self.U_op = U_op
        self.A_gates = tuple(A_gates)

    def bind_runtime(self, dev, interface="jax", diff_method="adjoint", use_jit=True):
        self.dev = dev
        self.interface = interface
        self.diff_method = diff_method
        self.use_jit = use_jit
        return self
    
    def _maybe_jit(self, fn):
        if self.use_jit and self.interface in ("jax", "jax-jit"):
            return jax.jit(fn)
        return fn
    
    # Variational blocks
    def W_var_block(self, betas):
        """Trainable W(beta) block"""
        n = self.n_input_qubit
        # qml.AngleEmbedding(features=betas, wires=range(n), rotation="Y")
        return qml.BasicEntanglerLayers(weights=betas, wires=range(n), rotation=qml.RY)
   

    def V_var_block(self, alphas):
        """Trainable V(alpha) block."""
        n = self.n_input_qubit 
        # qml.AngleEmbedding(features=alphas, wires=range(n), rotation="Y")
        return qml.BasicEntanglerLayers(weights=alphas, wires=range(n), rotation=qml.RY)

# ======================================================================
# Child terms (system-aware versions)
# ======================================================================

class OmegaTerm(HadamardTest):
    '''
    OMEGA_{k1,k2} = \text{Re}(\langle 0|W_{k1}^\dagger W_{k2}|0 \rangle)
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n = self.n_input_qubit
        self.anc = n  # ancilla wire index

  # ----------- qnode path (NEW) -----------
    # @lru_cache(None)
    def compute(self, b1, b2):
        if self.dev is None:
            raise RuntimeError("OmegaTerm: runtime not bound. Call bind_runtime(dev, ...).")

        anc = self.anc
        dev = self.dev
        interface = self.interface
        diff_method = self.diff_method

        @qml.qnode(dev, interface=interface, diff_method=diff_method)
        def qnode(b1, b2):
            qml.Hadamard(wires=anc)
            # controlled W2, then controlled W1^\dagger
            qml.ctrl(self.W_var_block(b2), control=anc)
            qml.ctrl(qml.adjoint(self.W_var_block(b1)), control=anc)
            qml.Hadamard(wires=anc)
            return qml.expval(qml.PauliZ(anc))

        return qnode(b1, b2)

    def eval_re(self, b1, b2):
        return self.compute(b1, b2)

class DeltaTerm(HadamardTest):
    '''
    DELTA_{k1,k2} = \langle 0|U_{k1}^\dagger W_{k2}|0\rangle
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anc = self.n_input_qubit
        self._qnode_cached = None

   # ----------- qnode path (NEW) -----------
    # @lru_cache(None)
    def compute(self, bk, phase):
        if self.dev is None:
            raise RuntimeError("DeltaTerm: runtime not bound. Call bind_runtime(...)")

        anc = self.anc
        dev = self.dev
        interface = self.interface
        diff_method = self.diff_method

        # def U_dag():
        #     return qml.adjoint(self.U_op)()

        @qml.qnode(dev, interface=interface, diff_method=diff_method)
        def qnode(bk, phase):
            qml.Hadamard(wires=anc)
            qml.PhaseShift(phase, wires=anc)
            qml.ctrl(self.U_op, control=anc)()
            qml.ctrl(qml.adjoint(self.W_var_block(bk)), control=anc)
            qml.Hadamard(wires=anc)
            return qml.expval(qml.PauliZ(anc))

        return qnode(bk, phase)
    
    def eval(self, bk, phase):
        return self.compute(bk, phase)

    def eval_re(self, bk):
        return self.eval(bk, 0.0)

    def eval_im(self, bk):
        return self.eval(bk, -np.pi / 2)

class ZetaTerm(HadamardTest):
    '''
    ZETA_{l}^{k}  = \text{Re}(\langle 0|V_k^\dagger A_{k,l} W_k|0 \rangle)
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anc = self.n_input_qubit

    # ----------- qnode path (NEW) -----------
    # @lru_cache(None)
    def compute(self, alphas, betak, l: int):
        if self.dev is None:
            raise RuntimeError("ZetaTerm: runtime not bound. Call bind_runtime(dev, ...).")

        anc = self.anc
        dev = self.dev
        interface = self.interface
        diff_method = self.diff_method

        def A_l_dag():
            return qml.adjoint(self.A_gates[l]())

        @qml.qnode(dev, interface=interface, diff_method=diff_method)
        def qnode(a, bk):
            qml.Hadamard(wires=anc)
            qml.ctrl(self.W_var_block(bk), control=anc)
            qml.ctrl(A_l_dag, control=anc)()
            qml.ctrl(qml.adjoint(self.V_var_block(a)), control=anc)
            qml.Hadamard(wires=anc)
            return qml.expval(qml.PauliZ(anc))

        return qnode(alphas, betak)

    def eval_re(self, alphas, betak, l: int):
        return self.compute(alphas, betak, l)

class TauTerm(HadamardTest):
    '''
    TAU_{l}^k = \langle 0|U_k^\dagger A_{k,l} V_k | 0\rangle
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anc = self.n_input_qubit

   # ----------- qnode path (NEW) -----------
    # @lru_cache(None)
    def compute(self, alphas, l, phase):
        if self.dev is None:
            raise RuntimeError("TauTerm: runtime not bound. Call bind_runtime(dev, ...).")

        anc = self.anc
        dev = self.dev
        interface = self.interface
        diff_method = self.diff_method

        def A_l_dag():
            return qml.adjoint(self.A_gates[l]())

        @qml.qnode(dev, interface=interface, diff_method=diff_method)
        def qnode(alphas, phase):
            qml.Hadamard(wires=anc)
            qml.PhaseShift(phase, wires=anc)
            qml.ctrl(self.U_op, control=anc)()
            qml.ctrl(A_l_dag, control=anc)()
            # Have confirm that do not need to put (alphas) at the end. But for qml.ctrl(Al_dag, control=anc)() need the ()
            qml.ctrl(qml.adjoint(self.V_var_block(alphas)), control=anc)
            qml.Hadamard(wires=anc)
            return qml.expval(qml.PauliZ(anc))

        return qnode(alphas, phase)

    def eval(self, alphas, l: int, phase):
        return self.compute(alphas, l, phase)

    def eval_re(self, alphas, l: int):
        return self.eval(alphas, l, 0.0)

    def eval_im(self, alphas, l: int):
        return self.eval(alphas, l, -np.pi / 2)

class BetaTerm(HadamardTest):
    '''
    BETA_{l,lp}^k = \langle 0|,V_k^{\dagger} A_{k,l}^\dagger A_{k,lp} V_k\,|0\rangle
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anc = self.n_input_qubit

   # ----------- qnode path (NEW) -----------
    # @lru_cache(None)
    def compute(self, alphas, phase, l: int, lp: int):
        if self.dev is None:
            raise RuntimeError("BetaTerm: runtime not bound. Call bind_runtime(dev, ...).")

        anc = self.anc
        dev = self.dev
        interface = self.interface
        diff_method = self.diff_method

        def A_lp_dagger():
            return qml.adjoint(self.A_gates[lp]())

        def A_l():
            return self.A_gates[l]()

        @qml.qnode(dev, interface=interface, diff_method=diff_method)
        def qnode(alphas, phase):
            qml.Hadamard(wires=anc)
            qml.PhaseShift(phase, wires=anc)
            self.V_var_block(alphas)
            qml.ctrl(A_l, control=anc)()
            qml.ctrl(A_lp_dagger, control=anc)()
            qml.Hadamard(wires=anc)
            return qml.expval(qml.PauliZ(anc))

        return qnode(alphas, phase)

    def eval(self, alphas, l: int, lp: int, phase):
        return self.compute(alphas, phase, l, lp)

    def eval_re(self, alphas, l: int, lp: int):
        return self.compute(alphas, 0.0, l, lp)

    def eval_im(self, alphas, l: int, lp: int):
        return self.eval(alphas, l, lp, -np.pi / 2)

def make_term_bundle(
    n_input_qubit: int,
    U_op,
    A_gates,
    *,
    dev=None,
    interface="jax",
    diff_method="adjoint",
    use_jit=True,
):
    """
    If dev is provided -> bind_runtime so eval_* works (QNode path).
    Tape path still works regardless.
    """
    omega = OmegaTerm(n_input_qubit=n_input_qubit, U_op=U_op, A_gates=A_gates)
    delta = DeltaTerm(n_input_qubit=n_input_qubit, U_op=U_op, A_gates=A_gates)
    zeta  = ZetaTerm(n_input_qubit=n_input_qubit, U_op=U_op, A_gates=A_gates)
    tau   = TauTerm(n_input_qubit=n_input_qubit, U_op=U_op, A_gates=A_gates)
    beta  = BetaTerm(n_input_qubit=n_input_qubit, U_op=U_op, A_gates=A_gates)

    if dev is not None:
        omega.bind_runtime(dev, interface=interface, diff_method=diff_method, use_jit=use_jit)
        delta.bind_runtime(dev, interface=interface, diff_method=diff_method, use_jit=use_jit)
        zeta.bind_runtime(dev, interface=interface, diff_method=diff_method, use_jit=use_jit)
        tau.bind_runtime(dev, interface=interface, diff_method=diff_method, use_jit=use_jit)
        beta.bind_runtime(dev, interface=interface, diff_method=diff_method, use_jit=use_jit)

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
