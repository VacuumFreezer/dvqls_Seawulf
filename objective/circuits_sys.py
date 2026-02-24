from pennylane import numpy as pnp
import numpy as np
from typing import Callable, Any
import pennylane as qml


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
    """ Re(ω) with ω = ⟨0| W₁† W₂ |0⟩ via a Hadamard test. """
    def compute(self, betas, k1: int, k2: int) -> float:
        # unpack β for local agent and row neighbors. 
        betak1, betak2 = betas[f"W{k1}"], (betas[f"W{k2}"])

        n = self.n_input_qubit 
        anc = n
        dev = dev_cpu(n + 1)

        @qml.qnode(dev, interface="autograd", diff_method = "best")
        def _circuit(b1, b2):
            qml.Hadamard(wires=anc)
            # controlled W2, then controlled W1^\dagger
            qml.ctrl(self.W_var_block, control=anc)(b2)
            qml.ctrl(qml.adjoint(self.W_var_block), control=anc)(b1)
            qml.Hadamard(wires=anc)
            return qml.expval(qml.PauliZ(anc))

        return _circuit(betak1, betak2)


class DeltaTerm(HadamardTest):
    """
    δ_k = <0|U^† W_k|0> (complex) via two Hadamard tests.
    Index k starts from 1, denote the k-th agent in the column partition.    
    Do not need the index of U here, since delta term is computeuated on a single agentm with the same U. Only messanger W need to specify index k.
    """
    def compute(self, betas, k: int) -> complex:

        beta_k = betas[f"W{k}"]
        n = self.n_input_qubit  
        anc = n
        dev = dev_cpu(n + 1)

        def U_dag():
            qml.adjoint(self.U_op)()
        # ---------- Real part ----------
        @qml.qnode(dev, interface="autograd", diff_method="best")
        def _real(bk):
            qml.Hadamard(wires=anc)
            qml.ctrl(self.W_var_block, control=anc)(bk)
            qml.ctrl(U_dag, control=anc)()
            qml.Hadamard(wires=anc)
            return qml.expval(qml.PauliZ(anc))

        # ---------- Imaginary part ----------
        @qml.qnode(dev, interface="autograd", diff_method="best")
        def _imag(bk):
            qml.Hadamard(wires=anc)
            qml.PhaseShift(-np.pi/2, wires=anc)
            qml.ctrl(self.W_var_block, control=anc)(bk)
            qml.ctrl(U_dag, control=anc)()
            qml.Hadamard(wires=anc)
            return qml.expval(qml.PauliZ(anc))

        return _real(beta_k) + 1j * _imag(beta_k)

class ZetaTerm(HadamardTest):
    """ Re(ζ_l^{(k)}) with ζ_l^{(k)} = <0| V1^† A_{1,l}^† W_k |0> via a Hadamard test. """
    def compute(self, alphas, betas, l: int, k: int) -> float:

        beta_k = betas[f"W{k}"]
        n = self.n_input_qubit 
        anc = n
        dev = dev_cpu(n + 1)

        # Local wrapper to apply A_{1,l} so we can take its adjoint
        def A_l_dag():
            qml.adjoint(self.A_op)(l)

        @qml.qnode(dev, interface="autograd", diff_method="best")
        def _circuit(bk, a):
            qml.Hadamard(wires=anc)
            qml.ctrl(self.W_var_block, control=anc)(bk)
            qml.ctrl(A_l_dag, control=anc)()
            qml.ctrl(qml.adjoint(self.V_var_block), control=anc)(a)
            qml.Hadamard(wires=anc)
            return qml.expval(qml.PauliZ(anc)) 

        return _circuit(beta_k, alphas)

class TauTerm(HadamardTest):
    """τ_l = <0|U^† A_l V|0> (complex)."""
    def compute(self, alphas, l: int) -> complex:
        n = self.n_input_qubit 
        anc = n
        dev = dev_cpu(n + 1)

        def op_chain():
            # U_dagger -> A_l_dagger
            self.U_op()
            qml.adjoint(self.A_op)(l)

        @qml.qnode(dev, interface="autograd", diff_method="best")
        def _real(a):
            qml.Hadamard(wires=anc)
            qml.ctrl(op_chain, control=anc)()   
            qml.ctrl(qml.adjoint(self.V_var_block), control=anc)(a)
            qml.Hadamard(wires=anc)
            return qml.expval(qml.PauliZ(anc))

        @qml.qnode(dev, interface="autograd", diff_method="best")
        def _imag(a):
            qml.Hadamard(wires=anc)
            qml.PhaseShift(-np.pi/2, wires=anc)
            qml.ctrl(op_chain, control=anc)()
            qml.ctrl(qml.adjoint(self.V_var_block), control=anc)(a)
            qml.Hadamard(wires=anc)
            return qml.expval(qml.PauliZ(anc))

        return _real(alphas) + 1j * _imag(alphas)

class BetaTerm(HadamardTest):
    """β_{ll'} = ⟨0|V^† A_{l'}^† A_l V|0⟩ (complex)."""
    def compute(self, alphas, l: int, lp: int) -> complex:
        n = self.n_input_qubit 
        anc = n
        dev = dev_cpu(n + 1)

        def op_chain():
            # A_l then A_lp_dagger
            self.A_op(l)
            qml.adjoint(self.A_op)(lp)

        # ----- Real part -----
        @qml.qnode(dev, interface="autograd", diff_method="best")
        def _real(a):
            qml.Hadamard(wires=anc)
            self.V_var_block(a)                       
            qml.ctrl(op_chain, control=anc)()
            qml.Hadamard(wires=anc)
            return qml.expval(qml.PauliZ(anc))        

        # ----- Imag part (S phase on ancilla) -----
        @qml.qnode(dev, interface="autograd", diff_method="best")
        def _imag(a):
            qml.Hadamard(wires=anc)
            qml.PhaseShift(-np.pi/2, wires=anc)
            self.V_var_block(a)
            qml.ctrl(op_chain, control=anc)()
            qml.Hadamard(wires=anc)
            return qml.expval(qml.PauliZ(anc))        

        return _real(alphas) + 1j * _imag(alphas)

# ========== Term instances ==========
OMEGA = OmegaTerm(n_input_qubit=2)
DELTA = DeltaTerm(n_input_qubit=2)
ZETA  = ZetaTerm(n_input_qubit=2)
TAU   = TauTerm(n_input_qubit=2)
BETA  = BetaTerm(n_input_qubit=2)

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
