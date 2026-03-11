from __future__ import annotations

from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


_DEV = {}


def dev_cpu(nwires: int):
    if nwires not in _DEV:
        _DEV[nwires] = qml.device("lightning.qubit", wires=nwires)
    return _DEV[nwires]


class HadamardTest:
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

    def _cluster_scaffold(self):
        for w in range(self.n_input_qubit):
            qml.Hadamard(wires=w)
        for left in range(1, self.n_input_qubit - 1):
            qml.CZ(wires=[left, left + 1])

    def W_var_block(self, betas):
        self._cluster_scaffold()
        for layer in range(int(betas.shape[0])):
            for wire in range(self.n_input_qubit):
                qml.RZ(betas[layer, wire], wires=wire)

    def V_var_block(self, alphas):
        self._cluster_scaffold()
        for layer in range(int(alphas.shape[0])):
            for wire in range(self.n_input_qubit):
                qml.RZ(alphas[layer, wire], wires=wire)


class OmegaTerm(HadamardTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anc = self.n_input_qubit

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
            qml.ctrl(self.W_var_block, control=anc)(b2)
            qml.ctrl(qml.adjoint(self.W_var_block), control=anc)(b1)
            qml.Hadamard(wires=anc)
            return qml.expval(qml.PauliZ(anc))

        return qnode(b1, b2)

    def eval_re(self, b1, b2):
        return self.compute(b1, b2)


class DeltaTerm(HadamardTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anc = self.n_input_qubit

    def compute(self, bk, phase):
        if self.dev is None:
            raise RuntimeError("DeltaTerm: runtime not bound. Call bind_runtime(...)")

        anc = self.anc
        dev = self.dev
        interface = self.interface
        diff_method = self.diff_method

        @qml.qnode(dev, interface=interface, diff_method=diff_method)
        def qnode(bk, phase):
            qml.Hadamard(wires=anc)
            qml.PhaseShift(phase, wires=anc)
            qml.ctrl(self.U_op, control=anc)()
            qml.ctrl(qml.adjoint(self.W_var_block), control=anc)(bk)
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anc = self.n_input_qubit

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
            qml.ctrl(self.W_var_block, control=anc)(bk)
            qml.ctrl(A_l_dag, control=anc)()
            qml.ctrl(qml.adjoint(self.V_var_block), control=anc)(a)
            qml.Hadamard(wires=anc)
            return qml.expval(qml.PauliZ(anc))

        return qnode(alphas, betak)

    def eval_re(self, alphas, betak, l: int):
        return self.compute(alphas, betak, l)


class TauTerm(HadamardTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anc = self.n_input_qubit

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
            qml.ctrl(qml.adjoint(self.V_var_block), control=anc)(alphas)
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anc = self.n_input_qubit

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
    omega = OmegaTerm(n_input_qubit=n_input_qubit, U_op=U_op, A_gates=A_gates)
    delta = DeltaTerm(n_input_qubit=n_input_qubit, U_op=U_op, A_gates=A_gates)
    zeta = ZetaTerm(n_input_qubit=n_input_qubit, U_op=U_op, A_gates=A_gates)
    tau = TauTerm(n_input_qubit=n_input_qubit, U_op=U_op, A_gates=A_gates)
    beta = BetaTerm(n_input_qubit=n_input_qubit, U_op=U_op, A_gates=A_gates)

    if dev is not None:
        omega.bind_runtime(dev, interface=interface, diff_method=diff_method, use_jit=use_jit)
        delta.bind_runtime(dev, interface=interface, diff_method=diff_method, use_jit=use_jit)
        zeta.bind_runtime(dev, interface=interface, diff_method=diff_method, use_jit=use_jit)
        tau.bind_runtime(dev, interface=interface, diff_method=diff_method, use_jit=use_jit)
        beta.bind_runtime(dev, interface=interface, diff_method=diff_method, use_jit=use_jit)

    return {"OMEGA": omega, "DELTA": delta, "ZETA": zeta, "TAU": tau, "BETA": beta}
