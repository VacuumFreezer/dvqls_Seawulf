from __future__ import annotations

from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


_DEV = {}

ANSATZ_CLUSTER_RZ = "cluster_rz"
ANSATZ_CLUSTER_RY = "cluster_ry"
ANSATZ_BRICKWALL_RY_CZ = "brickwall_ry_cz"
ANSATZ_NO_HADAMARD_RY = "no_hadamard_ry"
ANSATZ_CLUSTER_RZ_LOCAL_RY = "cluster_rz_local_ry"
ANSATZ_CLUSTER_LOCAL_RY = "cluster_local_ry"
VALID_ANSATZ_KINDS = (
    ANSATZ_CLUSTER_RZ,
    ANSATZ_CLUSTER_RY,
    ANSATZ_BRICKWALL_RY_CZ,
    ANSATZ_NO_HADAMARD_RY,
    ANSATZ_CLUSTER_RZ_LOCAL_RY,
    ANSATZ_CLUSTER_LOCAL_RY,
)


def normalize_ansatz_kind(ansatz_kind: str) -> str:
    kind = str(ansatz_kind).strip().lower()
    if kind not in VALID_ANSATZ_KINDS:
        raise ValueError(f"Unsupported ansatz {ansatz_kind!r}. Expected one of {VALID_ANSATZ_KINDS}.")
    return kind


def describe_ansatz(ansatz_kind: str) -> str:
    kind = normalize_ansatz_kind(ansatz_kind)
    if kind == ANSATZ_CLUSTER_RZ:
        return "Hadamard scaffold + open-chain CZ + trainable RZ"
    if kind == ANSATZ_CLUSTER_RY:
        return "Hadamard scaffold + open-chain CZ + trainable RY"
    if kind == ANSATZ_BRICKWALL_RY_CZ:
        return "Trainable RY layer followed by open-chain CZ"
    if kind == ANSATZ_CLUSTER_RZ_LOCAL_RY:
        return "Hadamard scaffold + open-chain CZ + trainable RZ with local RY correction"
    if kind == ANSATZ_CLUSTER_LOCAL_RY:
        return "Hadamard scaffold + open-chain CZ + local trainable RY support"
    return "Open-chain CZ only + trainable RY"


def _apply_open_chain_cz(n_input_qubit: int):
    for left in range(1, n_input_qubit - 1):
        qml.CZ(wires=[left, left + 1])


def _normalize_scaffold_edges(scaffold_edges, n_input_qubit: int) -> tuple[tuple[int, int], ...]:
    if scaffold_edges is None:
        return tuple((left, left + 1) for left in range(1, n_input_qubit - 1))

    edges = []
    for pair in scaffold_edges:
        if len(pair) != 2:
            raise ValueError(f"Invalid scaffold edge {pair!r}; expected a pair of local wire indices.")
        left, right = int(pair[0]), int(pair[1])
        if left < 0 or right < 0 or left >= int(n_input_qubit) or right >= int(n_input_qubit):
            raise ValueError(
                f"Scaffold edge {(left, right)} is out of range for n_input_qubit={int(n_input_qubit)}."
            )
        if left == right:
            raise ValueError(f"Scaffold edge {(left, right)} must connect two distinct wires.")
        edges.append((left, right))
    return tuple(edges)


def _apply_scaffold_edges(scaffold_edges):
    for left, right in scaffold_edges:
        qml.CZ(wires=[left, right])


def _apply_scaffold_edges_reverse(scaffold_edges):
    for left, right in reversed(tuple(scaffold_edges)):
        qml.CZ(wires=[left, right])


def _normalize_local_wire_subset(local_ry_support, n_input_qubit: int) -> tuple[int, ...]:
    if local_ry_support is None:
        return ()

    subset = []
    for wire in local_ry_support:
        idx = int(wire)
        if idx < 0 or idx >= int(n_input_qubit):
            raise ValueError(
                f"Local RY support wire {idx} is out of range for n_input_qubit={int(n_input_qubit)}."
            )
        subset.append(idx)
    return tuple(sorted(set(subset)))


def _apply_rotation_layer(weights_row, n_input_qubit: int, *, kind: str, local_ry_support: tuple[int, ...]):
    if kind == ANSATZ_CLUSTER_LOCAL_RY:
        for wire in local_ry_support:
            qml.RY(weights_row[wire], wires=wire)
        return

    for wire in range(n_input_qubit):
        if kind in (ANSATZ_CLUSTER_RZ, ANSATZ_CLUSTER_RZ_LOCAL_RY):
            qml.RZ(weights_row[wire], wires=wire)
        else:
            qml.RY(weights_row[wire], wires=wire)

    if kind == ANSATZ_CLUSTER_RZ_LOCAL_RY:
        for wire in local_ry_support:
            qml.RY(weights_row[wire], wires=wire)


def _apply_rotation_layer_inverse(weights_row, n_input_qubit: int, *, kind: str, local_ry_support: tuple[int, ...]):
    if kind == ANSATZ_CLUSTER_LOCAL_RY:
        for wire in reversed(local_ry_support):
            qml.RY(-weights_row[wire], wires=wire)
        return

    if kind == ANSATZ_CLUSTER_RZ_LOCAL_RY:
        for wire in reversed(local_ry_support):
            qml.RY(-weights_row[wire], wires=wire)

    for wire in range(n_input_qubit - 1, -1, -1):
        if kind in (ANSATZ_CLUSTER_RZ, ANSATZ_CLUSTER_RZ_LOCAL_RY):
            qml.RZ(-weights_row[wire], wires=wire)
        else:
            qml.RY(-weights_row[wire], wires=wire)


def apply_selected_ansatz(
    weights,
    n_input_qubit: int,
    *,
    ansatz_kind: str = ANSATZ_CLUSTER_RZ,
    repeat_cz_each_layer: bool = False,
    local_ry_support=None,
    scaffold_edges=None,
):
    kind = normalize_ansatz_kind(ansatz_kind)
    local_ry_support = _normalize_local_wire_subset(local_ry_support, n_input_qubit)
    scaffold_edges = _normalize_scaffold_edges(scaffold_edges, n_input_qubit)

    if kind in (ANSATZ_CLUSTER_RZ_LOCAL_RY, ANSATZ_CLUSTER_LOCAL_RY) and not local_ry_support:
        raise ValueError(
            f"Ansatz `{kind}` requires a non-empty local_ry_support wire set."
        )

    if kind == ANSATZ_BRICKWALL_RY_CZ:
        for layer in range(int(weights.shape[0])):
            for wire in range(n_input_qubit):
                qml.RY(weights[layer, wire], wires=wire)
            _apply_scaffold_edges(scaffold_edges)
        return

    if kind != ANSATZ_NO_HADAMARD_RY:
        for w in range(n_input_qubit):
            qml.Hadamard(wires=w)

    if not repeat_cz_each_layer:
        _apply_scaffold_edges(scaffold_edges)

    for layer in range(int(weights.shape[0])):
        if repeat_cz_each_layer:
            _apply_scaffold_edges(scaffold_edges)
        _apply_rotation_layer(weights[layer], n_input_qubit, kind=kind, local_ry_support=local_ry_support)


def apply_selected_ansatz_inverse(
    weights,
    n_input_qubit: int,
    *,
    ansatz_kind: str = ANSATZ_CLUSTER_RZ,
    repeat_cz_each_layer: bool = False,
    local_ry_support=None,
    scaffold_edges=None,
):
    kind = normalize_ansatz_kind(ansatz_kind)
    local_ry_support = _normalize_local_wire_subset(local_ry_support, n_input_qubit)
    scaffold_edges = _normalize_scaffold_edges(scaffold_edges, n_input_qubit)

    if kind in (ANSATZ_CLUSTER_RZ_LOCAL_RY, ANSATZ_CLUSTER_LOCAL_RY) and not local_ry_support:
        raise ValueError(f"Ansatz `{kind}` requires a non-empty local_ry_support wire set.")

    if kind == ANSATZ_BRICKWALL_RY_CZ:
        for layer in range(int(weights.shape[0]) - 1, -1, -1):
            _apply_scaffold_edges_reverse(scaffold_edges)
            for wire in range(n_input_qubit - 1, -1, -1):
                qml.RY(-weights[layer, wire], wires=wire)
        return

    for layer in range(int(weights.shape[0]) - 1, -1, -1):
        _apply_rotation_layer_inverse(weights[layer], n_input_qubit, kind=kind, local_ry_support=local_ry_support)
        if repeat_cz_each_layer:
            _apply_scaffold_edges_reverse(scaffold_edges)

    if not repeat_cz_each_layer:
        _apply_scaffold_edges_reverse(scaffold_edges)

    if kind != ANSATZ_NO_HADAMARD_RY:
        for w in range(n_input_qubit - 1, -1, -1):
            qml.Hadamard(wires=w)


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
        ansatz_kind: str = ANSATZ_CLUSTER_RZ,
        repeat_cz_each_layer: bool = False,
        local_ry_support=None,
        scaffold_edges=None,
    ):
        self.n_input_qubit = n_input_qubit
        self.U_op = U_op
        self.A_gates = tuple(A_gates)
        self.ansatz_kind = normalize_ansatz_kind(ansatz_kind)
        self.repeat_cz_each_layer = bool(repeat_cz_each_layer)
        self.local_ry_support = _normalize_local_wire_subset(local_ry_support, n_input_qubit)
        self.scaffold_edges = _normalize_scaffold_edges(scaffold_edges, n_input_qubit)

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
        apply_selected_ansatz(
            jnp.zeros((0, self.n_input_qubit), dtype=jnp.float64),
            self.n_input_qubit,
            ansatz_kind=self.ansatz_kind,
            repeat_cz_each_layer=self.repeat_cz_each_layer,
            local_ry_support=self.local_ry_support,
            scaffold_edges=self.scaffold_edges,
        )

    def W_var_block(self, betas):
        apply_selected_ansatz(
            betas,
            self.n_input_qubit,
            ansatz_kind=self.ansatz_kind,
            repeat_cz_each_layer=self.repeat_cz_each_layer,
            local_ry_support=self.local_ry_support,
            scaffold_edges=self.scaffold_edges,
        )

    def W_var_block_inverse(self, betas):
        apply_selected_ansatz_inverse(
            betas,
            self.n_input_qubit,
            ansatz_kind=self.ansatz_kind,
            repeat_cz_each_layer=self.repeat_cz_each_layer,
            local_ry_support=self.local_ry_support,
            scaffold_edges=self.scaffold_edges,
        )

    def V_var_block(self, alphas):
        apply_selected_ansatz(
            alphas,
            self.n_input_qubit,
            ansatz_kind=self.ansatz_kind,
            repeat_cz_each_layer=self.repeat_cz_each_layer,
            local_ry_support=self.local_ry_support,
            scaffold_edges=self.scaffold_edges,
        )

    def V_var_block_inverse(self, alphas):
        apply_selected_ansatz_inverse(
            alphas,
            self.n_input_qubit,
            ansatz_kind=self.ansatz_kind,
            repeat_cz_each_layer=self.repeat_cz_each_layer,
            local_ry_support=self.local_ry_support,
            scaffold_edges=self.scaffold_edges,
        )


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
            qml.ctrl(self.W_var_block_inverse, control=anc)(b1)
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
            qml.ctrl(self.W_var_block_inverse, control=anc)(bk)
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
            qml.ctrl(self.V_var_block_inverse, control=anc)(a)
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
            qml.ctrl(self.V_var_block_inverse, control=anc)(alphas)
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
    ansatz_kind=ANSATZ_CLUSTER_RZ,
    repeat_cz_each_layer=False,
    local_ry_support=None,
    scaffold_edges=None,
    dev=None,
    interface="jax",
    diff_method="adjoint",
    use_jit=True,
):
    omega = OmegaTerm(
        n_input_qubit=n_input_qubit,
        U_op=U_op,
        A_gates=A_gates,
        ansatz_kind=ansatz_kind,
        repeat_cz_each_layer=repeat_cz_each_layer,
        local_ry_support=local_ry_support,
        scaffold_edges=scaffold_edges,
    )
    delta = DeltaTerm(
        n_input_qubit=n_input_qubit,
        U_op=U_op,
        A_gates=A_gates,
        ansatz_kind=ansatz_kind,
        repeat_cz_each_layer=repeat_cz_each_layer,
        local_ry_support=local_ry_support,
        scaffold_edges=scaffold_edges,
    )
    zeta = ZetaTerm(
        n_input_qubit=n_input_qubit,
        U_op=U_op,
        A_gates=A_gates,
        ansatz_kind=ansatz_kind,
        repeat_cz_each_layer=repeat_cz_each_layer,
        local_ry_support=local_ry_support,
        scaffold_edges=scaffold_edges,
    )
    tau = TauTerm(
        n_input_qubit=n_input_qubit,
        U_op=U_op,
        A_gates=A_gates,
        ansatz_kind=ansatz_kind,
        repeat_cz_each_layer=repeat_cz_each_layer,
        local_ry_support=local_ry_support,
        scaffold_edges=scaffold_edges,
    )
    beta = BetaTerm(
        n_input_qubit=n_input_qubit,
        U_op=U_op,
        A_gates=A_gates,
        ansatz_kind=ansatz_kind,
        repeat_cz_each_layer=repeat_cz_each_layer,
        local_ry_support=local_ry_support,
        scaffold_edges=scaffold_edges,
    )

    if dev is not None:
        omega.bind_runtime(dev, interface=interface, diff_method=diff_method, use_jit=use_jit)
        delta.bind_runtime(dev, interface=interface, diff_method=diff_method, use_jit=use_jit)
        zeta.bind_runtime(dev, interface=interface, diff_method=diff_method, use_jit=use_jit)
        tau.bind_runtime(dev, interface=interface, diff_method=diff_method, use_jit=use_jit)
        beta.bind_runtime(dev, interface=interface, diff_method=diff_method, use_jit=use_jit)

    return {"OMEGA": omega, "DELTA": delta, "ZETA": zeta, "TAU": tau, "BETA": beta}
