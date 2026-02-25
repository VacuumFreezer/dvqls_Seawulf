from typing import List, Callable, Dict, Any, Tuple
import pennylane as qml
from pennylane import numpy as np

# =========================
# Config
# =========================
N_DATA_QUBITS = 7
DATA_WIRES = list(range(N_DATA_QUBITS))
N_AGENTS = 4

# Block-friendly params (you can change)
J = 0.25     # controls +/- J * I shift among diagonal blocks (from Z0_idx Z1_idx)
h = 0.25     # controls +/- h * Z0 (data boundary term, from Z1_idx coupled to data)
eta = 1.5
zeta = 1.0  # 注意：你如果在 LinearSystemData.__init__ 里吸收 1/zeta，就别在这里除

# =========================
# Gate factories (return operator)
# =========================
def I():
    return qml.Identity(wires=DATA_WIRES[0])

def X(k: int):
    return (lambda k=k: qml.PauliX(wires=k))

def Z(k: int):
    return (lambda k=k: qml.PauliZ(wires=k))

# Convenience: boundary term uses Z on the first data qubit (wire 0 in data-space)
Z0 = Z(0)

# =========================
# Block-friendly diagonal blocks as (gates, coeffs)
#
# 物理含义（等价于 9-qubit 的简化模型，但这里直接写成 4x4 blocks）：
#   A = (1/zeta) * [[D00, I, I, 0],
#                   [I, D01, 0, I],
#                   [I, 0, D10, I],
#                   [0, I, I, D11]]
#
# 对角块：
#   D = (eta + s01*J)*I  + (s12*h)*Z0  + sum_k X_k
#
# s01 来自 index 的 Z0_idx Z1_idx 在 |b0 b1> 上的本征值：
#   +1 if b0=b1 else -1
# s12 来自 index 的 Z1_idx 在 |b0 b1> 上的本征值：
#   +1 if b1=0 else -1
#
# mapping:
#   |00>: (s01=+1, s12=+1)
#   |01>: (s01=-1, s12=-1)
#   |10>: (s01=-1, s12=+1)
#   |11>: (s01=+1, s12=-1)
# =========================
def make_D_block(s01: float, s12: float):
    gates: List[Callable[[], qml.operation.Operator]] = []
    coeffs: List[float] = []

    # Identity with coefficient (eta + s01*J)
    gates.append(I)
    coeffs.append(eta + s01 * J)

    # boundary term ±h Z0 (data)
    gates.append(Z0)
    coeffs.append(s12 * h)

    # sum X_k over data qubits
    for k in range(N_DATA_QUBITS):
        gates.append(X(k))
        coeffs.append(1.0)

    return gates, coeffs

# 4 diagonal blocks
D00_g, D00_c = make_D_block(s01=+1.0, s12=+1.0)
D01_g, D01_c = make_D_block(s01=-1.0, s12=-1.0)
D10_g, D10_c = make_D_block(s01=-1.0, s12=+1.0)
D11_g, D11_c = make_D_block(s01=+1.0, s12=-1.0)

# Off-diagonal identity/zero blocks
Id_g, Id_c = [I], [1.0]
Zr_g, Zr_c = [I], [0.0]   # represent zero block as 0 * I

# The clean 4x4 block pattern (order: 00,01,10,11)
RAW_GATES = [
    [D00_g, Id_g, Id_g, Zr_g],
    [Id_g,  D01_g, Zr_g, Id_g],
    [Id_g,  Zr_g,  D10_g, Id_g],
    [Zr_g,  Id_g,  Id_g,  D11_g],
]

RAW_COEFFS = [
    [D00_c, Id_c, Id_c, Zr_c],
    [Id_c,  D01_c, Zr_c, Id_c],
    [Id_c,  Zr_c,  D10_c, Id_c],
    [Zr_c,  Id_c,  Id_c,  D11_c],
]

# =========================
# b prep gates: H^{\otimes N_DATA_QUBITS} everywhere
# =========================
def H_all():
    return qml.prod(*[qml.Hadamard(wires=w) for w in DATA_WIRES])

RAW_B_GATES = [[H_all for _ in range(N_AGENTS)] for __ in range(N_AGENTS)]


# ==========================================================
# 5. System Container Class
# ==========================================================
class LinearSystemData:
    def __init__(self, gates_grid, kappas_or_coeffs_grid, b_gates_grid, zeta: float = 1.0):
        self.n = len(gates_grid)
        self.gates_grid = gates_grid
        self.b_gates = b_gates_grid
        self.zeta = zeta

        # Pre-compute Coefficients and Op Wrappers
        self.coeffs = []
        self.ops = []

        for i in range(self.n):
            row_coeffs = []
            row_ops = []
            for j in range(self.n):
                g_list = gates_grid[i][j]
                c_list = kappas_or_coeffs_grid[i][j]   # 这里 kappas 实际上就是 RAW_COEFFS
                if len(c_list) != len(g_list):
                    raise ValueError(f"Coeff length mismatch at ({i},{j}): "
                                    f"len(coeffs)={len(c_list)} vs len(gates)={len(g_list)}")
                if self.zeta != 1.0:
                    c_list = [c / self.zeta for c in c_list]

                row_coeffs.append(c_list)
                row_ops.append(self._make_wrapper(g_list))

            self.coeffs.append(row_coeffs)
            self.ops.append(row_ops)

        # optional: matrix cache (important for 10 wires)
        self._mat_cache: Dict[int, np.ndarray] = {}

    def _make_wrapper(self, gate_factories):
        def wrapper(l):
            return gate_factories[int(l)]() 
        return wrapper

    def _as_matrix(self, gate_fn: Callable):
        """Convert a single gate function to (2^n x 2^n) matrix with caching."""
        key = id(gate_fn)
        if key in self._mat_cache:
            return self._mat_cache[key]

        def qfunc():
            gate_fn()

        M = np.array(qml.matrix(qfunc, wire_order=DATA_WIRES)())
        self._mat_cache[key] = M
        return M

    def get_global_matrix(self):
        """
        Full global matrix A:
        size = (n * 2^N_DATA_QUBITS, n * 2^N_DATA_QUBITS)
        """
        block_rows = []
        for i in range(self.n):
            block_cols = []
            for j in range(self.n):
                gates = self.gates_grid[i][j]
                coeffs = self.coeffs[i][j]

                mats = [self._as_matrix(g) for g in gates]
                combined = np.zeros_like(mats[0], dtype=complex)
                for c, M in zip(coeffs, mats):
                    combined = combined + c * M

                block_cols.append(combined)
            block_rows.append(block_cols)

        return np.block(block_rows)

    def get_b_vectors(self, sys_id: int):
        """
        Returns: (b_total, b_agent0, b_agent1, b_agent2, b_agent3)
        """
        u_gates = self.b_gates[sys_id]
        u_mats = [self._as_matrix(u) for u in u_gates]

        dim = 2 ** len(DATA_WIRES)
        ket0 = np.zeros(dim, dtype=float)
        ket0[0] = 1.0

        b_vecs = [M @ ket0 for M in u_mats]
        b_total = sum(b_vecs)
        return (b_total, *b_vecs)

    def get_global_b_vector(self):
        all_b_sums = []
        for sys_id in range(self.n):
            all_b_sums.append(self.get_b_vectors(sys_id)[0])
        return np.concatenate(all_b_sums)

    def get_b_op(self, sys_id: int, agent_id: int):
        return self.b_gates[sys_id][agent_id]


SYSTEM = LinearSystemData(RAW_GATES, RAW_COEFFS, RAW_B_GATES, zeta=zeta)

if __name__ == "__main__":
    # 1. Get the full 12x12 Matrix A
    A_global = SYSTEM.get_global_matrix()
    print(f"Global Matrix Shape: {A_global.shape}")

    # 2. Get b vectors for System 1 (Agent index 0)
    b_sum, b1, b2, b3 = SYSTEM.get_b_vectors(0)
    print(f"System 1 b_total: {b_sum}")