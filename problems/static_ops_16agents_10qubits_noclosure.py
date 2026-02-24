from typing import List, Callable, Dict, Any, Tuple
import pennylane as qml
from pennylane import numpy as np

# ==========================================================
# Configuration
# ==========================================================
N_DATA_QUBITS = 5
DATA_WIRES = list(range(N_DATA_QUBITS))   # [0,1,2,...,9]
N_AGENTS = 4

# ==========================================================
# 1. Helper: Automatic Coefficient Calculator
# ==========================================================
def get_coeffs(kappa: float, num_gates: int) -> List[float]:
    """
    Same logic as before:
    - num_gates == 3  -> [Identity, Pauli, Pauli]
    - num_gates == 2  -> [Identity, Pauli]
    """
    if num_gates == 3:
        return [
            2 * (kappa + 1) / (4 * kappa),
            (kappa - 1) / (4 * kappa),
            (kappa - 1) / (4 * kappa),
        ]
    elif num_gates == 2:
        return [
            (kappa + 1) / (2 * kappa),
            (kappa - 1) / (2 * kappa),
        ]
    else:
        return [1.0 / num_gates] * num_gates

# 5qubits
def I():  return qml.Identity(wires=DATA_WIRES[0])

def X0(): return qml.PauliX(wires=0)
def X1(): return qml.PauliX(wires=1)
def X2(): return qml.PauliX(wires=2)
def X3(): return qml.PauliX(wires=3)
def X4(): return qml.PauliX(wires=4)

def Z0(): return qml.PauliZ(wires=0)
def Z1(): return qml.PauliZ(wires=1)
def Z2(): return qml.PauliZ(wires=2)
def Z3(): return qml.PauliZ(wires=3)
def Z4(): return qml.PauliZ(wires=4)


# # ==========================================================
# # 设计原则：
# # - A_ii: [I, Z_i] + kdiag<1  => 对角块谱下界约为 1（另一支是 1/kdiag）
# # - A_ij: 单个 X/Z（不同 wires）打散耦合
# # - 对称：A_ij = A_ji  => 全局更不易出现极小奇异值
# # ==========================================================
RAW_GATES = [
    # Row 0
    [[I, Z0],  [X3],   [Z4],   [X1]],
    # Row 1
    [[X3],     [I, Z1],[X4],   [Z2]],
    # Row 2
    [[Z4],     [X4],   [I, Z2],[X0]],
    # Row 3
    [[X1],     [Z2],   [X0],   [I, Z3]],
]


# # 只对对角块用 kdiag（因为对角块是 len=2，会用 get_coeffs）
# # 经验：kdiag 在 0.08 ~ 0.2 之间，cond 通常比较好控制（>1 且不易爆）
# kdiag = 0.15

# KAPPAS = [
#     [kdiag, 1.0,  1.0,  1.0],
#     [1.0,   kdiag,1.0,  1.0],
#     [1.0,   1.0,  kdiag,1.0],
#     [1.0,   1.0,  1.0,  kdiag],
# ]
# -------- primitives: 每个函数都 return operator --------
# def I():  return qml.Identity(wires=DATA_WIRES[0])

# def X0(): return qml.PauliX(wires=0)
# def X1(): return qml.PauliX(wires=1)
# def X2(): return qml.PauliX(wires=2)
# def X3(): return qml.PauliX(wires=3)
# def X4(): return qml.PauliX(wires=4)
# def X5(): return qml.PauliX(wires=5)
# def X6(): return qml.PauliX(wires=6)

# def Z0(): return qml.PauliZ(wires=0)
# def Z1(): return qml.PauliZ(wires=1)
# def Z2(): return qml.PauliZ(wires=2)
# def Z3(): return qml.PauliZ(wires=3)
# def Z4(): return qml.PauliZ(wires=4)
# def Z5(): return qml.PauliZ(wires=5)
# def Z6(): return qml.PauliZ(wires=6)


# ==========================================================
# 这个 grid 的设计原则：
# - A_ii 用 [I, Z_i] 且 kappa<1 => 对角块有谱下界(>=1)，同时最大本征值=1/kappa
# - A_ij 用单个 X/Z（不同 wire）避免结构性相关
# - 整体对称 => 全局 Hermitian（更不容易出现极小奇异值）
# ==========================================================
# RAW_GATES = [
#     # Row 0
#     [[I, Z0],  [X6],   [Z3],   [X2]],
#     # Row 1
#     [[X6],     [I, Z1],[X3],   [Z3]],
#     # Row 2
#     [[Z3],     [X3],   [I, Z4],[X0]],
#     # Row 3
#     [[X2],     [Z3],   [X0],   [I, Z4]],
# ]

# 对角 kappa 控制“对角块强度”：
# - kappa 越小 => 对角块最大本征值 1/kappa 越大，通常能把全局最小奇异值顶起来，但也会增大最大奇异值
# - 经验上：kdiag=0.05~0.2 很常用（对应 1/kdiag=20~5）
kdiag = 0.2  # 你可以在 0.05~0.2 之间调，通常 cond 会落在 (1,100) 内

KAPPAS = [
    [kdiag, 1.0, 1.0, 1.0],
    [1.0, kdiag, 1.0, 1.0],
    [1.0, 1.0, kdiag, 1.0],
    [1.0, 1.0, 1.0, kdiag],
]

# --- Row 0 (System 1) ---
def U_sys0_ag0(): return qml.Identity(wires=DATA_WIRES)
def U_sys0_ag1(): return qml.PauliX(wires=DATA_WIRES[0])
def U_sys0_ag2(): return qml.PauliZ(wires=DATA_WIRES[1])
def U_sys0_ag3(): return qml.PauliZ(wires=DATA_WIRES[0])

# --- Row 1 (System 2) ---
def U_sys1_ag0(): return qml.PauliZ(wires=DATA_WIRES[1])
def U_sys1_ag1(): return qml.PauliX(wires=DATA_WIRES[1])
def U_sys1_ag2(): return qml.Identity(wires=DATA_WIRES)
def U_sys1_ag3(): return qml.PauliX(wires=DATA_WIRES[0])

# --- Row 2 (System 3) ---
def U_sys2_ag0(): return qml.PauliX(wires=DATA_WIRES[0])
def U_sys2_ag1(): return qml.Identity(wires=DATA_WIRES)
def U_sys2_ag2(): return qml.PauliZ(wires=DATA_WIRES[0])
def U_sys2_ag3(): return qml.PauliX(wires=DATA_WIRES[1])

# --- Row 3 (System 4) ---
def U_sys3_ag0(): return qml.PauliX(wires=DATA_WIRES[1])
def U_sys3_ag1(): return qml.PauliZ(wires=DATA_WIRES[0])
def U_sys3_ag2(): return qml.Identity(wires=DATA_WIRES)
def U_sys3_ag3(): return qml.PauliZ(wires=DATA_WIRES[1])
# # ==========================================================
# # 4. 4x4 Grid of state-prep unitaries U (for b vectors)
# # ==========================================================
# # --- Row 0 (System 0) ---
# def U_sys0_ag0(): return qml.Identity(wires=DATA_WIRES[0])
# def U_sys0_ag1(): return qml.PauliX(wires=DATA_WIRES[2])
# def U_sys0_ag2(): return qml.PauliZ(wires=DATA_WIRES[5])
# def U_sys0_ag3(): return qml.PauliZ(wires=DATA_WIRES[4])

# # --- Row 1 (System 1) ---
# def U_sys1_ag0(): return qml.PauliZ(wires=DATA_WIRES[6])
# def U_sys1_ag1(): return qml.PauliX(wires=DATA_WIRES[4])
# def U_sys1_ag2(): return qml.Identity(wires=DATA_WIRES[0])
# def U_sys1_ag3(): return qml.PauliX(wires=DATA_WIRES[1])

# # --- Row 2 (System 2) ---
# def U_sys2_ag0(): return qml.PauliX(wires=DATA_WIRES[3])
# def U_sys2_ag1(): return qml.Identity(wires=DATA_WIRES[0])
# def U_sys2_ag2(): return qml.PauliZ(wires=DATA_WIRES[2])
# def U_sys2_ag3(): return qml.PauliX(wires=DATA_WIRES[1])

# # --- Row 3 (System 3) ---
# def U_sys3_ag0(): return qml.PauliX(wires=DATA_WIRES[4])
# def U_sys3_ag1(): return qml.PauliZ(wires=DATA_WIRES[5])
# def U_sys3_ag2(): return qml.Identity(wires=DATA_WIRES[0])
# def U_sys3_ag3(): return qml.PauliZ(wires=DATA_WIRES[2])

RAW_B_GATES = [
    [U_sys0_ag0, U_sys0_ag1, U_sys0_ag2, U_sys0_ag3],
    [U_sys1_ag0, U_sys1_ag1, U_sys1_ag2, U_sys1_ag3],
    [U_sys2_ag0, U_sys2_ag1, U_sys2_ag2, U_sys2_ag3],
    [U_sys3_ag0, U_sys3_ag1, U_sys3_ag2, U_sys3_ag3],
]


# ==========================================================
# 5. System Container Class
# ==========================================================
class LinearSystemData:
    def __init__(self, gates_grid, kappas, b_gates_grid):
        self.n = len(gates_grid)
        self.gates_grid = gates_grid
        self.b_gates = b_gates_grid

        # Pre-compute Coefficients and Op Wrappers
        self.coeffs = []
        self.ops = []

        for i in range(self.n):
            row_coeffs = []
            row_ops = []
            for j in range(self.n):
                g_list = gates_grid[i][j]
                k = kappas[i][j]

                row_coeffs.append(get_coeffs(k, len(g_list)))
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


SYSTEM = LinearSystemData(RAW_GATES, KAPPAS, RAW_B_GATES)

if __name__ == "__main__":
    # 1. Get the full 12x12 Matrix A
    A_global = SYSTEM.get_global_matrix()
    print(f"Global Matrix Shape: {A_global.shape}")

    # 2. Get b vectors for System 1 (Agent index 0)
    b_sum, b1, b2, b3 = SYSTEM.get_b_vectors(0)
    print(f"System 1 b_total: {b_sum}")