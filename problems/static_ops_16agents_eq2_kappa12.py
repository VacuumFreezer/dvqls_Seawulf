from typing import List, Callable
import pennylane as qml
from pennylane import numpy as np

# Configuration
DATA_WIRES = [0, 1]
N_AGENTS = 4 # Change this to 9 or any number easily

# ==========================================
# 1. Helper: Automatic Coefficient Calculator
# ==========================================
def get_coeffs(kappa: float, num_gates: int) -> List[float]:
    """
    Automatically generates coefficients based on gate count and kappa
    to match your logic (Identity term vs Pauli terms).
    """
    if num_gates == 3:
        # Logic for [Identity, Pauli, Pauli]
        return [
            2 * (kappa + 1) / (4 * kappa),
            (kappa - 1) / (4 * kappa),
            (kappa - 1) / (4 * kappa)
        ]
    elif num_gates == 2:
        # Logic for [Identity, Pauli]
        return [
            (kappa + 1) / (2 * kappa),
            (kappa - 1) / (2 * kappa)
        ]
    else:
        # Default or custom logic if you have more gates
        return [1.0 / num_gates] * num_gates

# ==========================================
# 2. Define Gate Primitives (The "Raw" Data)
# ==========================================
# You still need to define what the gates *are*, but we put them 
# directly into a grid (List of Lists) instead of named variables.

# Helper lambdas to keep definitions short
I = lambda: qml.Identity(wires=DATA_WIRES)
X0 = lambda: qml.PauliX(wires=DATA_WIRES[0])
X1 = lambda: qml.PauliX(wires=DATA_WIRES[1])
Z0 = lambda: qml.PauliZ(wires=DATA_WIRES[0])
Z1 = lambda: qml.PauliZ(wires=DATA_WIRES[1])

# 3x3 Grid of Gate Definitions
RAW_GATES = [
    # Row 1
    [[Z1],    [X1],     [I,X0, Z1],   [Z0]],     
    # Row 2
    [[I, Z1],       [Z1],     [I, X0], [Z1]],    
    # Row 3
    [[Z0],       [Z0],     [X1],   [X1]],  
    # Row 4
    [[X1, Z0],       [Z1],     [I],   [X1]]  
]

# Grid of Kappa values corresponding to the blocks above

# Kappa proved to be successful in 3-agent case
KAPPAS = [
    [2.0, 1.2, 0.5, 1.0],
    [0.3, 0.3, 1.0, 0.5],
    [0.2, 0.5, 2.0, 0.8],
    [0.7, 1.1, 0.6, 1.0]
]


# ==========================================
# 3x3 Grid of State Prep Unitaries (9 distinct operators)
# ==========================================

# --- Row 0 (System 1) ---
def U_sys0_ag0(): qml.Identity(wires=DATA_WIRES)
def U_sys0_ag1(): qml.PauliX(wires=DATA_WIRES[0])
def U_sys0_ag2(): qml.PauliZ(wires=DATA_WIRES[1])
def U_sys0_ag3(): qml.PauliZ(wires=DATA_WIRES[0])

# --- Row 1 (System 2) ---
def U_sys1_ag0(): qml.PauliZ(wires=DATA_WIRES[1])
def U_sys1_ag1(): qml.PauliX(wires=DATA_WIRES[1])
def U_sys1_ag2(): qml.Identity(wires=DATA_WIRES)
def U_sys1_ag3(): qml.PauliX(wires=DATA_WIRES[0])

# --- Row 2 (System 3) ---
def U_sys2_ag0(): qml.PauliX(wires=DATA_WIRES[0])
def U_sys2_ag1(): qml.Identity(wires=DATA_WIRES)
def U_sys2_ag2(): qml.PauliZ(wires=DATA_WIRES[0])
def U_sys2_ag3(): qml.PauliX(wires=DATA_WIRES[1])

# --- Row 3 (System 4) ---
def U_sys3_ag0(): qml.PauliX(wires=DATA_WIRES[1])
def U_sys3_ag1(): qml.PauliZ(wires=DATA_WIRES[0])
def U_sys3_ag2(): qml.Identity(wires=DATA_WIRES)
def U_sys3_ag3(): qml.PauliZ(wires=DATA_WIRES[1])

# The 3x3 Grid (List of Lists)
RAW_B_GATES = [
    [U_sys0_ag0, U_sys0_ag1, U_sys0_ag2, U_sys0_ag3],  # Row 0
    [U_sys1_ag0, U_sys1_ag1, U_sys1_ag2, U_sys1_ag3],  # Row 1
    [U_sys2_ag0, U_sys2_ag1, U_sys2_ag2, U_sys2_ag3],  # Row 2
    [U_sys3_ag0, U_sys3_ag1, U_sys3_ag2, U_sys3_ag3]   # Row 3
]

# ==========================================
# 3. The System Container Class
# ==========================================
class LinearSystemData:
    def __init__(self, gates_grid, kappas, b_gates_grid):
        
        self.n = len(gates_grid)
        self.gates_grid = gates_grid
        self.b_gates = b_gates_grid
        
        # Pre-compute Coefficients and Op Wrappers
        self.coeffs = []
        self.ops = [] # Will hold the callables A_op(l)
        
        for i in range(self.n):
            row_coeffs = []
            row_ops = []
            for j in range(self.n):
                g_list = gates_grid[i][j]
                k = kappas[i][j]
                
                # 1. Calculate C_ij
                c = get_coeffs(k, len(g_list))
                row_coeffs.append(c)
                
                # 2. Create A_ij_op(l) function
                # We use a helper function to bind variables correctly in the loop
                row_ops.append(self._make_wrapper(g_list))
                
            self.coeffs.append(row_coeffs)
            self.ops.append(row_ops)

    def _make_wrapper(self, gate_list):
            def wrapper(l):
                gate_list[int(l)]() 
            return wrapper
# -------------------------------------------------------
# NEW: Matrix Construction Methods
# -------------------------------------------------------
    @staticmethod
    def _as_matrix(gate_fn: Callable):
        """Internal helper: Convert a single gate function to matrix."""
        def qfunc():
            gate_fn()
        # Uses the global DATA_WIRES defined in the module
        return np.array(qml.matrix(qfunc, wire_order=DATA_WIRES)())

    def get_global_matrix(self):
        """
        Constructs the full global matrix A (12x12 for 3 agents).
        Stitches sub-blocks A_ij = Σ c_k * Gate_k
        """
        block_rows = []
        
        for i in range(self.n):
            block_cols = []
            for j in range(self.n):
                # Retrieve gates and coeffs for block A_ij
                gates = self.gates_grid[i][j]
                coeffs = self.coeffs[i][j]
                
                # Calculate linear combination: Σ c * matrix(gate)
                # 1. Get matrices for all gates in this block
                mats = [self._as_matrix(g) for g in gates]
                
                # 2. Weighted sum
                combined_mat = np.zeros_like(mats[0], dtype=complex)
                for c, M in zip(coeffs, mats):
                    combined_mat += c * M
                
                block_cols.append(combined_mat)
            
            block_rows.append(block_cols)
            
        # Stitch into one large matrix
        return np.block(block_rows)

# --- NEW B-Vector Methods (As Requested) ---

    def get_b_vectors(self, sys_id: int):
        """
        Constructs the b vectors for a specific System (Row).
        
        Args:
            sys_id: The row index (0, 1, 2...) representing the system equation.
            
        Returns: 
            (b_total, b_agent0, b_agent1, b_agent2...)
            - b_total: The sum of all local b vectors for this system (target for A*x).
            - b_agentX: The individual vector corresponding to the unitary on agent X.
        """
        # 1. Get the list of gate functions for this System (Row)
        # Structure is b_gates[sys_id][agent_id] -> U function
        # So b_gates[sys_id] returns the list [U_0, U_1, U_2] associated with that row.
        u_gates = self.b_gates[sys_id]
        
        # 2. Get matrices for these gates
        u_mats = [self._as_matrix(u) for u in u_gates]
        
        # 3. Define |00> state
        dim = 2 ** len(DATA_WIRES)
        ket0 = np.zeros(dim, dtype=float)
        ket0[0] = 1.0
        
        # 4. Apply matrices to |00> to get individual state vectors
        b_vecs = [M @ ket0 for M in u_mats]
        
        # 5. Sum them up to get the total b vector for the system equation
        b_total = sum(b_vecs)
        
        return (b_total, *b_vecs)

    def get_global_b_vector(self):
        """
        Concatenates the b_sum vectors of all systems into one global vector.
        Returns shape (Total_Dim, ).
        """
        all_b_sums = []
        for sys_id in range(self.n):
            # Get only the first element (b_total) from the tuple
            b_total = self.get_b_vectors(sys_id)[0]
            all_b_sums.append(b_total)
            
        return np.concatenate(all_b_sums)

    def get_b_op(self, sys_id: int, agent_id: int):
        """
        Returns the specific state prep unitary for a grid position.
        
        Args:
            sys_id: The row index.
            agent_id: The column index (agent).
        """
        return self.b_gates[sys_id][agent_id]

# Initialize the single instance to export
SYSTEM = LinearSystemData(RAW_GATES, KAPPAS, RAW_B_GATES)

if __name__ == "__main__":
    # 1. Get the full 12x12 Matrix A
    A_global = SYSTEM.get_global_matrix()
    print(f"Global Matrix Shape: {A_global.shape}")

    # 2. Get b vectors for System 1 (Agent index 0)
    b_sum, b1, b2, b3 = SYSTEM.get_b_vectors(0)
    print(f"System 1 b_total: {b_sum}")