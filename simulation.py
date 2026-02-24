import math
from pennylane import numpy as pnp
import numpy as np
import pennylane as qml
import copy
pnp.random.seed(2)

from static_ops_16agents import SYSTEM

A_global = SYSTEM.get_global_matrix()
print(f"Global Matrix Shape: {A_global.shape}")
print("Global Matrix A =\n", A_global)
print("Condition number of A =", np.linalg.cond(A_global))

global_b = SYSTEM.get_global_b_vector()
print(f"Global_b: {global_b}")

true_sol = np.linalg.solve(A_global, global_b.T)
print("True solution =", true_sol.T) 

from static_ops_16agents import SYSTEM, DATA_WIRES

n_qubits = len(DATA_WIRES) 
LAYERS = 2

# -------------------------------
# Helper Functions
# -------------------------------
def init_angles(nq: int, layers: int = LAYERS) -> pnp.tensor:
    """
    BasicEntanglingLayers parameters: shape (layers, nq).
    Values ~ Uniform(-π, π).
    """
    shape = (layers, nq)
    vals = pnp.zeros(shape=shape, dtype=float)
    return pnp.array(vals, requires_grad=True)

def init_norms(low: float = 0.0, high: float = 0.5) -> pnp.tensor:
    """
    Scalars for σ, λ ~ Uniform(low, high).
    Returns a scalar tensor with gradient enabled.
    """
    # Note: generating size=(1,) and taking [0] makes it a scalar tensor
    # val = pnp.random.uniform(low, high, size=(1,))[0]
    val = 1.0
    return pnp.array(val, requires_grad=True) 

# -------------------------------
# Initialize Parameters (N x N Grid)
# -------------------------------

def initialize_global_params(system):
    """
    Creates a dictionary of 2D arrays (List of Lists) for all parameters.
    Access: params["alpha"][sys_id][agent_id]
    """
    n_agents = system.n
    
    # Dictionaries to hold the grid of parameters
    global_params = {
        "alpha": [],
        "beta": [],
        "sigma": [],
        "lambda": [],
        "b_norm": [] 
    }

    for sys_id in range(n_agents):
        
        # 1. Get B-Vectors for this System
        # We need the individual vectors to calculate local b_norm for each agent
        _, *b_individuals = system.get_b_vectors(sys_id)
        
        row_alpha, row_beta = [], []
        row_sigma, row_lam = [], []
        row_bnorms = []

        # Loop over Agents (Columns)
        for agent_id in range(n_agents):
            
            # Trainable Parameters
            row_alpha.append(init_angles(n_qubits))
            row_beta.append(init_angles(n_qubits))
            row_sigma.append(init_norms())   
            row_lam.append(init_norms())     
            
            # 2. Calculate Static b_norm for THIS SPECIFIC AGENT
            # The agent at column `agent_id` is responsible for the `agent_id`-th term 
            # in the system `sys_id` equation.
            b_local = b_individuals[agent_id]
            
            # Calculate norm (should be 1.0)
            local_norm_val = pnp.linalg.norm(b_local)
            
            # Store as non-trainable tensor
            row_bnorms.append(pnp.array(local_norm_val, requires_grad=False))

        # Append rows to global lists
        global_params["alpha"].append(row_alpha)
        global_params["beta"].append(row_beta)
        global_params["sigma"].append(row_sigma)
        global_params["lambda"].append(row_lam)
        global_params["b_norm"].append(row_bnorms)

    return global_params

ROW_TOPOLOGY = {
    0: [1],
    1: [0,2],
    2: [1,3],
    3: [2]
}

# Execute Initialization
GLOBAL_PARAMS = initialize_global_params(SYSTEM)

print(f"Grid Size (Alpha): {len(GLOBAL_PARAMS['alpha'])} x {len(GLOBAL_PARAMS['alpha'][0])}")
# Example Access
sys_id, agent_id = 0, 1
print(f"\nAccessing Agent ({sys_id}, {agent_id}):")
print(f"C value: {SYSTEM.coeffs[sys_id][agent_id]}")
print(f"Alpha value (shape): {GLOBAL_PARAMS['alpha'][sys_id][agent_id].shape}")
print(f"Sigma value: {GLOBAL_PARAMS['sigma'][sys_id][agent_id]}")
print(f"b_vector: {SYSTEM.get_b_vectors(sys_id)[agent_id+1]}") # +1 to skip total_b
print(f"b_norm (System {sys_id}): {GLOBAL_PARAMS['b_norm'][sys_id][agent_id]}")

# Show connected edge parameters for this agent
print(f"\nConnected Edge Parameters for Agent {agent_id} in System {sys_id}:")

def cal_sol_diff(global_params):
# Reconstruct Quantum Solution
# -------------------------------
    n_agents = SYSTEM.n
    n_qubits = 2 # Get qubit count dynamically

    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def get_local_state_segment(vec_np):
        # Must match the V_var_block used in training
        qml.BasicEntanglerLayers(weights=vec_np, wires=range(n_qubits), rotation=qml.RY)
        return qml.state()

    # We will store the full reconstructed vector from each Row's perspective
    row_estimates = []

    for i in range(n_agents):
        # Each row 'i' constructs a full vector [x_i0, x_i1, x_i2...]
        # based on its own parameters alpha[i][j] and sigma[i][j]
        
        segments = []
        for j in range(n_agents):
            # Retrieve parameters
            alpha = global_params['alpha'][i][j]
            sigma = global_params['sigma'][i][j]
            
            # Generate normalized quantum state |x_j>
            state = get_local_state_segment(alpha)
            
            # Scale by norm sigma to get actual segment x_j
            # Note: We assume state is column-like, usually 1D array in PennyLane
            segments.append(sigma * state)
            
        # Concatenate segments to form the full vector of size (N_agents * 2^n_qubits)
        # axis=0 for 1D arrays acts like simple joining
        full_row_vec = np.concatenate(segments, axis=0)
        row_estimates.append(full_row_vec)

    # 3. Average the estimates (Consensus Solution)
    # ---------------------------------------------
    # Stack rows to shape (N_agents, Total_Dim) and take mean across agents
    stacked_estimates = np.stack(row_estimates)
    recover_sol = np.mean(stacked_estimates, axis=0)

    # 4. Calculate Error
    # ------------------
    # Ensure true_sol is cast to tensor if needed, or convert recover to numpy
    # Here we keep it flexible but ensure shapes match.
    
    # diff = x_true - x_recovered
    diff = true_sol - recover_sol
    
    # Relative Error = ||diff|| / ||true||
    rel_error = np.linalg.norm(diff) / np.linalg.norm(true_sol)
    
    return rel_error

import incomlete_builder as ib

# print("Building Cost Function for Agent 0 (Self) with Neighbor [1]...")
test_builder = ib.make_local_costbuilder(agent_id=2, neighbor_ids=[1,3], show=True)

def flatten_params(global_params, keys=None):
    """
    Modified to accept a specific list of keys.
    If keys=None, defaults to all (for compute_grad).
    """
    if keys is None:
        keys = ["alpha", "beta", "sigma", "lambda"]
        
    flat_list = []
    n_agents = len(global_params["alpha"])
    
    for i in range(n_agents):
        for j in range(n_agents):
            # Only append the keys requested
            if "alpha" in keys:
                flat_list.append(global_params["alpha"][i][j])
            if "beta" in keys:
                flat_list.append(global_params["beta"][i][j])
            if "sigma" in keys:
                flat_list.append(global_params["sigma"][i][j])
            if "lambda" in keys:
                flat_list.append(global_params["lambda"][i][j])
            
    return flat_list

def update_global_from_flat(target_params, flat_list, keys):
    """
    Updates the EXISTING target_params dictionary in-place using values from flat_list.
    This replaces 'rebuild' for the optimization update step.
    """
    n_agents = len(target_params["alpha"])
    idx = 0
    
    for i in range(n_agents):
        for j in range(n_agents):
            if "alpha" in keys:
                target_params["alpha"][i][j] = flat_list[idx]; idx += 1
            if "beta" in keys:
                target_params["beta"][i][j] = flat_list[idx]; idx += 1
            if "sigma" in keys:
                target_params["sigma"][i][j] = flat_list[idx]; idx += 1
            if "lambda" in keys:
                target_params["lambda"][i][j] = flat_list[idx]; idx += 1
    
    return target_params

# Keep your original rebuild function for STEP 2 (Gradient Reconstruction)
# It is still needed because compute_grad returns a full vector.
def rebuild_global_params(flat_list, n_agents, original_bnorms):
    new_params = {
        "alpha": [], "beta": [], "sigma": [], "lambda": [], "b_norm": original_bnorms
    }
    idx = 0
    for i in range(n_agents):
        row_a, row_b, row_s, row_l = [], [], [], []
        for j in range(n_agents):
            row_a.append(flat_list[idx]); idx += 1
            row_b.append(flat_list[idx]); idx += 1
            row_s.append(flat_list[idx]); idx += 1
            row_l.append(flat_list[idx]); idx += 1
        new_params["alpha"].append(row_a)
        new_params["beta"].append(row_b)
        new_params["sigma"].append(row_s)
        new_params["lambda"].append(row_l)
    return new_params

# ==========================
# HELPER FUNCTIONS FOR FLATTENING
# ==========================
def get_flat_group_params(global_params, keys, n_sys):
    """
    Collects all tensors from the specified keys (e.g., 'alpha', 'sigma') 
    into a single flat list.
    """
    flat_list = []
    for k in keys:
        # We assume the structure is always a 2D grid [row][col]
        grid = global_params[k]
        for r in range(n_sys):
            for c in range(n_sys):
                flat_list.append(grid[r][c])
    return flat_list

def set_flat_group_params(global_params, flat_values, keys, n_sys):
    """
    Takes a flat list of new values and distributes them back 
    into the global_params dictionary structure.
    """
    idx = 0
    for k in keys:
        grid = global_params[k]
        for r in range(n_sys):
            for c in range(n_sys):
                # Update the tensor in the grid
                grid[r][c] = flat_values[idx]
                idx += 1
    return global_params

def total_loss_fn(*args):
    """
    Variadic function accepted by the Optimizer.
    args: The flattened list of all trainable parameters.
    """
    # 1. Reconstruct the parameter grid (Edge-Based)
    #    We need GLOBAL_PARAMS['b_norm'] and SORTED_EDGES from the global scope
    current_params = rebuild_global_params(
        args, 
        SYSTEM.n, 
        GLOBAL_PARAMS['b_norm']
    )
    
    total_loss = 0.0
    
    # 2. Iterate over every System (Row)
    for i in range(SYSTEM.n):
        
        # 3. Iterate over every Agent (Column)
        for j in range(SYSTEM.n):
            
            # Retrieve Interaction Neighbors for Agent j from the defined topology
            # (ROW_TOPOLOGY defines the edges within one system)
            # Default to empty list if isolated
            neighbor_ids = ROW_TOPOLOGY[j]
            
            # 4. Build Local Cost Function (Edge-Based)
            #    Creates terms for self + specific edges
            builder = ib.make_local_costbuilder(
                agent_id=j, 
                neighbor_ids=neighbor_ids,
                show=False
            )
            
            # 5. Prepare Parameters for this specific agent
            #    build_params_agent now handles fetching the shared Beta/Lambda 
            #    from the 'beta'/'lambda' dicts using the neighbor_ids
            agent_params = ib.build_params_agent(
                sys_id=i, 
                agent_id=j, 
                row_neighbor_ids=neighbor_ids, 
                global_params=current_params
            )
            
            # 6. Static Injection & Evaluation
            loss = ib.static_builder(agent_params, builder, sys_id=i, agent_id=j)
            
            total_loss = total_loss + loss

    return total_loss

def build_metropolis_matrix(neighbor_map, n=None, make_undirected=True):
    """
    Build Metropolis weight matrix W (n x n) from a neighbor map.

    Metropolis rule (with self-arcs considered):
      |N_i| = strict_degree(i) + 1
      w_ik = 1 / max(|N_i|, |N_k|)      if k in N_i, k != i
      w_ii = 1 - sum_{k != i} w_ik
      w_ik = 0 otherwise
    """
    if n is None:
        n = max(neighbor_map.keys()) + 1 if neighbor_map else 0

    # 1) adjacency sets (keep strictly off-diagonal to avoid double counting in loop)
    adj = [set() for _ in range(n)]
    for i, nbrs in neighbor_map.items():
        for k in nbrs:
            if 0 <= i < n and 0 <= k < n and k != i:
                adj[i].add(k)
                if make_undirected:
                    adj[k].add(i)

    # CHANGE 1: Calculate degree including the self-arc (+1)
    deg = [len(adj[i]) + 1 for i in range(n)]

    # 2) weights
    W = np.zeros((n, n), dtype=float)

    for i in range(n):
        # Isolated node (degree is 1 because of self-loop)
        if len(adj[i]) == 0: 
            W[i, i] = 1.0
            continue

        s = 0.0
        for k in adj[i]:
            # CHANGE 2: Use the inclusive degrees directly
            # The "+1" is effectively already inside deg[i] and deg[k]
            w = 1.0 / (1 + max(deg[i], deg[k]))
            
            W[i, k] = w
            s += w

        # The diagonal is the remainder probability
        W[i, i] = 1.0 - s

    return W

def consensus_mix_metropolis(global_params, W):
    """
    Mix alpha/sigma using row-stochastic W along the 'row/system' index r.
    beta/lambda unchanged.
    """
    n_rows = len(global_params["alpha"])
    n_cols = len(global_params["alpha"][0])

    new_params = {
        "alpha": [[None]*n_cols for _ in range(n_rows)],
        "sigma": [[None]*n_cols for _ in range(n_rows)],
        "beta":   global_params["beta"],
        "lambda": global_params["lambda"],
        "b_norm": global_params["b_norm"],
    }

    for j in range(n_cols):
        for r in range(n_rows):
            idx = np.nonzero(W[r])[0]
            wts = W[r, idx]  # (m,)

            # alpha: (layers, nq)
            alpha_stack = pnp.stack([global_params["alpha"][k][j] for k in idx], axis=0)  # (m, L, nq)
            mixed_alpha = pnp.tensordot(pnp.array(wts), alpha_stack, axes=(0, 0))         # (L, nq)

            # sigma: scalar
            sigma_vec = pnp.array([global_params["sigma"][k][j] for k in idx])            # (m,)
            mixed_sigma = pnp.dot(pnp.array(wts), sigma_vec)                              # scalar

            new_params["alpha"][r][j] = pnp.array(mixed_alpha, requires_grad=True)
            new_params["sigma"][r][j] = pnp.array(mixed_sigma, requires_grad=True)

    return new_params

def update_gradient_tracker_metropolis(current_tracker, current_grads, prev_grads, W):
    """
    Gradient Tracking:
      y(t+1) = W y(t) + (g(t+1) - g(t))

    Only tracks distributed params (alpha, sigma).
    beta/lambda are local -> just use current gradients.
    """
    n_rows = len(current_grads["alpha"])
    n_cols = len(current_grads["alpha"][0])

    new_tracker = {
        "alpha": [[None]*n_cols for _ in range(n_rows)],
        "sigma": [[None]*n_cols for _ in range(n_rows)],
        "beta":  current_grads["beta"],     # local
        "lambda": current_grads["lambda"],  # local
        "b_norm": None
    }

    for j in range(n_cols):
        for r in range(n_rows):
            idx = np.nonzero(W[r])[0]
            wts = W[r, idx]

            # Mix OLD tracker: (W y(t))_r
            y_alpha_stack = pnp.stack([current_tracker["alpha"][k][j] for k in idx], axis=0)
            y_sigma_vec   = pnp.array([current_tracker["sigma"][k][j] for k in idx])

            mix_y_alpha = pnp.tensordot(pnp.array(wts), y_alpha_stack, axes=(0, 0))
            mix_y_sigma = pnp.dot(pnp.array(wts), y_sigma_vec)

            # Gradient difference: g(t+1)-g(t)
            grad_diff_alpha = current_grads["alpha"][r][j] - prev_grads["alpha"][r][j]
            grad_diff_sigma = current_grads["sigma"][r][j] - prev_grads["sigma"][r][j]

            # Update tracker
            new_tracker["alpha"][r][j] = mix_y_alpha + grad_diff_alpha
            new_tracker["sigma"][r][j] = mix_y_sigma + grad_diff_sigma

    return new_tracker

# 1. Group 1: Alpha & Sigma (Adam)
opt_adam = qml.AdamOptimizer(stepsize=0.1)

# 2. Group 2: Beta & Lambda (Gradient Descent)
opt_gd = qml.GradientDescentOptimizer(stepsize=0.1)

COL_TOPOLOGY = {
    0: [1],
    1: [0,2],
    2: [1,3],
    3: [2]
}
Wm = build_metropolis_matrix(COL_TOPOLOGY, n=SYSTEM.n, make_undirected=True)
print("Metropolis Weight Matrix Wm =\n", Wm)

# Histories
loss_history = []
diff_history = []

# ==========================
# PHASE 0: INIT
# ==========================
# We still need flat params for the LOSS FUNCTION (compute_grad), 
# because total_loss_fn expects *args.
flat_params_init = flatten_params(GLOBAL_PARAMS)

# Note: We use opt_adam.compute_grad merely as a tool to get gradients; 
# the optimizer instance used for computing grad doesn't matter for the values themselves.
grads_flat_init, current_cost = opt_adam.compute_grad(
    total_loss_fn,
    args=flat_params_init,
    kwargs={}
)

grad_grid_init = rebuild_global_params(
    grads_flat_init,
    SYSTEM.n,
    GLOBAL_PARAMS["b_norm"]
)

# y(0) = g(0)
tracker_grid = copy.deepcopy(grad_grid_init)
prev_grad_grid = copy.deepcopy(grad_grid_init)

loss_history.append(current_cost)
print(f"[Init] Initial Loss = {current_cost:.5e}")

EPOCHS = 2000
# Set initial stepsize for both
# opt_adam.stepsize = 0.1
# opt_gd.stepsize = 0.1
decay = 0.9999

# ==========================
# PHASE 1: MAIN LOOP
# ==========================
for ep in range(1, EPOCHS + 1):

    # 0) DECAY STEPSIZE
    opt_adam.stepsize = decay * opt_adam.stepsize
    opt_gd.stepsize = decay * opt_gd.stepsize

    # 1) PARAMETER CONSENSUS (Mix alpha/sigma)
    GLOBAL_PARAMS = consensus_mix_metropolis(GLOBAL_PARAMS, Wm)

    # =========================================================
    # 4) APPLY TRACKER AS THE "GRADIENT" (Before compute_grad)
    # =========================================================
    
    # --- GROUP 1: Beta & Lambda (ADAM) ---
    adam_keys = ["beta", "lambda"]
    # Use modified flatten to get ONLY beta and lambda
    flat_tracker_adam = flatten_params(tracker_grid, keys=adam_keys)
    flat_vals_adam    = flatten_params(GLOBAL_PARAMS, keys=adam_keys)
    
    # Apply Optimizer
    new_flat_vals_adam = opt_adam.apply_grad(flat_tracker_adam, flat_vals_adam)
    
    # Update GLOBAL_PARAMS in-place
    update_global_from_flat(GLOBAL_PARAMS, new_flat_vals_adam, keys=adam_keys)


    # --- GROUP 2: Alpha & Sigma (GD) ---
    gd_keys = ["alpha", "sigma"]
    # Use modified flatten to get ONLY alpha and sigma
    flat_tracker_gd = flatten_params(tracker_grid, keys=gd_keys)
    flat_vals_gd    = flatten_params(GLOBAL_PARAMS, keys=gd_keys)
    
    # Apply Optimizer
    new_flat_vals_gd = opt_gd.apply_grad(flat_tracker_gd, flat_vals_gd)
    
    # Update GLOBAL_PARAMS in-place
    update_global_from_flat(GLOBAL_PARAMS, new_flat_vals_gd, keys=gd_keys)

    
    # =========================================================
    # 2) COMPUTE CURRENT GRADS g(t+1)
    # =========================================================
    # Flatten EVERYTHING (keys=None) because the loss function needs all params
    # This maintains the interleaved order your loss function expects
    new_flat_params = flatten_params(GLOBAL_PARAMS, keys=None)

    # We use opt_adam.compute_grad just to get gradients (the optimizer instance doesn't matter here)
    grads_flat, current_cost = opt_adam.compute_grad(
        total_loss_fn,
        args=new_flat_params,
        kwargs={}
    )

    # Rebuild the gradient grid (Requires FULL rebuild)
    current_grad_grid = rebuild_global_params(
        grads_flat,
        SYSTEM.n,
        GLOBAL_PARAMS["b_norm"]
    )

    # =========================================================
    # 3) TRACKER UPDATE y(t+1) = W y(t) + (g(t+1) - g(t))
    # =========================================================
    tracker_grid = update_gradient_tracker_metropolis(
        current_tracker=tracker_grid,
        current_grads=current_grad_grid,
        prev_grads=prev_grad_grid,
        W=Wm
    )

    # Update history for next step
    prev_grad_grid = copy.deepcopy(current_grad_grid)

    # 6) LOGGING
    loss_history.append(current_cost)
    sol_diff = cal_sol_diff(GLOBAL_PARAMS)
    diff_history.append(sol_diff)

    print(
        f"[Epoch {ep:04d}] Total Loss = {current_cost:.5e} | Sol Diff = {sol_diff:.5e}"
    )

    import matplotlib.pyplot as plt
xs = list(range(1, len(diff_history) + 1))

plt.figure()
plt.plot(xs, diff_history, label="Difference to True Solution")
plt.xlabel("Epoch"); plt.ylabel("L2 error"); plt.yscale('log')
plt.title("Convergence of 4-agent Line Network with Tracker \n $\kappa$=196")
plt.grid(True); plt.legend(); plt.show()

loss_xs = list(range(0, len(loss_history)))
plt.figure()
plt.plot(loss_xs, loss_history, label="Total Loss")
plt.xlabel("Epoch"); plt.ylabel("Cost"); plt.yscale('log')
plt.grid(True); plt.legend(); plt.show()

def recover_and_verify_solution(global_params):
    """
    Reconstructs the quantum states and verifies the solution against the true b vectors.
    """
    n_agents = SYSTEM.n
    n_qubits = len(DATA_WIRES)
    dim = 2**n_qubits
    
    # 1. Define the State Preview Node
    #    (Must match the V_var_block used in training)
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def get_state(weights):
        qml.BasicEntanglerLayers(weights=weights, wires=range(n_qubits), rotation=qml.RY)
        return qml.state()

    # 2. Get the Global Matrix for block extraction
    #    A_global is composed of N x N blocks of size (dim x dim)
    A_global = SYSTEM.get_global_matrix()

    print("\n" + "="*60)
    print(f"      SOLUTION RECOVERY & VERIFICATION (N={n_agents})")
    print("="*60)

    # Iterate over each System (Row)
    for sys_id in range(n_agents):
        print(f"\n>>> SYSTEM {sys_id} (Row {sys_id}) <<<")
        
        # Retrieve True b vectors
        # b_total is the target for the sum. b_individuals are the local components.
        true_b_sum, *individual_b = SYSTEM.get_b_vectors(sys_id)
        
        # Accumulator for the row sum: Sum_j (sigma * A_ij * x_j)
        row_recovered_Ax = np.zeros(dim, dtype=complex)
        
        # Iterate over each Agent (Column)
        for agent_id in range(n_agents):
            # A. Retrieve Optimized Parameters
            alpha = global_params['alpha'][sys_id][agent_id]
            # beta  = global_params['beta'][sys_id][agent_id]
            sigma = global_params['sigma'][sys_id][agent_id]
            # lam   = global_params['lambda'][sys_id][agent_id]
            
            # B. Reconstruct Quantum States
            state_x = get_state(alpha)
            # state_z = get_state(beta)
            
            # C. Extract specific Matrix A_ij from Global Matrix
            #    Slice range: [row_start:row_end, col_start:col_end]
            r_start, r_end = sys_id * dim, (sys_id + 1) * dim
            c_start, c_end = agent_id * dim, (agent_id + 1) * dim
            A_ij_matrix = A_global[r_start:r_end, c_start:c_end]
            # D. Calculate Terms
            #    term_Ax = σ * A * |x>
            #    term_z  = λ * |z>
            term_Ax = sigma * (A_ij_matrix @ state_x)
            # term_z  = lam * state_z
            
            # E. Local "b" reconstruction (Ansatz Component Check)
            #    Corresponds to your 'recover_b11' logic
            #    Note: The sign depends on topology (-degree vs +1), 
            #    but we show the raw difference here for inspection.
            # rec_b_local = term_Ax - term_z
            
            # Accumulate for Global Check
            row_recovered_Ax += term_Ax
            
            # --- Print Local Details ---
            print(f"  [Agent {agent_id}]")
            print(f"alpha: {alpha}")
            # print(f"    sigma: {sigma:.4f}, lambda: {lam:.4f}")
            print(f"    |x>: {sigma*np.round(state_x, 3)}")
            # print(f"    rec_b_local (σAx - λz): {np.round(rec_b_local, 3)}")
            # print(f"    true_b_local          : {np.round(true_b_individuals[agent_id], 3)}")

        # --- Global Row Verification ---
        print("-" * 40)
        print(f"  Recovered b_total (Sum σAx): {np.round(row_recovered_Ax, 4)}")
        print(f"  True b_total               : {np.round(true_b_sum, 4)}")
        
        # Calculate Error
        diff = np.linalg.norm(row_recovered_Ax - true_b_sum)
        rel_err = diff / np.linalg.norm(true_b_sum)
        print(f"  > L2 Difference: {diff:.5e}")
        print(f"  > Relative Err : {rel_err:.2%}")

recover_and_verify_solution(GLOBAL_PARAMS)

def analyze_consensus_variance(global_params):
    """
    Calculates the variance of the reconstructed state for each agent across all systems.
    Ideal Variance = 0 (Perfect Consensus).
    """
    n_agents = SYSTEM.n
    n_qubits = len(DATA_WIRES)
    
    # 1. Setup Quantum Device
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def get_state(weights):
        qml.BasicEntanglerLayers(weights=weights, wires=range(n_qubits), rotation=qml.RY)
        return qml.state()

    print("\n" + "="*60)
    print(f"      CONSENSUS VARIANCE ANALYSIS (N={n_agents})")
    print("="*60)
    
    global_avg_variance = 0.0

    # Iterate over AGENTS (Columns) first, to compare across Systems (Rows)
    for agent_id in range(n_agents):
        print(f"\n>>> Agent {agent_id} Consensus <<<")
        
        # Collect vectors from all rows for this agent
        reconstructed_vectors = []
        
        for sys_id in range(n_agents):
            # Retrieve parameters for Agent j stored in System i
            alpha = global_params['alpha'][sys_id][agent_id]
            sigma = global_params['sigma'][sys_id][agent_id]
            
            # Reconstruct scaled state v_ij = sigma * |x>
            # Use np.array to ensure we work with values, not tensors
            state = np.array(get_state(alpha))
            v_ij = float(sigma) * state
            reconstructed_vectors.append(v_ij)

        # Convert to stack for vectorized math
        vec_stack = np.stack(reconstructed_vectors)
        
        # 1. Calculate Mean Vector (Centroid)
        mean_vec = np.mean(vec_stack, axis=0)
        
        # 2. Calculate Squared Euclidean Distances from Centroid
        # || v_i - mean ||^2
        diffs = vec_stack - mean_vec
        sq_dists = np.sum(np.abs(diffs)**2, axis=1)
        
        # 3. Variance = Mean of Squared Distances
        variance = np.mean(sq_dists)
        global_avg_variance += variance
        
        # Print Stats
        print(f"  Vectors collected: {n_agents}")
        print(f"  Mean Vector Norm : {np.linalg.norm(mean_vec):.5f}")
        print(f"  Variance (Error) : {variance:.5e}")
        
        if variance < 1e-5:
            print("  [STATUS] -> Perfect Consensus")
        else:
            print("  [STATUS] -> Disagreement Detected")

    print("-" * 40)
    print(f"Average System-Wide Variance: {global_avg_variance / n_agents:.5e}")

analyze_consensus_variance(GLOBAL_PARAMS)