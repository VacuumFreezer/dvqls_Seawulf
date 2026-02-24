from pennylane import numpy as pnp
import math

def init_angles(nq: int, layers: int) -> pnp.tensor:
    """
    BasicEntanglingLayers parameters: shape (layers, nq).
    Values ~ Uniform(-π, π).
    """
    shape = (layers, nq)
    # vals = pnp.zeros(shape=shape, dtype=float)
    vals = pnp.random.uniform(-math.pi, math.pi, size=shape)
    return pnp.array(vals, requires_grad=True)

def init_norms(low: float = 0.0, high: float = 2.0) -> pnp.tensor:
    """
    Scalars for σ, λ ~ Uniform(low, high).
    Returns a scalar tensor with gradient enabled.
    """
    # Note: generating size=(1,) and taking [0] makes it a scalar tensor
    val = pnp.random.uniform(low, high, size=(1,))[0]
    # val = 1.0
    return pnp.array(val, requires_grad=True) 

# -------------------------------
# Initialize Parameters (N x N Grid)
# -------------------------------

def initialize_global_params(system, n_qubits: int, layers: int,) -> dict:
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
            row_alpha.append(init_angles(n_qubits, layers))
            row_beta.append(init_angles(n_qubits, layers))
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