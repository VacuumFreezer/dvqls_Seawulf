import math
import jax
import jax.numpy as jnp
import numpy as np

# -------------------------------
# Helper Functions
# -------------------------------
def init_angles_jax(key, nq: int, layers: int) -> tuple[jax.Array, jax.Array]:
    """BasicEntanglerLayers params: shape (layers, nq), Uniform(-pi, pi)."""
    key, sub = jax.random.split(key)
    vals = jax.random.uniform(sub, shape=(layers, nq),
                              minval=-math.pi, maxval=math.pi,
                              dtype=jnp.float64)
    return key, vals

def init_norms_jax(key, low: float = 0.0, high: float = 2.0) -> tuple[jax.Array, jax.Array]:
    """Scalar sigma/lambda: Uniform(low, high), returned as 0-d jax array."""
    key, sub = jax.random.split(key)
    val = jax.random.uniform(sub, shape=(), minval=low, maxval=high, dtype=jnp.float64)
    return key, val

# -------------------------------
# Initialize Parameters (N x N Grid)
# -------------------------------

def initialize_global_params_jax(system, n_qubits: int, layers: int, seed: int = 0):
    """
    Creates global params (list-of-lists) with ALL trainable tensors in JAX.
    Returns: global_params, key
    """
    n_agents = system.n
    key = jax.random.PRNGKey(seed)

    global_params = {"alpha": [], "beta": [], "sigma": [], "lambda": [], "b_norm": []}

    for sys_id in range(n_agents):
        # b vectors from your system (likely numpy); treat as constants
        _, *b_individuals = system.get_b_vectors(sys_id)

        row_alpha, row_beta = [], []
        row_sigma, row_lam  = [], []
        row_bnorms          = []

        for agent_id in range(n_agents):
            # trainable
            key, a = init_angles_jax(key, n_qubits, layers)
            key, b = init_angles_jax(key, n_qubits, layers)
            key, s = init_norms_jax(key)
            key, l = init_norms_jax(key)

            row_alpha.append(a)
            row_beta.append(b)
            row_sigma.append(s)
            row_lam.append(l)

            # constant b_norm for this agent (compute in numpy, store as JAX constant)
            b_local = b_individuals[agent_id]
            local_norm_val = float(np.linalg.norm(b_local))
            row_bnorms.append(jax.lax.stop_gradient(jnp.asarray(local_norm_val, dtype=jnp.float64)))

        global_params["alpha"].append(row_alpha)
        global_params["beta"].append(row_beta)
        global_params["sigma"].append(row_sigma)
        global_params["lambda"].append(row_lam)
        global_params["b_norm"].append(row_bnorms)

    return global_params, key