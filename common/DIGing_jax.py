import numpy as np
import jax
import jax.numpy as jnp

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


def consensus_mix_metropolis_jax(global_params, W_np: np.ndarray):
    """
    Mix alpha/sigma using row-stochastic W along the row/system index r.
    beta/lambda unchanged.
    All math uses jnp; W is numpy constant.
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
            idx = np.nonzero(W_np[r])[0]      # numpy indices
            wts = jnp.asarray(W_np[r, idx], dtype=jnp.float64)  # (m,)

            # alpha: (layers, nq)
            alpha_stack = jnp.stack([global_params["alpha"][k][j] for k in idx], axis=0)  # (m, L, nq)
            mixed_alpha = jnp.tensordot(wts, alpha_stack, axes=(0, 0))                   # (L, nq)

            # sigma: scalar
            sigma_vec = jnp.stack([global_params["sigma"][k][j] for k in idx], axis=0)   # (m,)
            mixed_sigma = jnp.dot(wts, sigma_vec)                                        # scalar

            new_params["alpha"][r][j] = mixed_alpha
            new_params["sigma"][r][j] = mixed_sigma

    return new_params



def update_gradient_tracker_metropolis_jax(current_tracker, current_grads, prev_grads, W_np: np.ndarray):
    """
    y(t+1) = W y(t) + (g(t+1) - g(t))
    tracks alpha/sigma only; beta/lambda are local (pass-through).
    """
    n_rows = len(current_grads["alpha"])
    n_cols = len(current_grads["alpha"][0])

    new_tracker = {
        "alpha": [[None]*n_cols for _ in range(n_rows)],
        "sigma": [[None]*n_cols for _ in range(n_rows)],
        "beta":  current_grads["beta"],
        "lambda": current_grads["lambda"],
        "b_norm": None
    }

    for j in range(n_cols):
        for r in range(n_rows):
            idx = np.nonzero(W_np[r])[0]
            wts = jnp.asarray(W_np[r, idx], dtype=jnp.float64)

            y_alpha_stack = jnp.stack([current_tracker["alpha"][k][j] for k in idx], axis=0)
            y_sigma_vec   = jnp.stack([current_tracker["sigma"][k][j] for k in idx], axis=0)

            mix_y_alpha = jnp.tensordot(wts, y_alpha_stack, axes=(0, 0))
            mix_y_sigma = jnp.dot(wts, y_sigma_vec)

            grad_diff_alpha = current_grads["alpha"][r][j] - prev_grads["alpha"][r][j]
            grad_diff_sigma = current_grads["sigma"][r][j] - prev_grads["sigma"][r][j]

            new_tracker["alpha"][r][j] = mix_y_alpha + grad_diff_alpha
            new_tracker["sigma"][r][j] = mix_y_sigma + grad_diff_sigma

    return new_tracker


def init_tracker_from_grad(grad_grid):
    """
    Initialize the tracker from the gradient grid.
    """
    return {
        "alpha": [[jnp.array(grad_grid["alpha"][r][c]) for c in range(len(grad_grid["alpha"][0]))]
                 for r in range(len(grad_grid["alpha"]))],
        "sigma": [[jnp.array(grad_grid["sigma"][r][c]) for c in range(len(grad_grid["sigma"][0]))]
                 for r in range(len(grad_grid["sigma"]))],
        "beta": grad_grid["beta"],
        "lambda": grad_grid["lambda"],
        "b_norm": None
    }