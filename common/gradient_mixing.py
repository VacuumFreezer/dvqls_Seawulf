import numpy as np
from pennylane import numpy as pnp

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


def consensus_mix_gradients_metropolis(current_grads, W):
    """
    Gradient Consensus Mixing:
      g_mixed = W * g_local

    Only mixes distributed params (alpha, sigma).
    beta/lambda are local -> keep original local gradients.
    """
    n_rows = len(current_grads["alpha"])
    n_cols = len(current_grads["alpha"][0])

    # 初始化新的梯度字典，beta/lambda 直接复制本地梯度
    mixed_grads = {
        "alpha": [[None]*n_cols for _ in range(n_rows)],
        "sigma": [[None]*n_cols for _ in range(n_rows)],
        "beta":  current_grads["beta"],    # local, no mixing
        "lambda": current_grads["lambda"], # local, no mixing
        "b_norm": None
    }

    for j in range(n_cols):
        for r in range(n_rows):
            idx = np.nonzero(W[r])[0]
            wts = W[r, idx]

            # --- Consensus Step for Gradients ---
            grad_alpha_stack = pnp.stack([current_grads["alpha"][k][j] for k in idx], axis=0)
            grad_sigma_vec   = pnp.array([current_grads["sigma"][k][j] for k in idx])

            # weighted average of gradients
            mix_grad_alpha = pnp.tensordot(pnp.array(wts), grad_alpha_stack, axes=(0, 0))
            mix_grad_sigma = pnp.dot(pnp.array(wts), grad_sigma_vec)

            # 存入混合后的梯度
            mixed_grads["alpha"][r][j] = mix_grad_alpha
            mixed_grads["sigma"][r][j] = mix_grad_sigma

    return mixed_grads


def init_tracker_from_grad(grad_grid):
    """
    Initialize the tracker from the gradient grid.
    """
    return {
        "alpha": [[pnp.array(grad_grid["alpha"][r][c]) for c in range(len(grad_grid["alpha"][0]))]
                 for r in range(len(grad_grid["alpha"]))],
        "sigma": [[pnp.array(grad_grid["sigma"][r][c]) for c in range(len(grad_grid["sigma"][0]))]
                 for r in range(len(grad_grid["sigma"]))],
        "beta": grad_grid["beta"],
        "lambda": grad_grid["lambda"],
        "b_norm": None
    }