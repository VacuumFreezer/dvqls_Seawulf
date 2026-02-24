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