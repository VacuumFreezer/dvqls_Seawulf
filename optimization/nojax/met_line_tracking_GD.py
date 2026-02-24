# simulation_run_tracking_adamZ_only.py
import argparse, time, copy
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import argparse, importlib

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]   
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.params_init import initialize_global_params
from common.params_io import flatten_params, rebuild_global_params, update_global_from_flat
from common.DIGing import build_metropolis_matrix, consensus_mix_metropolis, init_tracker_from_grad, update_gradient_tracker_metropolis
from common.reporting import make_run_dir, setup_logger, JsonlWriter

import objective.incomplete_builder as ib

def load_static_ops(module_name: str):

    return importlib.import_module(module_name)

# ==========================================

def cal_sol_diff(global_params, true_sol, n_qubits):
    """
    Calculate true_sol vs recovered solution difference.
    """
    n_agents = GLOBAL_AGENTS_N
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def get_local_state_segment(weights):
        qml.BasicEntanglerLayers(weights=weights, wires=range(n_qubits), rotation=qml.RY)
        return qml.state()

    row_estimates = []
    for i in range(n_agents):
        segs = []
        for j in range(n_agents):
            alpha = global_params["alpha"][i][j]
            sigma = global_params["sigma"][i][j]
            state = get_local_state_segment(alpha)
            segs.append(sigma * state)
        full_row_vec = np.concatenate(segs, axis=0)
        row_estimates.append(full_row_vec)

    recover_sol = np.mean(np.stack(row_estimates), axis=0)
    diff = true_sol - recover_sol
    return np.linalg.norm(diff) / np.linalg.norm(true_sol)

# ==========================================

def build_total_loss_fn(global_params_bnorm, row_topology):
    """
    closure to build total loss function
    """
    def total_loss_fn(*args):
        current_params = rebuild_global_params(args, GLOBAL_AGENTS_N, global_params_bnorm)
        total_loss = 0.0
        for i in range(GLOBAL_AGENTS_N):
            for j in range(GLOBAL_AGENTS_N):
                neighbor_ids = row_topology[j]
                builder = ib.make_local_costbuilder(agent_id=j, neighbor_ids=neighbor_ids, show=False)
                agent_params = ib.build_params_agent(
                    sys_id=i, agent_id=j, row_neighbor_ids=neighbor_ids, global_params=current_params
                )
                total_loss = total_loss + ib.static_builder(agent_params, builder, sys_id=i, agent_id=j)
        return total_loss
    return total_loss_fn

# ==========================================

def recover_and_verify_solution(global_params, SYSTEM, DATA_WIRES, out=None):
    """
    Reconstructs the quantum states and verifies the solution against the true b vectors.
    Writes to `out` (file-like), default stdout.
    """
    if out is None:
        out = sys.stdout

    n_agents = SYSTEM.n
    n_qubits = len(DATA_WIRES)
    dim = 2**n_qubits

    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def get_state(weights):
        qml.BasicEntanglerLayers(weights=weights, wires=range(n_qubits), rotation=qml.RY)
        return qml.state()

    A_global = SYSTEM.get_global_matrix()

    print("\n" + "="*60, file=out)
    print(f"      SOLUTION RECOVERY & VERIFICATION (N={n_agents})", file=out)
    print("="*60, file=out)

    for sys_id in range(n_agents):
        print(f"\n>>> SYSTEM {sys_id} (Row {sys_id}) <<<", file=out)

        true_b_sum, *individual_b = SYSTEM.get_b_vectors(sys_id)

        row_recovered_Ax = np.zeros(dim, dtype=complex)

        for agent_id in range(n_agents):
            alpha = global_params['alpha'][sys_id][agent_id]
            sigma = global_params['sigma'][sys_id][agent_id]

            state_x = np.array(get_state(alpha))  # detach to numpy

            r_start, r_end = sys_id * dim, (sys_id + 1) * dim
            c_start, c_end = agent_id * dim, (agent_id + 1) * dim
            A_ij_matrix = A_global[r_start:r_end, c_start:c_end]

            term_Ax = float(sigma) * (A_ij_matrix @ state_x)
            row_recovered_Ax += term_Ax

            print(f"  [Agent {agent_id}]", file=out)
            print(f"alpha: {alpha}", file=out)
            print(f"    |x>: {np.round(float(sigma)*state_x, 3)}", file=out)

        print("-" * 40, file=out)
        print(f"  Recovered b_total (Sum σAx): {np.round(row_recovered_Ax, 4)}", file=out)
        print(f"  True b_total               : {np.round(true_b_sum, 4)}", file=out)

        diff = np.linalg.norm(row_recovered_Ax - true_b_sum)
        rel_err = diff / np.linalg.norm(true_b_sum)
        print(f"  > L2 Difference: {diff:.5e}", file=out)
        print(f"  > Relative Err : {rel_err:.2%}", file=out)

# ==========================================

def analyze_consensus_variance(global_params, SYSTEM, DATA_WIRES, out=None):
    """
    Calculates the variance of the reconstructed state for each agent across all systems.
    Ideal Variance = 0 (Perfect Consensus).
    Writes to `out` (file-like), default stdout.
    """
    if out is None:
        out = sys.stdout

    n_agents = SYSTEM.n
    n_qubits = len(DATA_WIRES)

    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def get_state(weights):
        qml.BasicEntanglerLayers(weights=weights, wires=range(n_qubits), rotation=qml.RY)
        return qml.state()

    print("\n" + "="*60, file=out)
    print(f"      CONSENSUS VARIANCE ANALYSIS (N={n_agents})", file=out)
    print("="*60, file=out)

    global_avg_variance = 0.0

    for agent_id in range(n_agents):
        print(f"\n>>> Agent {agent_id} Consensus <<<", file=out)

        reconstructed_vectors = []
        for sys_id in range(n_agents):
            alpha = global_params['alpha'][sys_id][agent_id]
            sigma = global_params['sigma'][sys_id][agent_id]

            state = np.array(get_state(alpha))
            v_ij = float(sigma) * state
            reconstructed_vectors.append(v_ij)

        vec_stack = np.stack(reconstructed_vectors)
        mean_vec = np.mean(vec_stack, axis=0)

        diffs = vec_stack - mean_vec
        sq_dists = np.sum(np.abs(diffs)**2, axis=1)
        variance = np.mean(sq_dists)

        global_avg_variance += variance

        print(f"  Vectors collected: {n_agents}", file=out)
        print(f"  Mean Vector Norm : {np.linalg.norm(mean_vec):.5f}", file=out)
        print(f"  Variance (Error) : {variance:.5e}", file=out)

        if variance < 1e-5:
            print("  [STATUS] -> Perfect Consensus", file=out)
        else:
            print("  [STATUS] -> Disagreement Detected", file=out)

    print("-" * 40, file=out)
    print(f"Average System-Wide Variance: {global_avg_variance / n_agents:.5e}", file=out)

# ==========================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static_ops", required=True,
                    help="e.g. problems.static_ops_equation1_kappa196")
    ap.add_argument("--out", required=True)

    ap.add_argument("--system_id", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr_g", type=float, default=0.1)
    ap.add_argument("--lr_a", type=float, default=0.1)
    ap.add_argument("--decay", type=float, default=0.9999)
    ap.add_argument("--log_every", type=int, default=10)
    args = ap.parse_args()

    ops = load_static_ops(args.static_ops)

    SYSTEM = ops.SYSTEM
    DATA_WIRES = ops.DATA_WIRES

    global GLOBAL_AGENTS_N
    GLOBAL_AGENTS_N = SYSTEM.n
    
    # ---- inject static ops ----
    ib.bind_static_ops(ops)

    # ---- seeds ----
    np.random.seed(args.seed)
    pnp.random.seed(args.seed)

    paths = make_run_dir(args.out)
    logger = setup_logger(paths.report_txt)
    metrics = JsonlWriter(paths.metrics_jsonl)

    # ---- print the same “report header” as notebook ----
    A_global = SYSTEM.get_global_matrix()
    global_b = SYSTEM.get_global_b_vector()
    true_sol = np.linalg.solve(A_global, global_b.T)

    logger.info(f"Global Matrix Shape: {A_global.shape}")
    logger.info("Global Matrix A =\n" + str(A_global))
    logger.info("Condition number of A = " + str(np.linalg.cond(A_global)))
    logger.info(f"Global_b: {global_b}")
    logger.info("True solution = " + str(true_sol.T))

    n_qubits = len(DATA_WIRES)
    LAYERS = 2

    # ---- init params ----
    GLOBAL_PARAMS = initialize_global_params(SYSTEM, n_qubits=n_qubits, layers=LAYERS)

    # topology: line graph
    ROW_TOPOLOGY = {0:[1], 1:[0,2], 2:[1,3], 3:[2]}
    COL_TOPOLOGY = {0:[1], 1:[0,2], 2:[1,3], 3:[2]}

    # metropolis W
    Wm = build_metropolis_matrix(COL_TOPOLOGY, n=SYSTEM.n, make_undirected=True)
    logger.info("Metropolis Weight Matrix Wm =\n" + str(Wm))

    total_loss_fn = build_total_loss_fn(GLOBAL_PARAMS["b_norm"], ROW_TOPOLOGY)

    # optimizers (case: Adam for z only, GD for x)
    opt_adam = qml.GradientDescentOptimizer(stepsize=args.lr_a)      # alpha/sigma
    opt_gd   = qml.GradientDescentOptimizer(stepsize=args.lr_g)      # beta/lambda

    loss_history, diff_history = [], []

    # ==========================================    
    # Start optimization
    # ==========================================

    # ---- Phase 0: init tracker y(0)=g(0) ----
    flat_params_init = flatten_params(GLOBAL_PARAMS, keys=None)
    grads_flat_init, current_cost = opt_adam.compute_grad(total_loss_fn, 
                                                          args=flat_params_init, 
                                                          kwargs={})
    
    grad_grid_init = rebuild_global_params(grads_flat_init, 
                                           SYSTEM.n, 
                                           GLOBAL_PARAMS["b_norm"])

    tracker_grid = init_tracker_from_grad(grad_grid_init)
    prev_grad_grid = copy.deepcopy(grad_grid_init)

    loss_history.append(float(current_cost))
    logger.info(f"[Init] Initial Loss = {current_cost:.5e}")

    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        # decay
        opt_adam.stepsize *= args.decay
        opt_gd.stepsize   *= args.decay

        # consensus for alpha/sigma
        GLOBAL_PARAMS = consensus_mix_metropolis(GLOBAL_PARAMS, Wm)

        # apply “tracker as gradient”
        # Group Z: alpha/sigma -> Adam
        adam_keys = ["alpha", "sigma"]
        flat_tracker_adam = flatten_params(tracker_grid, keys=adam_keys)
        flat_vals_adam    = flatten_params(GLOBAL_PARAMS, keys=adam_keys)
        new_flat_vals_adam = opt_adam.apply_grad(flat_tracker_adam, flat_vals_adam)
        update_global_from_flat(GLOBAL_PARAMS, new_flat_vals_adam, keys=adam_keys)

        # Group X: beta/lambda -> GD
        gd_keys = ["beta", "lambda"]
        flat_tracker_gd = flatten_params(tracker_grid, keys=gd_keys)
        flat_vals_gd    = flatten_params(GLOBAL_PARAMS, keys=gd_keys)
        new_flat_vals_gd = opt_gd.apply_grad(flat_tracker_gd, flat_vals_gd)
        update_global_from_flat(GLOBAL_PARAMS, new_flat_vals_gd, keys=gd_keys)

        # compute current grads
        new_flat_params = flatten_params(GLOBAL_PARAMS, keys=None)
        grads_flat, current_cost = opt_adam.compute_grad(total_loss_fn, 
                                                         args=new_flat_params, 
                                                         kwargs={})
        
        current_grad_grid = rebuild_global_params(grads_flat, 
                                                  SYSTEM.n, 
                                                  GLOBAL_PARAMS["b_norm"])

        # tracker update
        tracker_grid = update_gradient_tracker_metropolis(
            current_tracker=tracker_grid,
            current_grads=current_grad_grid,
            prev_grads=prev_grad_grid,
            W=Wm
        )
        prev_grad_grid = copy.deepcopy(current_grad_grid)

        # metrics
        loss_history.append(float(current_cost))
        sol_diff = cal_sol_diff(GLOBAL_PARAMS, true_sol=true_sol, n_qubits=n_qubits)
        diff_history.append(float(sol_diff))

        if (ep % args.log_every) == 0 or ep == 1:
            metrics.write({
            "epoch": ep,
            "loss": float(current_cost),
            "sol_diff": float(sol_diff),
            "lr_g": float(opt_gd.stepsize),
            "lr_a": float(opt_adam.stepsize),
            "wall_s": time.time() - t0,
            })
            
            logger.info(f"[Epoch {ep:04d}] Total Loss = {current_cost:.5e} | Sol Diff = {sol_diff:.5e}")

    metrics.close()

    # ---- save figures (no plt.show) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- build title string ----
    # kappa from condition number (rounded to int)
    kappa_int = int(np.rint(np.linalg.cond(A_global)))

    # optimizer tag: ideally define opt_tag near optimizer construction
    # e.g. opt_tag = "tracking + AdamA only (g:GD)"  (whatever you want shown)
    try:
        opt_tag
    except NameError:
        from pathlib import Path
        opt_tag = Path(__file__).stem  # fallback: script name

    title = f"kappa≈{kappa_int} | lr_g0={args.lr_g:g}, lr_a0={args.lr_a:g} | {opt_tag}"

    # ---- diff figure ----
    xs = np.arange(1, len(diff_history) + 1)
    plt.figure()
    plt.plot(xs, diff_history)
    plt.xlabel("Epoch")
    plt.ylabel("L2 error")
    plt.yscale("log")
    plt.grid(True)
    plt.title(title)
    plt.savefig(paths.fig_diff, dpi=200, bbox_inches="tight")
    plt.close()

    # ---- loss figure ----
    xs2 = np.arange(0, len(loss_history))
    plt.figure()
    plt.plot(xs2, loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.yscale("log")
    plt.grid(True)
    plt.title(title)
    plt.savefig(paths.fig_loss, dpi=200, bbox_inches="tight")
    plt.close()

    # ---- save artifacts ----
    np.savez(paths.artifacts_npz, loss=np.array(loss_history), sol_diff=np.array(diff_history))

    # ---- extra analysis / verification ----
    analysis_path = paths.run_dir / "analysis.txt"
    with open(analysis_path, "w", encoding="utf-8") as f:
        recover_and_verify_solution(GLOBAL_PARAMS, SYSTEM, DATA_WIRES, out=f)
        analyze_consensus_variance(GLOBAL_PARAMS, SYSTEM, DATA_WIRES, out=f)

    logger.info(f"Post-analysis written to: {analysis_path}")

    logger.info("Finished. Outputs written to: " + str(paths.run_dir))

if __name__ == "__main__":
    main()
