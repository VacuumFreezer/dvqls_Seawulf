# simulation_run_tracking_adamZ_only.py
import argparse, time, copy
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import importlib
import jax
import jax.numpy as jnp
import optax
import catalyst

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]   
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.params_init_10qubits import initialize_global_params_jax
from common.params_io import flatten_params, rebuild_global_params, update_global_from_flat
from common.DIGing_jax import build_metropolis_matrix, consensus_mix_metropolis_jax, init_tracker_from_grad, update_gradient_tracker_metropolis_jax
from common.reporting import make_run_dir, setup_logger, JsonlWriter

import objective.builder_cat_nodispatch as ib


TOPOLOGY_LINE_2 = {0: [1], 1: [0]}


def load_static_ops(module_name: str):

    return importlib.import_module(module_name)


def load_2x2_system_and_wires(ops_module):
    if not hasattr(ops_module, "SYSTEMS"):
        raise RuntimeError(
            f"{ops_module.__name__} does not expose SYSTEMS; cannot pick 2x2 variant."
        )
    systems = getattr(ops_module, "SYSTEMS")
    if "2x2" not in systems:
        raise RuntimeError(
            f"{ops_module.__name__} does not contain SYSTEMS['2x2']."
        )
    system = systems["2x2"]

    if hasattr(ops_module, "DATA_WIRES_BY_SYSTEM"):
        wires_map = getattr(ops_module, "DATA_WIRES_BY_SYSTEM")
        if "2x2" in wires_map:
            data_wires = list(wires_map["2x2"])
        else:
            data_wires = list(getattr(ops_module, "DATA_WIRES"))
    elif hasattr(system, "data_wires"):
        data_wires = list(system.data_wires)
    else:
        data_wires = list(getattr(ops_module, "DATA_WIRES"))

    return system, data_wires

# ==========================================
def to_jax_flat(flat_list):
    # flat_list: list of scalars / vectors / arrays
    return [jnp.asarray(x) for x in flat_list]

# ==========================================

def recover_global_solution(global_params, n_qubits):
    """
    Reconstruct global x by averaging all system-wise recovered rows.
    """
    n_agents = GLOBAL_AGENTS_N
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="jax")
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

    return np.mean(np.stack(row_estimates), axis=0)


def cal_sol_diff(global_params, true_sol, n_qubits, recover_sol=None):
    """
    Calculate true_sol vs recovered solution difference.
    """
    if recover_sol is None:
        recover_sol = recover_global_solution(global_params, n_qubits)

    true_sol_vec = np.asarray(true_sol).reshape(-1)
    diff = true_sol_vec - recover_sol
    return np.linalg.norm(diff) / np.linalg.norm(true_sol_vec)


def cal_ax_minus_b_norm(global_params, A_global, global_b, n_qubits, recover_sol=None):
    """
    Calculate ||Ax - b||_2 from the recovered global solution x.
    """
    if recover_sol is None:
        recover_sol = recover_global_solution(global_params, n_qubits)

    b_vec = np.asarray(global_b).reshape(-1)
    residual = A_global @ recover_sol - b_vec
    return np.linalg.norm(residual)

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

    @qml.qnode(dev, interface="jax")
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

    @qml.qnode(dev, interface="jax")
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
    ap.add_argument("--lr", type=float, default=0.01)
    # ap.add_argument("--lr_a", type=float, default=0.1)
    ap.add_argument("--decay", type=float, default=0.9999)
    ap.add_argument("--log_every", type=int, default=10)
    args = ap.parse_args()

    # ---- Dynamic import of static ops ----
    ops = load_static_ops(args.static_ops)

    SYSTEM, DATA_WIRES = load_2x2_system_and_wires(ops)

    global GLOBAL_AGENTS_N
    GLOBAL_AGENTS_N = SYSTEM.n
    if GLOBAL_AGENTS_N != 2:
        raise RuntimeError(
            f"2x2 script expects 2 agents from SYSTEMS['2x2'], got {GLOBAL_AGENTS_N}"
        )
    
    # # ---- inject static ops ----
    # ib.bind_static_ops(ops)

    # ---- seeds ----
    pnp.random.seed(args.seed)
    np.random.seed(args.seed)
    jax.config.update("jax_enable_x64", True) 

    paths = make_run_dir(args.out)
    logger = setup_logger(paths.report_txt)
    metrics = JsonlWriter(paths.metrics_jsonl)

    # ---- print the same “report header” as notebook ----
    A_global = SYSTEM.get_global_matrix()
    global_b = SYSTEM.get_global_b_vector()
    true_sol = np.linalg.solve(A_global, global_b.T)

    logger.info(f"Global Matrix Shape: {A_global.shape}")
    logger.info("System variant: 2x2 (dedicated script)")
    logger.info("Topology: line (fixed)")
    logger.info("Global Matrix A =\n" + str(A_global))
    logger.info("Condition number of A = " + str(np.linalg.cond(A_global)))
    logger.info(f"Global_b: {global_b}")
    logger.info("True solution = " + str(true_sol.T))

    n_qubits = len(DATA_WIRES)
    LAYERS = n_qubits  # number of layers in ansatz

    # ---- init params ----
    GLOBAL_PARAMS, key = initialize_global_params_jax(
        SYSTEM, n_qubits=n_qubits, layers=LAYERS, seed=args.seed
    )


    COL_TOPOLOGY = TOPOLOGY_LINE_2
    ROW_TOPOLOGY = COL_TOPOLOGY  # row same as col
    
    # metropolis W
    Wm = build_metropolis_matrix(COL_TOPOLOGY, n=SYSTEM.n, make_undirected=True)
    logger.info("Metropolis Weight Matrix Wm =\n" + str(Wm))

    ib.prebuild_local_evals(
        SYSTEM, ROW_TOPOLOGY,
        n_input_qubit=n_qubits,       
        diff_method="adjoint",
        interface="jax",
    )

    @qml.qjit
    def compute_grad(args):
        def total_loss_fn(args):
            current_params = rebuild_global_params(
                args,
                SYSTEM.n,
                GLOBAL_PARAMS["b_norm"]
        )
            return ib.eval_total_loss(current_params)
        
        return catalyst.grad(total_loss_fn, method="auto")(args)

    @qml.qjit
    def compute_loss(args):
        def total_loss_fn(args):
            current_params = rebuild_global_params(
                args,
                SYSTEM.n,
                GLOBAL_PARAMS["b_norm"]
            )
            return ib.eval_total_loss(current_params)

        return total_loss_fn(args)

    current_loss = compute_loss(to_jax_flat(flatten_params(GLOBAL_PARAMS)))
    print(f"Initial Total Loss: {current_loss}")

    lr_schedule = optax.exponential_decay(
    init_value=args.lr,
    transition_steps=1,   # 每步衰减
    decay_rate=args.decay,
    staircase=False,
)

    # single optimizer for all trainable params
    opt_adam = optax.adam(lr_schedule)

    loss_history, diff_history, ax_minus_b_history = [], [], []

    # ==========================================    
    # Start optimization
    # ==========================================

    # ---- Phase 0: init tracker y(0)=g(0) ----
    flat_params_init = to_jax_flat(flatten_params(GLOBAL_PARAMS, keys=None))
    grads_flat_init = compute_grad(flat_params_init)
    current_cost = compute_loss(flat_params_init)
    
    grad_grid_init = rebuild_global_params(grads_flat_init, 
                                           SYSTEM.n, 
                                           GLOBAL_PARAMS["b_norm"])

    tracker_grid = init_tracker_from_grad(grad_grid_init)
    prev_grad_grid = copy.deepcopy(grad_grid_init)

    all_keys = ["alpha", "beta", "sigma", "lambda"]
    opt_adam_state = opt_adam.init(to_jax_flat(flatten_params(GLOBAL_PARAMS, keys=all_keys)))

    loss_history.append(float(current_cost))
    logger.info(f"[Init] Initial Loss = {current_cost:.5e}")

    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        # consensus for alpha/sigma
        GLOBAL_PARAMS = consensus_mix_metropolis_jax(GLOBAL_PARAMS, W_np=Wm)

        # apply tracker as gradient to all trainable params
        flat_tracker = to_jax_flat(flatten_params(tracker_grid, keys=all_keys))
        flat_params = to_jax_flat(flatten_params(GLOBAL_PARAMS, keys=all_keys))
        adam_updates, opt_adam_state = opt_adam.update(flat_tracker, opt_adam_state, params=flat_params)
        new_flat_params = optax.apply_updates(flat_params, adam_updates)
        update_global_from_flat(GLOBAL_PARAMS, new_flat_params, keys=all_keys)

        # compute current grads
        flat_params_all = to_jax_flat(flatten_params(GLOBAL_PARAMS, keys=None))
        grads_flat = compute_grad(flat_params_all)
        current_cost = compute_loss(flat_params_all)
        
        current_grad_grid = rebuild_global_params(grads_flat, 
                                                  SYSTEM.n, 
                                                  GLOBAL_PARAMS["b_norm"])

        # tracker update
        tracker_grid = update_gradient_tracker_metropolis_jax(
            current_tracker=tracker_grid,
            current_grads=current_grad_grid,
            prev_grads=prev_grad_grid,
            W_np=Wm
        )
        prev_grad_grid = copy.deepcopy(current_grad_grid)

        # metrics
        loss_history.append(float(current_cost))
        recover_sol = recover_global_solution(GLOBAL_PARAMS, n_qubits=n_qubits)
        sol_diff = cal_sol_diff(
            GLOBAL_PARAMS,
            true_sol=true_sol,
            n_qubits=n_qubits,
            recover_sol=recover_sol,
        )
        ax_minus_b = cal_ax_minus_b_norm(
            GLOBAL_PARAMS,
            A_global=A_global,
            global_b=global_b,
            n_qubits=n_qubits,
            recover_sol=recover_sol,
        )
        diff_history.append(float(sol_diff))
        ax_minus_b_history.append(float(ax_minus_b))


        if (ep % args.log_every) == 0 or ep == 1:
            metrics.write({
            "epoch": ep,
            "loss": float(current_cost),
            "sol_diff": float(sol_diff),
            "ax_minus_b_norm": float(ax_minus_b),
            "lr_g": float(args.lr),
            "lr_a": float(args.lr),
            "wall_s": time.time() - t0,
            })
            
            logger.info(
                f"[Epoch {ep:04d}] Total Loss = {current_cost:.5e} | "
                f"Sol Diff = {sol_diff:.5e} | ||Ax-b|| = {ax_minus_b:.5e}"
            )

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

    title = f"kappa≈{kappa_int} | lr_g0={args.lr:g}, lr_a0={args.lr:g} | {opt_tag}"

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

    # ---- ||Ax-b|| figure ----
    xs3 = np.arange(1, len(ax_minus_b_history) + 1)
    plt.figure()
    plt.plot(xs3, ax_minus_b_history)
    plt.xlabel("Epoch")
    plt.ylabel("||Ax-b||")
    plt.yscale("log")
    plt.grid(True)
    plt.title(title)
    plt.savefig(paths.run_dir / "ax_minus_b_norm.png", dpi=200, bbox_inches="tight")
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
    np.savez(
        paths.artifacts_npz,
        loss=np.array(loss_history),
        sol_diff=np.array(diff_history),
        ax_minus_b_norm=np.array(ax_minus_b_history),
    )

    # ---- extra analysis / verification ----
    analysis_path = paths.run_dir / "analysis.txt"
    with open(analysis_path, "w", encoding="utf-8") as f:
        recover_and_verify_solution(GLOBAL_PARAMS, SYSTEM, DATA_WIRES, out=f)
        analyze_consensus_variance(GLOBAL_PARAMS, SYSTEM, DATA_WIRES, out=f)

    logger.info(f"Post-analysis written to: {analysis_path}")

    logger.info("Finished. Outputs written to: " + str(paths.run_dir))

if __name__ == "__main__":
    main()
