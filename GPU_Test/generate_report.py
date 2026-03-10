#!/usr/bin/env python3
"""Aggregate JSON benchmark outputs into a markdown report and CSV summary."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import median


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark report from JSON result files.")
    parser.add_argument("--suite-dir", required=True, help="Path created by generate_tasks.py.")
    parser.add_argument("--output-md", default=None, help="Defaults to <suite-dir>/report.md")
    parser.add_argument("--output-csv", default=None, help="Defaults to <suite-dir>/summary.csv")
    return parser.parse_args()


def as_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value, digits=6):
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def short_text(text: str, limit: int = 140) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def load_record(json_path: Path, suite_dir: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    cfg = payload.get("configuration", {})
    timings = payload.get("timings", {})
    eval_stats = timings.get("eval_seconds", {})
    grad_stats = timings.get("grad_seconds", {})
    err = payload.get("error", {})

    compute_target = cfg.get("compute_target")
    if not compute_target:
        try:
            compute_target = json_path.relative_to(suite_dir).parts[1]
        except Exception:  # noqa: BLE001
            compute_target = "unknown"

    return {
        "path": str(json_path.relative_to(suite_dir)),
        "status": payload.get("status", "unknown"),
        "compute_target": compute_target,
        "device": cfg.get("device"),
        "interface": cfg.get("interface_requested") or cfg.get("interface"),
        "diff_method": cfg.get("diff_method_requested") or cfg.get("diff_method"),
        "num_parameters": cfg.get("num_parameters"),
        "eval_mean_s": as_float(eval_stats.get("mean")),
        "grad_mean_s": as_float(grad_stats.get("mean")),
        "expectation": as_float(payload.get("metrics", {}).get("last_expectation")),
        "grad_l2": as_float(payload.get("metrics", {}).get("gradient_l2_norm")),
        "error_message": err.get("message"),
    }


def status_count(records: list[dict], status: str) -> int:
    return sum(1 for r in records if r["status"] == status)


def best_row(rows: list[dict], key: str):
    valid = [r for r in rows if r.get(key) is not None]
    if not valid:
        return None
    return min(valid, key=lambda r: r[key])


def main() -> int:
    args = parse_args()
    suite_dir = Path(args.suite_dir).resolve()
    output_md = Path(args.output_md) if args.output_md else (suite_dir / "report.md")
    output_csv = Path(args.output_csv) if args.output_csv else (suite_dir / "summary.csv")
    suite_cfg_path = suite_dir / "suite_config.json"
    suite_cfg = {}
    if suite_cfg_path.exists():
        with suite_cfg_path.open("r", encoding="utf-8") as f:
            suite_cfg = json.load(f)

    json_files = sorted((suite_dir / "results").glob("*/*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON results found under {suite_dir / 'results'}")

    records = [load_record(path, suite_dir) for path in json_files]
    ok_records = [r for r in records if r["status"] == "ok"]

    by_target = {
        "cpu": [r for r in records if r["compute_target"] == "cpu"],
        "gpu": [r for r in records if r["compute_target"] == "gpu"],
    }

    # Matched speedups for successful configurations.
    cpu_ok_map = {
        (r["device"], r["interface"], r["diff_method"]): r
        for r in by_target["cpu"]
        if r["status"] == "ok"
    }
    gpu_ok_map = {
        (r["device"], r["interface"], r["diff_method"]): r
        for r in by_target["gpu"]
        if r["status"] == "ok"
    }
    shared_keys = sorted(set(cpu_ok_map.keys()) & set(gpu_ok_map.keys()))
    eval_speedups = []
    grad_speedups = []
    for key in shared_keys:
        cpu_row = cpu_ok_map[key]
        gpu_row = gpu_ok_map[key]
        if cpu_row["eval_mean_s"] and gpu_row["eval_mean_s"]:
            eval_speedups.append(cpu_row["eval_mean_s"] / gpu_row["eval_mean_s"])
        if cpu_row["grad_mean_s"] and gpu_row["grad_mean_s"]:
            grad_speedups.append(cpu_row["grad_mean_s"] / gpu_row["grad_mean_s"])

    lines: list[str] = []
    lines.append("# 20-Qubit Hadamard Test PennyLane Benchmark Report")
    lines.append("")
    lines.append(f"- Generated: {datetime.utcnow().isoformat()}Z")
    lines.append(f"- Suite directory: `{suite_dir}`")
    total_qubits = suite_cfg.get("num_qubits_total", 20)
    system_qubits = suite_cfg.get("num_system_qubits", 19)
    layers = suite_cfg.get("layers", "unknown")
    lines.append(
        "- Circuit definition: Hadamard test with one ancilla + "
        f"{system_qubits} system qubits ({total_qubits} total), "
        f"shallow `StronglyEntanglingLayers` ansatz (layers={layers})."
    )
    lines.append(
        "- Sweep dimensions: device (`default.qubit`, `lightning.qubit`), "
        "interface (`numpy`, `torch`, `jax`), diff method (`backprop`, `adjoint`, `finite_diff`), "
        "compute target (`cpu`, `gpu`)."
    )
    lines.append("")
    lines.append("## Execution Summary")
    lines.append("")
    lines.append("| Target | Total | OK | Unsupported | Failed | Missing Dependency |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for target in ("cpu", "gpu"):
        rows = by_target[target]
        lines.append(
            f"| {target} | {len(rows)} | {status_count(rows, 'ok')} | "
            f"{status_count(rows, 'unsupported')} | {status_count(rows, 'failed')} | "
            f"{status_count(rows, 'missing_dependency')} |"
        )

    lines.append("")
    lines.append("## Best Configurations")
    lines.append("")
    lines.append("| Target | Fastest Eval Config | Eval Mean (s) | Fastest Grad Config | Grad Mean (s) |")
    lines.append("|---|---|---:|---|---:|")
    for target in ("cpu", "gpu"):
        ok_rows = [r for r in by_target[target] if r["status"] == "ok"]
        best_eval = best_row(ok_rows, "eval_mean_s")
        best_grad = best_row(ok_rows, "grad_mean_s")
        eval_cfg = (
            f"{best_eval['device']}, {best_eval['interface']}, {best_eval['diff_method']}"
            if best_eval
            else "NA"
        )
        grad_cfg = (
            f"{best_grad['device']}, {best_grad['interface']}, {best_grad['diff_method']}"
            if best_grad
            else "NA"
        )
        lines.append(
            f"| {target} | {eval_cfg} | {fmt(best_eval['eval_mean_s']) if best_eval else 'NA'} | "
            f"{grad_cfg} | {fmt(best_grad['grad_mean_s']) if best_grad else 'NA'} |"
        )

    lines.append("")
    lines.append("## Successful Runs (sorted by grad time)")
    lines.append("")
    lines.append(
        "| Target | Device | Interface | Diff Method | Eval Mean (s) | Grad Mean (s) | "
        "Params | Expectation | Grad L2 |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
    ok_sorted = sorted(
        ok_records,
        key=lambda r: (
            r["compute_target"],
            float("inf") if r["grad_mean_s"] is None else r["grad_mean_s"],
            float("inf") if r["eval_mean_s"] is None else r["eval_mean_s"],
        ),
    )
    for r in ok_sorted:
        lines.append(
            f"| {r['compute_target']} | {r['device']} | {r['interface']} | {r['diff_method']} | "
            f"{fmt(r['eval_mean_s'])} | {fmt(r['grad_mean_s'])} | "
            f"{r['num_parameters'] if r['num_parameters'] is not None else 'NA'} | "
            f"{fmt(r['expectation'])} | {fmt(r['grad_l2'])} |"
        )

    bad_rows = [r for r in records if r["status"] != "ok"]
    if bad_rows:
        lines.append("")
        lines.append("## Unsupported / Failed Configurations")
        lines.append("")
        lines.append("| Target | Device | Interface | Diff Method | Status | Reason |")
        lines.append("|---|---|---|---|---|---|")
        for r in bad_rows:
            reason = short_text(r["error_message"] or "No error message captured")
            lines.append(
                f"| {r['compute_target']} | {r['device']} | {r['interface']} | {r['diff_method']} | "
                f"{r['status']} | {reason} |"
            )

    lines.append("")
    lines.append("## Commentary")
    lines.append("")
    lines.append(
        f"- Successful configurations: {len(ok_records)} / {len(records)} total. "
        "Unsupported combinations are expected for some interface/device/diff-method pairs in PennyLane."
    )
    if eval_speedups:
        lines.append(
            f"- For matched successful CPU/GPU configs, median eval speedup (CPU_time / GPU_time) is {median(eval_speedups):.3f}."
        )
    else:
        lines.append("- No matched successful CPU/GPU pairs were available to compute eval speedup.")
    if grad_speedups:
        lines.append(
            f"- For matched successful CPU/GPU configs, median gradient speedup (CPU_time / GPU_time) is {median(grad_speedups):.3f}."
        )
    else:
        lines.append("- No matched successful CPU/GPU pairs were available to compute gradient speedup.")
    lines.append(
        "- If GPU speedup is limited, likely causes are non-GPU simulator backends or host-device transfer overhead."
    )

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    fieldnames = [
        "path",
        "status",
        "compute_target",
        "device",
        "interface",
        "diff_method",
        "num_parameters",
        "eval_mean_s",
        "grad_mean_s",
        "expectation",
        "grad_l2",
        "error_message",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k) for k in fieldnames})

    print(f"REPORT_MD={output_md.resolve()}")
    print(f"SUMMARY_CSV={output_csv.resolve()}")
    print(f"TOTAL_RESULTS={len(records)}")
    print(f"OK_RESULTS={len(ok_records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
