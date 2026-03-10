#!/usr/bin/env python3
"""Aggregate Qiskit benchmark JSON outputs into markdown/CSV report."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import median


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Qiskit benchmark report.")
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--output-csv", default=None)
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


def status_count(records: list[dict], status: str) -> int:
    return sum(1 for r in records if r["status"] == status)


def short_text(text: str, limit: int = 140) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def load_record(path: Path, suite_dir: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    cfg = payload.get("configuration", {})
    timings = payload.get("timings", {})
    eval_stats = timings.get("eval_seconds", {})
    grad_stats = timings.get("grad_seconds", {})
    err = payload.get("error", {})

    target = cfg.get("compute_target")
    if not target:
        try:
            target = path.relative_to(suite_dir).parts[1]
        except Exception:  # noqa: BLE001
            target = "unknown"

    md = payload.get("metadata", {}).get("eval_result_metadata", {})
    sim_md = md.get("simulator_metadata", {})
    return {
        "path": str(path.relative_to(suite_dir)),
        "status": payload.get("status", "unknown"),
        "target": target,
        "eval_mode": cfg.get("eval_mode"),
        "gradient_method": cfg.get("gradient_method"),
        "num_parameters": cfg.get("num_parameters"),
        "sampler_shots": cfg.get("sampler_shots"),
        "eval_mean_s": as_float(eval_stats.get("mean")),
        "grad_mean_s": as_float(grad_stats.get("mean")),
        "expectation": as_float(payload.get("metrics", {}).get("last_expectation")),
        "grad_l2": as_float(payload.get("metrics", {}).get("gradient_l2_norm")),
        "backend_device_requested": cfg.get("backend_options", {}).get("device"),
        "sim_max_gpu_memory_mb": as_float(sim_md.get("max_gpu_memory_mb")),
        "error_message": err.get("message"),
    }


def best_row(rows: list[dict], key: str):
    valid = [r for r in rows if r[key] is not None]
    if not valid:
        return None
    return min(valid, key=lambda r: r[key])


def env_delta(before: dict | None, after: dict | None) -> list[str]:
    if not before or not after:
        return ["Environment snapshots missing or incomplete."]
    lines = []
    bp = before.get("packages", {})
    ap = after.get("packages", {})
    keys = sorted(set(bp.keys()) | set(ap.keys()))
    for k in keys:
        b = bp.get(k, "missing")
        a = ap.get(k, "missing")
        if b == a:
            lines.append(f"- {k}: {a} (unchanged)")
        else:
            lines.append(f"- {k}: {b} -> {a}")
    return lines


def main() -> int:
    args = parse_args()
    suite_dir = Path(args.suite_dir).resolve()
    output_md = Path(args.output_md) if args.output_md else (suite_dir / "report.md")
    output_csv = Path(args.output_csv) if args.output_csv else (suite_dir / "summary.csv")

    cfg_path = suite_dir / "suite_config.json"
    cfg = {}
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

    before_path = suite_dir / "env_versions_before.json"
    after_path = suite_dir / "env_versions_after.json"
    before = None
    after = None
    if before_path.exists():
        with before_path.open("r", encoding="utf-8") as f:
            before = json.load(f)
    if after_path.exists():
        with after_path.open("r", encoding="utf-8") as f:
            after = json.load(f)

    json_files = sorted((suite_dir / "results").glob("*/*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON results found under {suite_dir / 'results'}")

    rows = [load_record(p, suite_dir) for p in json_files]
    ok_rows = [r for r in rows if r["status"] == "ok"]
    by_target = {
        "cpu": [r for r in rows if r["target"] == "cpu"],
        "gpu": [r for r in rows if r["target"] == "gpu"],
    }

    cpu_map = {(r["eval_mode"], r["gradient_method"]): r for r in by_target["cpu"] if r["status"] == "ok"}
    gpu_map = {(r["eval_mode"], r["gradient_method"]): r for r in by_target["gpu"] if r["status"] == "ok"}
    shared = sorted(set(cpu_map) & set(gpu_map))
    eval_speedups = []
    grad_speedups = []
    for key in shared:
        c = cpu_map[key]
        g = gpu_map[key]
        if c["eval_mean_s"] and g["eval_mean_s"]:
            eval_speedups.append(c["eval_mean_s"] / g["eval_mean_s"])
        if c["grad_mean_s"] and g["grad_mean_s"]:
            grad_speedups.append(c["grad_mean_s"] / g["grad_mean_s"])

    lines: list[str] = []
    lines.append("# Qiskit 20-Qubit Hadamard Test Benchmark Report")
    lines.append("")
    lines.append(f"- Generated: {datetime.utcnow().isoformat()}Z")
    lines.append(f"- Suite directory: `{suite_dir}`")
    lines.append(
        "- Circuit: Hadamard test with shallow variational ansatz "
        f"(total qubits={cfg.get('num_qubits_total', 20)}, layers={cfg.get('layers', 'unknown')})."
    )
    lines.append(
        f"- SamplerV2 shots: {cfg.get('sampler_shots', 'unknown')} (as requested, no noise model)."
    )
    lines.append("")
    lines.append("## Execution Summary")
    lines.append("")
    lines.append("| Target | Total | OK | Unsupported | Failed | Missing Dependency |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for target in ("cpu", "gpu"):
        r = by_target[target]
        lines.append(
            f"| {target} | {len(r)} | {status_count(r, 'ok')} | {status_count(r, 'unsupported')} | "
            f"{status_count(r, 'failed')} | {status_count(r, 'missing_dependency')} |"
        )

    lines.append("")
    lines.append("## Successful Configurations")
    lines.append("")
    lines.append(
        "| Target | Eval Mode | Gradient | Eval Mean (s) | Grad Mean (s) | "
        "Expectation | Grad L2 | GPU Mem MB |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for r in sorted(ok_rows, key=lambda x: (x["target"], x["eval_mode"], x["gradient_method"])):
        lines.append(
            f"| {r['target']} | {r['eval_mode']} | {r['gradient_method']} | "
            f"{fmt(r['eval_mean_s'])} | {fmt(r['grad_mean_s'])} | {fmt(r['expectation'])} | "
            f"{fmt(r['grad_l2'])} | {fmt(r['sim_max_gpu_memory_mb'], 2)} |"
        )

    lines.append("")
    lines.append("## Best Cases")
    lines.append("")
    lines.append("| Target | Fastest Eval | Eval Mean (s) | Fastest Gradient | Grad Mean (s) |")
    lines.append("|---|---|---:|---|---:|")
    for target in ("cpu", "gpu"):
        ok_t = [r for r in by_target[target] if r["status"] == "ok"]
        be = best_row(ok_t, "eval_mean_s")
        bg = best_row(ok_t, "grad_mean_s")
        eval_cfg = f"{be['eval_mode']} + {be['gradient_method']}" if be else "NA"
        grad_cfg = f"{bg['eval_mode']} + {bg['gradient_method']}" if bg else "NA"
        lines.append(
            f"| {target} | {eval_cfg} | {fmt(be['eval_mean_s']) if be else 'NA'} | "
            f"{grad_cfg} | {fmt(bg['grad_mean_s']) if bg else 'NA'} |"
        )

    if rows and any(r["status"] != "ok" for r in rows):
        lines.append("")
        lines.append("## Unsupported / Failed")
        lines.append("")
        lines.append("| Target | Eval Mode | Gradient | Status | Reason |")
        lines.append("|---|---|---|---|---|")
        for r in rows:
            if r["status"] == "ok":
                continue
            lines.append(
                f"| {r['target']} | {r['eval_mode']} | {r['gradient_method']} | {r['status']} | "
                f"{short_text(r['error_message'] or 'No message')} |"
            )

    lines.append("")
    lines.append("## Version Check")
    lines.append("")
    lines.extend(env_delta(before, after))

    lines.append("")
    lines.append("## Commentary")
    lines.append("")
    if eval_speedups:
        lines.append(f"- Median CPU/GPU eval speedup (CPU_time / GPU_time): {median(eval_speedups):.3f}.")
    else:
        lines.append("- No matched successful CPU/GPU pairs to compute eval speedup.")
    if grad_speedups:
        lines.append(f"- Median CPU/GPU gradient speedup (CPU_time / GPU_time): {median(grad_speedups):.3f}.")
    else:
        lines.append("- No matched successful CPU/GPU pairs to compute gradient speedup.")
    lines.append(
        "- SamplerV2 results include shot noise (shots=1024), while estimator-based values are deterministic in this setup."
    )

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    fieldnames = [
        "path",
        "status",
        "target",
        "eval_mode",
        "gradient_method",
        "num_parameters",
        "sampler_shots",
        "eval_mean_s",
        "grad_mean_s",
        "expectation",
        "grad_l2",
        "backend_device_requested",
        "sim_max_gpu_memory_mb",
        "error_message",
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})

    print(f"REPORT_MD={output_md.resolve()}")
    print(f"SUMMARY_CSV={output_csv.resolve()}")
    print(f"TOTAL_RESULTS={len(rows)}")
    print(f"OK_RESULTS={len(ok_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
