# common/reporting.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json, time
import logging

@dataclass
class RunPaths:
    run_dir: Path
    report_txt: Path
    metrics_jsonl: Path
    fig_loss: Path
    fig_diff: Path
    artifacts_npz: Path

def make_run_dir(out_dir: str | Path) -> RunPaths:
    run_dir = Path(out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        run_dir=run_dir,
        report_txt=run_dir / "report.txt",
        metrics_jsonl=run_dir / "metrics.jsonl",
        fig_loss=run_dir / "loss.png",
        fig_diff=run_dir / "sol_diff.png",
        artifacts_npz=run_dir / "artifacts.npz",
    )

def setup_logger(report_path: Path):
    logger = logging.getLogger("dvqls")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(report_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

class JsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self.f = open(path, "w", encoding="utf-8")

    def write(self, obj: dict):
        self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()
