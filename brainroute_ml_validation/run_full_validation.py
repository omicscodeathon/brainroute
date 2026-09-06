#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

STEPS = [
    "01_standardize_and_audit.py",
    "02_calculate_morgan_fingerprints.py",
    "03_calculate_pretrained_embeddings.py",
    "04_build_feature_matrices.py",
    "05_create_validation_splits.py",
    "06_near_duplicate_analysis.py",
    "07_leakage_controls.py",
    "08_train_models.py",
    "09_external_validation.py",
    "10_statistical_comparison.py",
    "11_make_summary_tables.py",
    "13_reconstruct_provenance.py",
    "14_revision_reanalysis.py",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the BrainRoute strict-validation workflow.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "validation_config.yaml"))
    parser.add_argument("--start-at", default=None, help="Optional script filename to start from.")
    parser.add_argument("--stop-after", default=None, help="Optional script filename to stop after.")
    args = parser.parse_args()

    steps = STEPS[:]
    if args.start_at:
        steps = steps[steps.index(args.start_at) :]
    if args.stop_after:
        steps = steps[: steps.index(args.stop_after) + 1]

    for step in steps:
        script = ROOT / "scripts" / step
        print(f"\n=== Running {step} ===", flush=True)
        env = os.environ.copy()
        repo_root = str(ROOT.parent)
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = repo_root if not existing_pythonpath else f"{repo_root}{os.pathsep}{existing_pythonpath}"
        result = subprocess.run([sys.executable, str(script), "--config", args.config], cwd=ROOT.parent, env=env)
        if result.returncode != 0:
            print(f"Step failed: {step}", file=sys.stderr)
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
