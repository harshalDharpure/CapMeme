#!/usr/bin/env python3
"""
Run this once all 21 experiments are done (all *_metrics.json present).
Updates paper tables (mean ± std, best bolded) and saves test predictions for image_only if missing.
Usage: python run_when_all_done.py
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

EXPECTED = 21  # 7 models x 3 seeds
output_dir = os.path.join(ROOT, "outputs")
metrics_pattern = os.path.join(output_dir, "*_seed*_metrics.json")

import glob
files = [f for f in glob.glob(metrics_pattern) if "aggregate" not in f]
n = len(files)
if n < EXPECTED:
    print(f"Only {n}/{EXPECTED} runs have metrics. Wait for pipeline to finish, then run again.")
    sys.exit(1)

print(f"All {n} runs have metrics. Updating tables and predictions...")
subprocess.run([sys.executable, "aggregate_results.py", "--output_dir", "outputs", "--update_tables"], check=True)
subprocess.run([
    sys.executable, "save_test_predictions.py",
    "--output_dir", "outputs", "--splits_file", "outputs/splits.json", "--filter_missing", "--gpu", "0"
], check=True)
print("Done. Tables: RESULTS_TABLES_FOR_PAPER.md. Predictions: outputs/predictions/")
