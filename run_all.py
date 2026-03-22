"""
Run all experiments from RESEARCH_PLAN_TOP_JOURNAL.md: multiple seeds and models.
Generates one JSON metrics file per (model, seed) and optionally one aggregated summary.
Usage:
  python run_all.py --csv EMOFF_MEME.csv --image_dir my_meme_data/my_meme_data --filter_missing
  python run_all.py --models capmeme_no_kg text_only --seeds 42 123 456
"""
import argparse
import json
import os
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="EMOFF_MEME.csv")
    p.add_argument("--image_dir", default="my_meme_data/my_meme_data")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--models", nargs="+", default=None, help="Default: all MODEL_NAMES")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--filter_missing", action="store_true")
    p.add_argument("--save_splits", default="outputs/splits.json", help="Save once and reuse")
    p.add_argument("--use_splits", default=None, help="Reuse existing splits JSON")
    p.add_argument("--use_kg", action="store_true", help="Only for capmeme (slow)")
    p.add_argument("--gpu", type=int, default=None, help="GPU id for train.py (e.g. 1 or 4 if 0 is busy)")
    p.add_argument("--skip_existing", action="store_true", help="Skip (model, seed) if metrics file already exists")
    p.add_argument("--dry_run", action="store_true", help="Print commands only")
    args = p.parse_args()

    from model import MODEL_NAMES
    models = args.models or MODEL_NAMES
    # For capmeme_no_kg we never use_kg in data
    base_cmd = [
        sys.executable, "train.py",
        "--csv", args.csv,
        "--image_dir", args.image_dir,
        "--output_dir", args.output_dir,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
    ]
    if args.filter_missing:
        base_cmd.append("--filter_missing")
    if args.use_splits:
        base_cmd += ["--splits_file", args.use_splits]
    elif args.save_splits:
        # First run will create and save splits; subsequent runs need to reuse them
        pass  # add per iteration below

    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    first = True
    for model in models:
        for seed in args.seeds:
            use_kg = args.use_kg and model == "capmeme"
            cmd = base_cmd + [
                "--model", model,
                "--seed", str(seed),
                "--run_name", f"{model}_seed{seed}",
            ]
            if args.save_splits and not args.use_splits:
                if first:
                    cmd += ["--save_splits", args.save_splits]
                    first = False
                else:
                    cmd += ["--splits_file", args.save_splits]
            if use_kg:
                cmd.append("--use_kg")
            if model == "capmeme_no_emotion":
                cmd += ["--affect_weight", "0"]
            if args.gpu is not None:
                cmd += ["--gpu", str(args.gpu)]
            if args.dry_run:
                print(" ".join(cmd))
                continue
            metrics_file = os.path.join(args.output_dir, f"{model}_seed{seed}_metrics.json")
            if args.skip_existing and os.path.isfile(metrics_file):
                try:
                    with open(metrics_file) as f:
                        m = json.load(f)
                    results.append({"model": model, "seed": seed, "test_binary_f1": m.get("binary_f1"), "test_macro_f1": m.get("macro_f1")})
                except Exception:
                    results.append({"model": model, "seed": seed, "status": "skip_read_failed"})
                print(f"Skipped (existing): {model} seed {seed}")
                continue
            env = os.environ.copy()
            if args.gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            ret = subprocess.run(cmd, env=env)
            if ret.returncode != 0:
                print(f"Failed: {' '.join(cmd)}", file=sys.stderr)
                results.append({"model": model, "seed": seed, "status": "failed"})
            else:
                metrics_file = os.path.join(args.output_dir, f"{model}_seed{seed}_metrics.json")
                if os.path.isfile(metrics_file):
                    with open(metrics_file) as f:
                        m = json.load(f)
                    results.append({"model": model, "seed": seed, "test_binary_f1": m.get("binary_f1"), "test_macro_f1": m.get("macro_f1")})
                else:
                    results.append({"model": model, "seed": seed, "status": "no_metrics"})

    if not args.dry_run and results:
        summary_path = os.path.join(args.output_dir, "run_all_summary.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
