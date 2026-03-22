"""
McNemar's test for paired binary classifiers (same test set).
Usage:
  python mcnemar_significance.py --pred_dir outputs/predictions --model_a capmeme --model_b text_only --seed 42
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def mcnemar(y_true, y_a, y_b):
    b = sum(1 for i in range(len(y_true)) if y_a[i] == y_true[i] and y_b[i] != y_true[i])
    c = sum(1 for i in range(len(y_true)) if y_a[i] != y_true[i] and y_b[i] == y_true[i])
    if b + c == 0:
        return {"chi2": 0.0, "p_value": 1.0, "n_discordant": 0, "b": 0, "c": 0, "significant_005": False}
    chi2 = (b - c) ** 2 / (b + c)
    try:
        from scipy import stats
        p_value = 1 - stats.chi2.cdf(chi2, df=1)
    except ImportError:
        p_value = None
    return {
        "chi2": float(chi2),
        "p_value": float(p_value) if p_value is not None else None,
        "n_discordant": int(b + c),
        "b": int(b),
        "c": int(c),
        "significant_005": bool(p_value is not None and p_value < 0.05),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", default="outputs/predictions")
    p.add_argument("--model_a", required=True)
    p.add_argument("--model_b", required=True)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    pred_dir = os.path.join(ROOT, args.pred_dir) if not os.path.isabs(args.pred_dir) else args.pred_dir
    seeds = [args.seed] if args.seed is not None else [42, 123, 456]
    results = []
    for seed in seeds:
        path_a = os.path.join(pred_dir, f"{args.model_a}_seed{seed}.json")
        path_b = os.path.join(pred_dir, f"{args.model_b}_seed{seed}.json")
        if not os.path.isfile(path_a) or not os.path.isfile(path_b):
            print(f"Missing: {path_a} or {path_b}")
            continue
        with open(path_a) as f:
            d_a = json.load(f)
        with open(path_b) as f:
            d_b = json.load(f)
        y_true, y_a, y_b = d_a["y_true"], d_a["y_pred"], d_b["y_pred"]
        if len(y_true) != len(y_a) or len(y_true) != len(y_b):
            print(f"Length mismatch seed {seed}")
            continue
        r = mcnemar(y_true, y_a, y_b)
        r["model_a"], r["model_b"], r["seed"] = args.model_a, args.model_b, seed
        results.append({
            "model_a": r["model_a"], "model_b": r["model_b"], "seed": r["seed"],
            "chi2": float(r["chi2"]), "p_value": float(r["p_value"]) if r.get("p_value") is not None else None,
            "n_discordant": int(r["n_discordant"]), "b": int(r["b"]), "c": int(r["c"]),
            "significant_005": bool(r.get("significant_005", False)),
        })
        p_str = f"{r['p_value']:.4f}" if r['p_value'] is not None else "N/A"
        print(f"Seed {seed}: chi2={r['chi2']:.4f} p={p_str} (p<0.05: {r['significant_005']}) A better: {r['b']} B better: {r['c']}")

    if results:
        out_path = os.path.join(pred_dir, f"mcnemar_{args.model_a}_vs_{args.model_b}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
