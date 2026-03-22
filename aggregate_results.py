"""
Aggregate metrics from outputs/<model>_seed<seed>_metrics.json.
Computes mean ± std over seeds (42, 123, 456) per model for paper tables.
Usage:
  python aggregate_results.py --output_dir outputs
  python aggregate_results.py --output_dir outputs --update_tables
"""
import argparse
import json
import os
import glob
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    p.add_argument("--update_tables", action="store_true", help="Write RESULTS_TABLES_FOR_PAPER.md with mean±std")
    args = p.parse_args()

    output_dir = os.path.join(ROOT, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    metrics_keys = ["accuracy", "binary_f1", "macro_f1", "weighted_f1", "precision", "recall", "roc_auc", "pr_auc"]
    pattern = os.path.join(output_dir, "*_seed*_metrics.json")
    files = sorted(glob.glob(pattern))
    by_model = {}
    for path in files:
        basename = os.path.basename(path)
        rest = basename.replace("_metrics.json", "")
        if "_seed" not in rest:
            continue
        model = rest.split("_seed")[0]
        seed_str = rest.split("_seed")[1]
        try:
            seed = int(seed_str)
        except ValueError:
            continue
        if seed not in args.seeds:
            continue
        with open(path) as f:
            m = json.load(f)
        if model not in by_model:
            by_model[model] = []
        by_model[model].append({"seed": seed, **m})

    import numpy as np
    aggregate = {}
    for model, runs in by_model.items():
        if len(runs) == 0:
            continue
        agg = {"model": model, "n_seeds": len(runs), "seeds": [r["seed"] for r in runs]}
        for key in metrics_keys:
            vals = [r.get(key) for r in runs if r.get(key) is not None]
            if not vals:
                continue
            vals = np.array(vals)
            if key in ("accuracy", "binary_f1", "macro_f1", "weighted_f1", "precision", "recall"):
                vals = vals * 100.0
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
        aggregate[model] = agg

    out_path = os.path.join(output_dir, "aggregate_metrics.json")
    with open(out_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"Wrote {out_path}")

    if args.update_tables:
        from model import MODEL_NAMES
        baseline_models = ["text_only", "image_only", "late_fusion"]
        proposed_models = [m for m in MODEL_NAMES if m not in baseline_models]

        def fmt(mean, std):
            if mean is None:
                return "—"
            if std is None or std == 0:
                return f"{mean:.2f}"
            return f"{mean:.2f} ± {std:.2f}"

        def row(model, agg):
            if not agg:
                return "| {} | — | — | — | — | — |".format(model.replace("_", " "))
            roc = agg.get("roc_auc_mean"), agg.get("roc_auc_std")
            return "| {} | {} | {} | {} | {} | {} |".format(
                model.replace("_", " "),
                fmt(agg.get("binary_f1_mean"), agg.get("binary_f1_std")),
                fmt(agg.get("accuracy_mean"), agg.get("accuracy_std")),
                fmt(agg.get("precision_mean"), agg.get("precision_std")),
                fmt(agg.get("recall_mean"), agg.get("recall_std")),
                fmt(roc[0], roc[1]) if roc[0] is not None else "—",
            )

        all_means = {m: ag for m, ag in aggregate.items() if ag.get("binary_f1_mean") is not None}
        best_f1 = max((a["binary_f1_mean"], m) for m, a in all_means.items())[1] if all_means else None
        best_acc = max((a["accuracy_mean"], m) for m, a in all_means.items())[1] if all_means else None

        def bold_cell(r, model, agg, col_key):
            if not agg or model not in all_means:
                return r
            best = best_f1 if col_key == "binary_f1_mean" else best_acc if col_key == "accuracy_mean" else None
            if model != best or agg.get(col_key) is None:
                return r
            val = agg[col_key]
            return r.replace(f"{val:.2f}", f"**{val:.2f}**", 1)

        lines = []
        lines.append("# CAPMeme — Results Tables for Paper")
        lines.append("")
        lines.append("Task: Binary sarcasm detection (Sarcastic vs Non-sarcastic) on EMOFF_MEME Hindi memes. Metrics: F1, A (Accuracy %), P (Precision %), R (Recall %), ROC-AUC. Modality: T = Text, V = Visual. Mean ± std over seeds; best per column in **bold**.")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## Table 1 — Main results: Models for sarcasm detection")
        lines.append("")
        lines.append("| Model | T | V | F1 | A | P | R | ROC-AUC |")
        lines.append("|-------|---|---|-----|-----|-----|-----|--------|")
        for model in baseline_models + proposed_models:
            ag = aggregate.get(model, {})
            t = "✓" if model != "image_only" else "—"
            v = "✓" if model != "text_only" else "—"
            if not ag or ag.get("binary_f1_mean") is None:
                r = "| {} | {} | {} | — | — | — | — | — |".format(model.replace("_", " "), t, v)
            else:
                r = "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    model.replace("_", " "), t, v,
                    fmt(ag.get("binary_f1_mean"), ag.get("binary_f1_std")),
                    fmt(ag.get("accuracy_mean"), ag.get("accuracy_std")),
                    fmt(ag.get("precision_mean"), ag.get("precision_std")),
                    fmt(ag.get("recall_mean"), ag.get("recall_std")),
                    fmt(ag.get("roc_auc_mean"), ag.get("roc_auc_std")) if ag.get("roc_auc_mean") is not None else "—",
                )
            r = bold_cell(r, model, ag, "binary_f1_mean")
            r = bold_cell(r, model, ag, "accuracy_mean")
            lines.append(r)
        lines.append("")
        lines.append("## Table 2 — Ablation 1: Effect of components (KG, emotion)")
        lines.append("")
        lines.append("| Model | T | V | F1 | A | P | R |")
        lines.append("|-------|---|---|-----|-----|-----|-----|")
        for model in ["capmeme", "capmeme_no_kg", "capmeme_no_emotion"]:
            ag = aggregate.get(model, {})
            if not ag or ag.get("binary_f1_mean") is None:
                r = "| {} | ✓ | ✓ | — | — | — | — |".format(model.replace("_", " "))
            else:
                r = "| {} | ✓ | ✓ | {} | {} | {} | {} |".format(
                    model.replace("_", " "),
                    fmt(ag.get("binary_f1_mean"), ag.get("binary_f1_std")),
                    fmt(ag.get("accuracy_mean"), ag.get("accuracy_std")),
                    fmt(ag.get("precision_mean"), ag.get("precision_std")),
                    fmt(ag.get("recall_mean"), ag.get("recall_std")),
                )
            r = bold_cell(r, model, ag, "binary_f1_mean")
            lines.append(r)
        lines.append("")
        lines.append("## Table 3 — Ablation 2: Fusion strategy")
        lines.append("")
        lines.append("| Model | T | V | F1 | A | P | R |")
        lines.append("|-------|---|---|-----|-----|-----|-----|")
        for model in ["capmeme", "capmeme_concat_fusion"]:
            ag = aggregate.get(model, {})
            if not ag or ag.get("binary_f1_mean") is None:
                r = "| {} | ✓ | ✓ | — | — | — | — |".format(model.replace("_", " "))
            else:
                r = "| {} | ✓ | ✓ | {} | {} | {} | {} |".format(
                    model.replace("_", " "),
                    fmt(ag.get("binary_f1_mean"), ag.get("binary_f1_std")),
                    fmt(ag.get("accuracy_mean"), ag.get("accuracy_std")),
                    fmt(ag.get("precision_mean"), ag.get("precision_std")),
                    fmt(ag.get("recall_mean"), ag.get("recall_std")),
                )
            r = bold_cell(r, model, ag, "binary_f1_mean")
            lines.append(r)
        lines.append("")
        lines.append("---")
        lines.append("Regenerate: `python aggregate_results.py --output_dir outputs --update_tables`")
        table_path = os.path.join(ROOT, "RESULTS_TABLES_FOR_PAPER.md")
        with open(table_path, "w") as f:
            f.write("\n".join(lines))
        print(f"Updated {table_path}")


if __name__ == "__main__":
    main()
