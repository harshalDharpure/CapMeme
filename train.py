"""
Unified training script for CAPMeme and all baselines/ablations (RESEARCH_PLAN_TOP_JOURNAL.md).
Uses stratified train/val/test splits; supports filter_missing; full metrics (Accuracy, F1, ROC-AUC, PR-AUC).
"""
import argparse
import json
import os
import torch
from torch.utils.data import DataLoader

from dataset import MemeDataset
from model import build_model, MODEL_NAMES
from loss import joint_capmeme_loss
from data_utils import filter_missing_images, stratified_split, save_splits, load_splits
from metrics import compute_metrics


def collate_fn(batch):
    keys = [k for k in batch[0].keys()]
    out = {}
    for k in keys:
        if k == "text_raw":
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


def train_epoch(model, loader, optimizer, device, bce_w=1.0, affect_w=0.5):
    model.train()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    for batch in loader:
        for k in batch:
            if k != "text_raw" and hasattr(batch[k], "to"):
                batch[k] = batch[k].to(device)
        optimizer.zero_grad()
        logits, A_v, A_t = model(
            batch["pixel_values"],
            batch["input_ids"],
            batch["attention_mask"],
            kg_embedding=batch.get("kg_embedding"),
            kg_valid=batch.get("kg_valid"),
        )
        loss, _, _ = joint_capmeme_loss(
            logits, batch["sarcasm"], A_v, A_t, batch["emotion_target"],
            bce_weight=bce_w, affect_weight=affect_w,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_preds.extend(preds)
        all_labels.extend(batch["sarcasm"].cpu().numpy().astype(int))
        all_probs.extend(probs)
    return total_loss / len(loader), all_labels, all_preds, all_probs


@torch.no_grad()
def evaluate(model, loader, device, bce_w=1.0, affect_w=0.5):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    for batch in loader:
        for k in batch:
            if k != "text_raw" and hasattr(batch[k], "to"):
                batch[k] = batch[k].to(device)
        logits, A_v, A_t = model(
            batch["pixel_values"],
            batch["input_ids"],
            batch["attention_mask"],
            kg_embedding=batch.get("kg_embedding"),
            kg_valid=batch.get("kg_valid"),
        )
        loss, _, _ = joint_capmeme_loss(
            logits, batch["sarcasm"], A_v, A_t, batch["emotion_target"],
            bce_weight=bce_w, affect_weight=affect_w,
        )
        total_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_preds.extend(preds)
        all_labels.extend(batch["sarcasm"].cpu().numpy().astype(int))
        all_probs.extend(probs)
    return total_loss / len(loader), all_labels, all_preds, all_probs


def main():
    p = argparse.ArgumentParser(description="CAPMeme research training")
    p.add_argument("--csv", default="EMOFF_MEME.csv")
    p.add_argument("--image_dir", default="my_meme_data/my_meme_data")
    p.add_argument("--model", choices=MODEL_NAMES, default="capmeme_no_kg")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--run_name", default=None, help="e.g. capmeme_no_kg_seed42")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_ratio", type=float, default=0.70)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.15)
    p.add_argument("--filter_missing", action="store_true", help="Drop rows with missing images")
    p.add_argument("--splits_file", default=None, help="JSON with train_idx, val_idx, test_idx (optional)")
    p.add_argument("--save_splits", default=None, help="Save splits to this JSON path")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--kg_dim", type=int, default=256)
    p.add_argument("--use_kg", action="store_true", help="Use ConceptNet KG (only for capmeme/capmeme_no_kg)")
    p.add_argument("--bce_weight", type=float, default=1.0)
    p.add_argument("--affect_weight", type=float, default=0.5, help="0 for capmeme_no_emotion ablation")
    p.add_argument("--text_model", default="bert-base-multilingual-cased", help="Text encoder (use ai4bharat/indic-bert for paper if you have HF access)")
    p.add_argument("--gpu", type=int, default=None, help="GPU id (default: use CUDA_VISIBLE_DEVICES or 0)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data: load CSV and optionally filter missing
    import pandas as pd
    df_full = pd.read_csv(args.csv)
    if args.filter_missing:
        df_full = filter_missing_images(df_full, args.image_dir)
        print(f"After filter_missing: {len(df_full)} samples")

    # Splits
    if args.splits_file and os.path.isfile(args.splits_file):
        train_idx, val_idx, test_idx = load_splits(args.splits_file)
        train_df = df_full.iloc[train_idx].reset_index(drop=True)
        val_df = df_full.iloc[val_idx].reset_index(drop=True)
        test_df = df_full.iloc[test_idx].reset_index(drop=True)
        print(f"Loaded splits: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    else:
        train_df, val_df, test_df, train_idx, val_idx, test_idx = stratified_split(
            df_full, stratify_col="Level1",
            train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
            seed=args.seed,
        )
        print(f"Stratified splits: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
        if args.save_splits:
            save_splits(train_idx, val_idx, test_idx, args.save_splits)
            print(f"Saved splits to {args.save_splits}")

    # Build full dataset once to get emotion vocab (from full filtered df)
    full_ds = MemeDataset(
        df=df_full,
        image_dir=args.image_dir,
        text_model_name=args.text_model,
        use_kg=args.use_kg and args.model in ("capmeme", "capmeme_no_kg"),
        kg_dim=args.kg_dim,
    )
    emotion_to_idx = full_ds.emotion_to_idx
    num_emotions = full_ds.num_emotions

    train_ds = MemeDataset(
        df=train_df,
        image_dir=args.image_dir,
        emotion_to_idx=emotion_to_idx,
        num_emotions=num_emotions,
        text_model_name=args.text_model,
        use_kg=args.use_kg and args.model in ("capmeme", "capmeme_no_kg"),
        kg_dim=args.kg_dim,
    )
    val_ds = MemeDataset(
        df=val_df,
        image_dir=args.image_dir,
        emotion_to_idx=emotion_to_idx,
        num_emotions=num_emotions,
        text_model_name=args.text_model,
        use_kg=args.use_kg and args.model in ("capmeme", "capmeme_no_kg"),
        kg_dim=args.kg_dim,
    )
    test_ds = MemeDataset(
        df=test_df,
        image_dir=args.image_dir,
        emotion_to_idx=emotion_to_idx,
        num_emotions=num_emotions,
        text_model_name=args.text_model,
        use_kg=args.use_kg and args.model in ("capmeme", "capmeme_no_kg"),
        kg_dim=args.kg_dim,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # Model
    affect_w = 0.0 if args.model == "capmeme_no_emotion" else args.affect_weight
    model = build_model(
        args.model,
        num_emotions=num_emotions,
        kg_dim=args.kg_dim,
        fusion_hidden=128,
        text_model_name=args.text_model,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_name = args.run_name or f"{args.model}_seed{args.seed}"
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, f"{run_name}_best.pt")
    metrics_path = os.path.join(args.output_dir, f"{run_name}_metrics.json")

    best_val_f1 = 0.0
    for ep in range(args.epochs):
        tr_loss, tr_labels, tr_preds, tr_probs = train_epoch(
            model, train_loader, optimizer, device, bce_w=args.bce_weight, affect_w=affect_w,
        )
        scheduler.step()
        val_loss, val_labels, val_preds, val_probs = evaluate(
            model, val_loader, device, bce_w=args.bce_weight, affect_w=affect_w,
        )
        val_metrics = compute_metrics(val_labels, val_preds, val_probs)
        val_f1 = val_metrics["binary_f1"]
        print(
            f"Epoch {ep+1}/{args.epochs}  train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  val_macro_f1={val_metrics['macro_f1']:.4f}  "
            f"val_binary_f1={val_f1:.4f}"
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), ckpt_path)

    # Final evaluation on test set with best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_loss, test_labels, test_preds, test_probs = evaluate(
        model, test_loader, device, bce_w=args.bce_weight, affect_w=affect_w,
    )
    test_metrics = compute_metrics(test_labels, test_preds, test_probs)
    test_metrics["test_loss"] = test_loss
    test_metrics["model"] = args.model
    test_metrics["seed"] = args.seed
    test_metrics["best_val_f1"] = best_val_f1
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Test metrics -> {metrics_path}")
    print(
        f"  accuracy={test_metrics['accuracy']:.4f}  macro_f1={test_metrics['macro_f1']:.4f}  "
        f"weighted_f1={test_metrics['weighted_f1']:.4f}  binary_f1={test_metrics['binary_f1']:.4f}  "
        f"precision={test_metrics['precision']:.4f}  recall={test_metrics['recall']:.4f}"
    )
    if test_metrics.get("roc_auc") is not None:
        print(f"  roc_auc={test_metrics['roc_auc']:.4f}  pr_auc={test_metrics['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
