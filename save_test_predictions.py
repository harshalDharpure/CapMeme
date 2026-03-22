"""
Save test-set predictions for each (model, seed) for McNemar / significance tests.
Loads best.pt, runs on test set with same splits, saves outputs/predictions/<model>_seed<seed>.json
Usage:
  python save_test_predictions.py --output_dir outputs --splits_file outputs/splits.json --filter_missing
"""
import argparse
import json
import os
import sys
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from dataset import MemeDataset
from model import build_model, MODEL_NAMES
from data_utils import filter_missing_images, load_splits
from train import collate_fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="EMOFF_MEME.csv")
    p.add_argument("--image_dir", default="my_meme_data/my_meme_data")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--splits_file", default="outputs/splits.json")
    p.add_argument("--filter_missing", action="store_true")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    args = p.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = os.path.join(ROOT, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    pred_dir = os.path.join(output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    import pandas as pd
    df_full = pd.read_csv(os.path.join(ROOT, args.csv))
    if args.filter_missing:
        df_full = filter_missing_images(df_full, os.path.join(ROOT, args.image_dir))
    splits_path = os.path.join(ROOT, args.splits_file) if not os.path.isabs(args.splits_file) else args.splits_file
    train_idx, val_idx, test_idx = load_splits(splits_path)
    test_df = df_full.iloc[test_idx].reset_index(drop=True)

    full_ds = MemeDataset(
        df=df_full,
        image_dir=os.path.join(ROOT, args.image_dir),
        text_model_name="bert-base-multilingual-cased",
        use_kg=False,
    )
    emotion_to_idx = full_ds.emotion_to_idx
    num_emotions = full_ds.num_emotions

    test_ds = MemeDataset(
        df=test_df,
        image_dir=os.path.join(ROOT, args.image_dir),
        emotion_to_idx=emotion_to_idx,
        num_emotions=num_emotions,
        text_model_name="bert-base-multilingual-cased",
        use_kg=False,
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    models = args.models or MODEL_NAMES
    for model_name in models:
        for seed in args.seeds:
            ckpt_path = os.path.join(output_dir, f"{model_name}_seed{seed}_best.pt")
            if not os.path.isfile(ckpt_path):
                print(f"Skip (no ckpt): {model_name} seed {seed}")
                continue
            out_path = os.path.join(pred_dir, f"{model_name}_seed{seed}.json")
            if os.path.isfile(out_path):
                print(f"Skip (exists): {out_path}")
                continue
            model = build_model(
                model_name,
                num_emotions=num_emotions,
                kg_dim=256,
                fusion_hidden=128,
                text_model_name="bert-base-multilingual-cased",
            ).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()

            y_true, y_pred, y_prob = [], [], []
            with torch.no_grad():
                for batch in test_loader:
                    for k in batch:
                        if k != "text_raw" and hasattr(batch[k], "to"):
                            batch[k] = batch[k].to(device)
                    logits, _, _ = model(
                        batch["pixel_values"],
                        batch["input_ids"],
                        batch["attention_mask"],
                        kg_embedding=batch.get("kg_embedding"),
                        kg_valid=batch.get("kg_valid"),
                    )
                    probs = torch.sigmoid(logits).cpu().numpy()
                    preds = (probs > 0.5).astype(int)
                    y_true.extend(batch["sarcasm"].cpu().numpy().astype(int).tolist())
                    y_pred.extend(preds.tolist())
                    y_prob.extend(probs.tolist())

            with open(out_path, "w") as f:
                json.dump({
                    "model": model_name,
                    "seed": seed,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "y_prob": y_prob,
                }, f, indent=2)
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
