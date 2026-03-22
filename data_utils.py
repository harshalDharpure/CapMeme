"""
Data utilities for CAPMeme research pipeline.
- Filter rows with missing images.
- Stratified train/val/test split by Level1 (sarcasm).
- Save/load split indices for reproducibility.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def filter_missing_images(df: pd.DataFrame, image_dir: str, name_col: str = "Name") -> pd.DataFrame:
    """Drop rows whose image file is missing. Returns filtered DataFrame (copy)."""
    def image_path(row):
        name = row[name_col]
        if not str(name).strip().endswith(".png"):
            name = str(name).strip() + ".png"
        return os.path.join(image_dir, name)

    mask = df.apply(lambda row: os.path.isfile(image_path(row)), axis=1)
    return df.loc[mask].reset_index(drop=True)


def stratified_split(
    df: pd.DataFrame,
    stratify_col: str = "Level1",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple:
    """
    Returns (train_df, val_df, test_df, train_idx, val_idx, test_idx).
    Splits are stratified by stratify_col (e.g. Level1 for sarcasm).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    y = df[stratify_col].values
    idx = np.arange(len(df))

    train_idx, rest_idx = train_test_split(idx, test_size=(1 - train_ratio), stratify=y, random_state=seed)
    rest_ratio = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        rest_idx,
        test_size=(1 - rest_ratio),
        stratify=y[rest_idx],
        random_state=seed,
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df, train_idx, val_idx, test_idx


def save_splits(train_idx, val_idx, test_idx, filepath: str):
    """Save split indices to JSON for reproducibility."""
    data = {
        "train_idx": [int(i) for i in train_idx],
        "val_idx": [int(i) for i in val_idx],
        "test_idx": [int(i) for i in test_idx],
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_splits(filepath: str) -> tuple:
    """Load split indices from JSON. Returns (train_idx, val_idx, test_idx) as lists."""
    with open(filepath) as f:
        data = json.load(f)
    return data["train_idx"], data["val_idx"], data["test_idx"]
