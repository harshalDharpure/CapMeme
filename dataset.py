import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer


def _default_emotion_vocab(df, emotion_cols):
    emotions = set()
    for col in emotion_cols:
        if col in df.columns:
            for v in df[col].dropna().unique():
                s = str(v).strip().lower()
                if s and s != "none":
                    emotions.add(s)
    emotion_to_idx = {e: i for i, e in enumerate(sorted(emotions))}
    return emotion_to_idx, len(emotion_to_idx)


class MemeDataset(Dataset):
    def __init__(
        self,
        csv_path=None,
        image_dir=".",
        df=None,
        emotion_to_idx=None,
        num_emotions=None,
        clip_model_name="openai/clip-vit-base-patch32",
        text_model_name="bert-base-multilingual-cased",
        max_text_len=128,
        image_size=224,
        use_kg=False,
        kg_dim=256,
        filter_missing=False,
    ):
        self.image_dir = image_dir
        self.max_text_len = max_text_len
        self.use_kg = use_kg
        self.kg_dim = kg_dim
        self.emotion_cols = ["Level 3(Emotion1)", "Level 4(Emotion2)", "Level 5(Emotion3)"]

        if df is not None:
            self.df = df.reset_index(drop=True)
            if emotion_to_idx is not None and num_emotions is not None:
                self.emotion_to_idx = emotion_to_idx
                self.num_emotions = num_emotions
            else:
                self.emotion_to_idx, self.num_emotions = _default_emotion_vocab(self.df, self.emotion_cols)
        else:
            self.df = pd.read_csv(csv_path)
            if filter_missing:
                from data_utils import filter_missing_images
                self.df = filter_missing_images(self.df, image_dir)
            self.emotion_to_idx, self.num_emotions = _default_emotion_vocab(self.df, self.emotion_cols)

        self.processor = AutoProcessor.from_pretrained(clip_model_name)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.image_size = image_size

    def _build_emotion_vocab(self):
        self.emotion_to_idx, self.num_emotions = _default_emotion_vocab(self.df, self.emotion_cols)

    def _multi_hot_emotions(self, row):
        vec = torch.zeros(self.num_emotions, dtype=torch.float32)
        for col in self.emotion_cols:
            if col not in row or pd.isna(row[col]):
                continue
            s = str(row[col]).strip().lower()
            if s and s != "none" and s in self.emotion_to_idx:
                vec[self.emotion_to_idx[s]] = 1.0
        return vec

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = row["Name"]
        if not str(name).endswith(".png"):
            name = str(name) + ".png"
        image_path = os.path.join(self.image_dir, name)
        pil_image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=pil_image, return_tensors="pt")["pixel_values"].squeeze(0)
        text = str(row["text"]) if pd.notna(row["text"]) else ""
        text_encoded = self.text_tokenizer(
            text,
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        sarcasm = torch.tensor(int(row["Level1"]), dtype=torch.float32)
        emotion_target = self._multi_hot_emotions(row)
        out = {
            "pixel_values": pixel_values,
            "input_ids": text_encoded["input_ids"].squeeze(0),
            "attention_mask": text_encoded["attention_mask"].squeeze(0),
            "sarcasm": sarcasm,
            "emotion_target": emotion_target,
            "text_raw": text,
        }
        if self.use_kg:
            try:
                from kg_extractor import get_kg_embedding
                kg_emb = get_kg_embedding(text, self.kg_dim)
                out["kg_embedding"] = torch.tensor(kg_emb, dtype=torch.float32)
                out["kg_valid"] = torch.tensor(1.0 if np.any(kg_emb != 0) else 0.0, dtype=torch.float32)
            except Exception:
                out["kg_embedding"] = torch.zeros(self.kg_dim, dtype=torch.float32)
                out["kg_valid"] = torch.tensor(0.0, dtype=torch.float32)
        return out
