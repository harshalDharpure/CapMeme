# CAPMeme — Project Documentation

## 1. Project Overview

**CAPMeme** (Contrastive Affect Modelling Meme) is a **multimodal sarcasm detection framework** for **Hindi memes**. It predicts whether a meme is sarcastic (binary) by modelling **affective incongruity** between the image and the text, supervised by **hierarchical emotion labels** and optionally enriched with **Knowledge Graph (ConceptNet)** representations.

---

## 2. Objective

- **Task:** Binary classification — Sarcastic vs Non-Sarcastic (Level1 label).
- **Approach:** Use **visual** (CLIP) and **textual** (IndicBERT) embeddings, map them to an **emotion space**, compute **contrast** between modalities, optionally add **KG embeddings**, then fuse and classify.
- **Supervision:** Final head is supervised by Level1 (sarcasm). The **CAP (Contrastive Affect Pair)** module is supervised by **multi-hot emotion** vectors from Level 3–5 (ignoring `"none"`).

---

## 3. Architecture Summary

```
[Image] → CLIP Vision → E_v → vis_affect → A_v  ─┐
                                                    ├→ F_contrast = |A_v - A_t| ─┐
[Text]  → IndicBERT  → E_t → text_affect → A_t  ─┘                              ├→ concat → Fusion(ReLU) → Classifier → Sigmoid → Sarcasm
                                                                                  │
[Text]  → ConceptNet (or fallback) → E_kg ────────────────────────────────────────┘
```

---

## 4. Inputs & Modalities

| Modality | Input | Encoder | Output |
|----------|--------|---------|--------|
| **Visual (V)** | Meme image (RGB) | CLIP Vision (`openai/clip-vit-base-patch32`) | E_v ∈ ℝ^768 |
| **Text (T)** | Hindi meme text | IndicBERT (`ai4bharat/indic-bert`) | E_t ∈ ℝ^768 |

- **E_v:** CLIP vision encoder output (pooler or `last_hidden_state[:, 0]`).
- **E_t:** IndicBERT `[CLS]` token representation (`last_hidden_state[:, 0]`).

---

## 5. CAP Module (Contrastive Affect Pair)

- **Purpose:** Capture affective incongruity between image and text.
- **Layers:**
  - `vis_affect`: Linear(E_v → num_emotions) → **A_v**
  - `text_affect`: Linear(E_t → num_emotions) → **A_t**
- **Affective contrast feature:**  
  **F_contrast = |A_v − A_t|** (element-wise absolute difference).
- **Supervision:** Multi-hot emotion vector from Level 3–5 (e.g. Envy, Rage, Joy); `"none"` is ignored. The loss encourages A_v and A_t to align with this emotion target (MSE), so F_contrast reflects where image and text emotions differ.

---

## 6. Knowledge Graph (KG) Integration

- **Source:** ConceptNet API (`https://api.conceptnet.io`), language code `hi` (Hindi).
- **Process:**
  1. Hindi text is tokenized; up to 10 tokens are used.
  2. For each token, query `GET /c/hi/{normalized_token}?limit=20`.
  3. From response edges, collect concept labels (`start.label`, `end.label`).
  4. Convert label list to a fixed-size vector: hash-based random vectors per label, averaged and L2-normalized → **E_kg** ∈ ℝ^kg_dim (default 256).
- **Fallback:** If the API fails or returns no concepts, the **model** uses a **fallback dense layer**: `E_kg = Linear(E_t → kg_dim)`. The dataset can pass precomputed `kg_embedding` and `kg_valid`; when `kg_valid == 0`, the model uses the fallback.

---

## 7. Fusion & Classification

- **Concatenation:** `[F_contrast; E_kg]` → dimension `num_emotions + kg_dim`.
- **Fusion layer:** Linear → ReLU → hidden dimension (default 128).
- **Classification head:** Linear(hidden → 1) → **logits**.
- **Prediction:** Sarcasm probability = Sigmoid(logits); binary label = (probability > 0.5).

---

## 8. Dataset Specifications

### 8.1 Data Layout

- **Images:** PNG files in a folder (e.g. `hindi_meme0.png`).
- **CSV:** One row per meme with columns:

| Column | Description |
|--------|-------------|
| `Name` | Image filename (e.g. `hindi_meme0.png`) |
| `text` | Hindi meme text |
| `Level1` | Binary sarcasm label (0/1) — **classification target** |
| `Level2` | Not used |
| `Level 3(Emotion1)` | Emotion 1 (e.g. Envy, Rage, Joy, none) |
| `Level 4(Emotion2)` | Emotion 2 |
| `Level 5(Emotion3)` | Emotion 3 |

### 8.2 Emotion Multi-Hot

- Emotion vocab is built from all non-null, non-`"none"` values in Level 3–5 (lowercased).
- Each sample gets a **multi-hot vector** of length `num_emotions`; positions corresponding to the three emotion labels are set to 1.

### 8.3 Optional KG at Dataset Time

- If `use_kg=True`, the dataset calls `get_kg_embedding(text, kg_dim)` per sample and returns `kg_embedding` and `kg_valid` (1.0 if any concepts found, else 0.0). This can slow down data loading due to API calls.

---

## 9. File Structure & Module Roles

| File | Role |
|------|------|
| **dataset.py** | PyTorch `MemeDataset`: loads CSV + images, CLIP processor for `pixel_values`, IndicBERT tokenizer for `input_ids`/`attention_mask`, builds emotion vocab and multi-hot `emotion_target`, optional KG embedding. |
| **kg_extractor.py** | ConceptNet helpers: `fetch_concept_labels(hindi_text)`, `concept_labels_to_embedding(labels, embed_dim)`, `get_kg_embedding(hindi_text, embed_dim)`. |
| **model.py** | `KGModule` (ConceptNet E_kg or fallback from E_t); **CAPMeme** (CLIP vision, IndicBERT, CAP, KG, fusion, classifier). Forward returns `(logits, A_v, A_t)`. |
| **loss.py** | `joint_capmeme_loss`: BCEWithLogitsLoss on logits vs sarcasm target + MSE(A_v, emotion_target) + MSE(A_t, emotion_target). Returns total loss and optional bce/affect terms. |
| **train.py** | CLI, train/val split, DataLoader, AdamW, CosineAnnealingLR, train/eval loops, Accuracy & F1, gradient clipping, best-model save by val F1. |
| **requirements.txt** | Python dependencies (torch, transformers, pandas, etc.). |

---

## 10. Loss Function

- **L = λ₁ · L_BCE + λ₂ · L_affect**
  - **L_BCE:** Binary cross-entropy with logits for sarcasm (Level1).
  - **L_affect:** MSE(A_v, emotion_target) + MSE(A_t, emotion_target).
- Default: `bce_weight=1.0`, `affect_weight=0.5` (configurable in training if you extend the script).

---

## 11. Training Pipeline

- **Optimizer:** AdamW.
- **Scheduler:** CosineAnnealingLR over epochs.
- **Metrics:** Train/Val Loss, Accuracy, F1 (binary).
- **Split:** Random train/val (e.g. 85% / 15%), seed fixed for reproducibility.
- **Best model:** Saved when validation F1 improves (`capmeme_best.pt`).
- **Gradient clipping:** max_norm=1.0.

---

## 12. How to Run

### 12.1 Install Dependencies

```bash
pip install -r requirements.txt
```

### 12.2 Train (without KG)

```bash
python train.py --csv EMOFF_MEME.csv --image_dir /path/to/png/folder
```

### 12.3 Train (with KG; slower)

```bash
python train.py --csv EMOFF_MEME.csv --image_dir /path/to/png/folder --use_kg
```

### 12.4 CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv` | `EMOFF_MEME.csv` | Path to dataset CSV. |
| `--image_dir` | `.` | Directory containing PNGs (names must match `Name` in CSV). |
| `--val_ratio` | 0.15 | Fraction of data for validation. |
| `--batch_size` | 16 | Batch size. |
| `--epochs` | 10 | Number of epochs. |
| `--lr` | 2e-5 | Learning rate. |
| `--kg_dim` | 256 | KG embedding dimension. |
| `--use_kg` | False | Whether to compute KG embeddings in the dataset. |
| `--seed` | 42 | Random seed. |

---

## 13. Important Design Choices

- **CLIP for vision:** Pretrained vision encoder; no separate image transform beyond CLIP processor.
- **IndicBERT for Hindi:** Language-specific encoder for the text modality.
- **Emotion space:** Dimension = number of distinct emotions in the dataset (Level 3–5, excluding `"none"`).
- **KG optional:** Training works without KG; with `--use_kg`, ConceptNet enriches the fusion input when the API returns concepts; otherwise the fallback layer is used.
- **Single GPU/CPU:** Script uses `cuda` if available, else `cpu`; no distributed logic.

---

## 14. Output Artifacts

- **capmeme_best.pt:** Best model state dict (by validation F1). Load with `model.load_state_dict(torch.load("capmeme_best.pt"))` after building the same `CAPMeme` (same `num_emotions`, `kg_dim`, etc.).

---

## 15. Dependencies (Summary)

- **torch**, **torchvision** — model and data.
- **transformers** — CLIP Vision, IndicBERT, tokenizer/processor.
- **pandas** — CSV and emotion vocab.
- **Pillow** — image loading.
- **scikit-learn** — Accuracy, F1.
- **requests** — ConceptNet API.
- **numpy** — KG embedding construction.

---

This document describes the full CAPMeme pipeline: data, encoders, CAP module, KG, fusion, loss, and training. For implementation details, refer to the corresponding Python files listed above.
