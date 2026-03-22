# CAPMeme — Results Tables for Paper

Task: Binary sarcasm detection (Sarcastic vs Non-sarcastic) on EMOFF_MEME Hindi memes. Metrics: F1, A (Accuracy %), P (Precision %), R (Recall %), ROC-AUC. Modality: T = Text, V = Visual. Mean ± std over seeds; best per column in **bold**.

---

## Table 1 — Main results: Models for sarcasm detection

| Model | T | V | F1 | A | P | R | ROC-AUC |
|-------|---|---|-----|-----|-----|-----|--------|
| text only | ✓ | — | 63.87 ± 1.28 | 83.30 ± 0.52 | 66.53 ± 1.77 | 61.53 ± 2.79 | 0.81 ± 0.01 |
| image only | — | ✓ | — | — | — | — | — |
| late fusion | ✓ | ✓ | 61.03 ± 2.67 | 81.65 ± 1.53 | 62.40 ± 3.73 | 59.77 ± 2.15 | 0.80 ± 0.02 |
| capmeme | ✓ | ✓ | 64.41 ± 0.95 | 83.94 ± 0.59 | 68.86 ± 1.92 | 60.53 ± 0.81 | 0.83 ± 0.01 |
| capmeme no kg | ✓ | ✓ | 64.41 ± 0.95 | 83.94 ± 0.59 | 68.86 ± 1.92 | 60.53 ± 0.81 | 0.83 ± 0.01 |
| capmeme no emotion | ✓ | ✓ | 65.07 ± 2.08 | 83.39 ± 1.10 | 65.77 ± 2.55 | 64.41 ± 1.88 | 0.81 ± 0.01 |
| capmeme concat fusion | ✓ | ✓ | **65.98** ± 1.34 | **84.33** ± 0.83 | 69.01 ± 2.66 | 63.28 ± 1.51 | 0.85 ± 0.01 |

## Table 2 — Ablation 1: Effect of components (KG, emotion)

| Model | T | V | F1 | A | P | R |
|-------|---|---|-----|-----|-----|-----|
| capmeme | ✓ | ✓ | 64.41 ± 0.95 | 83.94 ± 0.59 | 68.86 ± 1.92 | 60.53 ± 0.81 |
| capmeme no kg | ✓ | ✓ | 64.41 ± 0.95 | 83.94 ± 0.59 | 68.86 ± 1.92 | 60.53 ± 0.81 |
| capmeme no emotion | ✓ | ✓ | 65.07 ± 2.08 | 83.39 ± 1.10 | 65.77 ± 2.55 | 64.41 ± 1.88 |

## Table 3 — Ablation 2: Fusion strategy

| Model | T | V | F1 | A | P | R |
|-------|---|---|-----|-----|-----|-----|
| capmeme | ✓ | ✓ | 64.41 ± 0.95 | 83.94 ± 0.59 | 68.86 ± 1.92 | 60.53 ± 0.81 |
| capmeme concat fusion | ✓ | ✓ | **65.98** ± 1.34 | 84.33 ± 0.83 | 69.01 ± 2.66 | 63.28 ± 1.51 |

---
Regenerate: `python aggregate_results.py --output_dir outputs --update_tables`