# Additional Tables for Journal Paper — CAPMeme

All values: **mean ± std** over seeds 42, 123, 456 on the held-out test set. Task: binary sarcasm detection (Sarcastic vs Non-sarcastic) on EMOFF_MEME Hindi memes. **Bold** = best in column.

---

## Table 4 — Extended classification metrics (F1 variants, Precision, Recall)

**Table 4.** Binary F1, Macro F1, Weighted F1, Precision (%), and Recall (%) for each model.

| Model | Binary F1 | Macro F1 | Weighted F1 | Precision | Recall |
|-------|-----------|----------|-------------|-----------|--------|
| text only | 63.87 ± 1.28 | 76.50 ± 0.73 | 83.07 ± 0.49 | 66.53 ± 1.77 | 61.53 ± 2.79 |
| image only | 49.01 ± 2.66 | 68.08 ± 1.61 | 78.00 ± 1.07 | 60.70 ± 2.78 | 41.10 ± 2.50 |
| late fusion | 61.03 ± 2.67 | 74.51 ± 1.86 | 81.52 ± 1.44 | 62.40 ± 3.73 | 59.77 ± 2.15 |
| capmeme | 64.41 ± 0.95 | 77.02 ± 0.67 | 83.57 ± 0.53 | 68.86 ± 1.92 | 60.53 ± 0.81 |
| capmeme no kg | 64.41 ± 0.95 | 77.02 ± 0.67 | 83.57 ± 0.53 | 68.86 ± 1.92 | 60.53 ± 0.81 |
| capmeme no emotion | 65.07 ± 2.08 | 77.09 ± 1.41 | 83.34 ± 1.06 | 65.77 ± 2.55 | **64.41** ± 1.88 |
| capmeme concat fusion | **65.98** ± 1.34 | **77.90** ± 0.94 | **84.09** ± 0.75 | **69.01** ± 2.66 | 63.28 ± 1.51 |

---

## Table 5 — AUC metrics (ROC-AUC and PR-AUC)

**Table 5.** Area under the ROC curve (ROC-AUC) and Precision–Recall curve (PR-AUC). Higher is better.

| Model | ROC-AUC | PR-AUC |
|-------|---------|--------|
| text only | 0.81 ± 0.01 | 0.64 ± 0.03 |
| image only | 0.70 ± 0.02 | 0.53 ± 0.02 |
| late fusion | 0.80 ± 0.02 | 0.63 ± 0.03 |
| capmeme | 0.83 ± 0.01 | 0.66 ± 0.01 |
| capmeme no kg | 0.83 ± 0.01 | 0.66 ± 0.01 |
| capmeme no emotion | 0.81 ± 0.01 | 0.63 ± 0.03 |
| capmeme concat fusion | **0.85** ± 0.01 | **0.71** ± 0.02 |

---

## Table 6 — Accuracy and F1 summary (compact for main text)

**Table 6.** Test accuracy (%) and binary F1 (%) — compact form for inline or main results.

| Model | Accuracy (%) | Binary F1 (%) |
|-------|--------------|---------------|
| text only | 83.30 ± 0.52 | 63.87 ± 1.28 |
| image only | 79.48 ± 0.94 | 49.01 ± 2.66 |
| late fusion | 81.65 ± 1.53 | 61.03 ± 2.67 |
| capmeme | 83.94 ± 0.59 | 64.41 ± 0.95 |
| capmeme no kg | 83.94 ± 0.59 | 64.41 ± 0.95 |
| capmeme no emotion | 83.39 ± 1.10 | 65.07 ± 2.08 |
| capmeme concat fusion | **84.33** ± 0.83 | **65.98** ± 1.34 |

---

## Table 7 — Confusion matrix (best model, seed 42)

**Table 7.** Confusion matrix for the best-performing model (CAPMeme concat fusion, seed 42) on the test set. TN = true negative, FP = false positive, FN = false negative, TP = true positive.

| Model (seed 42) | TN | FP | FN | TP |
|-----------------|----|----|----|-----|
| capmeme concat fusion | 779 | 63 | 101 | 165 |

*Test set size: 1,108. Positive class (sarcastic) ≈ 24%.*

---

## Table 8 — Baseline vs proposed (side-by-side)

**Table 8.** Baseline (single-modality and late fusion) vs proposed multimodal models.

| Type | Model | Accuracy | Binary F1 | ROC-AUC |
|------|-------|----------|-----------|---------|
| Baseline | text only | 83.30 ± 0.52 | 63.87 ± 1.28 | 0.81 ± 0.01 |
| Baseline | image only | 79.48 ± 0.94 | 49.01 ± 2.66 | 0.70 ± 0.02 |
| Baseline | late fusion | 81.65 ± 1.53 | 61.03 ± 2.67 | 0.80 ± 0.02 |
| Proposed | capmeme | 83.94 ± 0.59 | 64.41 ± 0.95 | 0.83 ± 0.01 |
| Proposed | capmeme concat fusion | **84.33** ± 0.83 | **65.98** ± 1.34 | **0.85** ± 0.01 |

---

*Source: `outputs/aggregate_metrics.json`. Regenerate tables from metrics with `python aggregate_results.py --output_dir outputs --update_tables`.*
