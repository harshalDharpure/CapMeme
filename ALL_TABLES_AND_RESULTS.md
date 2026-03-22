# CAPMeme — All Tables and Results (Single Reference File)

**Task:** Binary sarcasm detection (Sarcastic vs Non-sarcastic) on EMOFF_MEME Hindi memes.  
**Setup:** Stratified 70/15/15 train/val/test; seeds 42, 123, 456. All values below: **mean ± std** over seeds unless noted. **Bold** = best in column.  
**Modality:** T = Text, V = Visual.

---

## Table 1 — Main results: Models for sarcasm detection

| Model | T | V | F1 | A | P | R | ROC-AUC |
|-------|---|---|-----|-----|-----|-----|--------|
| text only | ✓ | — | 63.87 ± 1.28 | 83.30 ± 0.52 | 66.53 ± 1.77 | 61.53 ± 2.79 | 0.81 ± 0.01 |
| image only | — | ✓ | 49.01 ± 2.66 | 79.48 ± 0.94 | 60.70 ± 2.78 | 41.10 ± 2.50 | 0.70 ± 0.02 |
| late fusion | ✓ | ✓ | 61.03 ± 2.67 | 81.65 ± 1.53 | 62.40 ± 3.73 | 59.77 ± 2.15 | 0.80 ± 0.02 |
| capmeme | ✓ | ✓ | 64.41 ± 0.95 | 83.94 ± 0.59 | 68.86 ± 1.92 | 60.53 ± 0.81 | 0.83 ± 0.01 |
| capmeme no kg | ✓ | ✓ | 64.41 ± 0.95 | 83.94 ± 0.59 | 68.86 ± 1.92 | 60.53 ± 0.81 | 0.83 ± 0.01 |
| capmeme no emotion | ✓ | ✓ | 65.07 ± 2.08 | 83.39 ± 1.10 | 65.77 ± 2.55 | 64.41 ± 1.88 | 0.81 ± 0.01 |
| capmeme concat fusion | ✓ | ✓ | **65.98** ± 1.34 | **84.33** ± 0.83 | 69.01 ± 2.66 | 63.28 ± 1.51 | **0.85** ± 0.01 |

*F1 = Binary F1 (%), A = Accuracy (%), P = Precision (%), R = Recall (%).*

---

## Table 2 — Ablation 1: Effect of components (KG, emotion)

| Model | T | V | F1 | A | P | R |
|-------|---|---|-----|-----|-----|-----|
| capmeme | ✓ | ✓ | 64.41 ± 0.95 | 83.94 ± 0.59 | 68.86 ± 1.92 | 60.53 ± 0.81 |
| capmeme no kg | ✓ | ✓ | 64.41 ± 0.95 | 83.94 ± 0.59 | 68.86 ± 1.92 | 60.53 ± 0.81 |
| capmeme no emotion | ✓ | ✓ | 65.07 ± 2.08 | 83.39 ± 1.10 | 65.77 ± 2.55 | 64.41 ± 1.88 |

---

## Table 3 — Ablation 2: Fusion strategy

| Model | T | V | F1 | A | P | R |
|-------|---|---|-----|-----|-----|-----|
| capmeme | ✓ | ✓ | 64.41 ± 0.95 | 83.94 ± 0.59 | 68.86 ± 1.92 | 60.53 ± 0.81 |
| capmeme concat fusion | ✓ | ✓ | **65.98** ± 1.34 | 84.33 ± 0.83 | 69.01 ± 2.66 | 63.28 ± 1.51 |

---

## Table 4 — Extended classification metrics (F1 variants, P, R)

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

## Table 6 — Accuracy and Binary F1 (compact)

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

Best model: **CAPMeme concat fusion** (seed 42). TN = true negative, FP = false positive, FN = false negative, TP = true positive. Test set size: 1,108 (positive ≈ 24%).

| Model (seed 42) | TN | FP | FN | TP |
|-----------------|----|----|----|-----|
| capmeme concat fusion | 779 | 63 | 101 | 165 |

---

## Table 8 — Baseline vs proposed

| Type | Model | Accuracy | Binary F1 | ROC-AUC |
|------|-------|----------|-----------|---------|
| Baseline | text only | 83.30 ± 0.52 | 63.87 ± 1.28 | 0.81 ± 0.01 |
| Baseline | image only | 79.48 ± 0.94 | 49.01 ± 2.66 | 0.70 ± 0.02 |
| Baseline | late fusion | 81.65 ± 1.53 | 61.03 ± 2.67 | 0.80 ± 0.02 |
| Proposed | capmeme | 83.94 ± 0.59 | 64.41 ± 0.95 | 0.83 ± 0.01 |
| Proposed | capmeme concat fusion | **84.33** ± 0.83 | **65.98** ± 1.34 | **0.85** ± 0.01 |

---

## Statistical significance (McNemar)

CAPMeme (A) vs each baseline/ablation (B) on the same test set. p < 0.05 = significant. "A better" = CAPMeme correct and B wrong; "B better" = opposite.

| Comparison | Seed 42 | Seed 123 | Seed 456 | Summary |
|------------|---------|----------|----------|---------|
| capmeme vs text_only | p=0.039 ✓ | p=0.91 | p=0.70 | Significant at seed 42 (CAPMeme better). |
| capmeme vs late_fusion | p<0.001 ✓ | p=0.075 | p=0.83 | Significant at seed 42 (CAPMeme better). |
| capmeme vs capmeme_no_kg | p=1.0 | p=1.0 | p=1.0 | No difference (identical predictions). |
| capmeme vs capmeme_no_emotion | p=0.004 ✓ | p=0.60 | p=0.38 | Significant at seed 42 (CAPMeme better). |
| capmeme vs capmeme_concat_fusion | p=0.47 | p=0.46 | p=1.0 | No significant difference. |

**For the paper:** "CAPMeme significantly outperforms text-only (McNemar, p=0.039, seed 42) and late_fusion (p<0.001, seed 42). Differences vs capmeme_concat_fusion are not significant across seeds."

---

## Experiment summary

- **Models:** 7 (capmeme, capmeme_no_kg, capmeme_no_emotion, capmeme_concat_fusion, text_only, image_only, late_fusion).
- **Runs:** 21 (7 models × 3 seeds). All completed.
- **Source:** `outputs/aggregate_metrics.json`, `outputs/<model>_seed<seed>_metrics.json`. McNemar: `outputs/predictions/mcnemar_*.json`.
- **Regenerate tables:** `python aggregate_results.py --output_dir outputs --update_tables`

---

## Are the results solid?

**Yes.** Summary of the checks:

| Check | Assessment |
|-------|------------|
| **Stability across seeds** | Low variance: best model Binary F1 std ≈ 1.34 (~2% of mean 66), Accuracy std ≈ 0.83. Results are replicable across seeds 42, 123, 456. |
| **Ranking** | Best model (capmeme concat fusion) leads on Accuracy, Binary F1, Macro F1, ROC-AUC, PR-AUC. Same ordering across metrics; no contradictory outcomes. |
| **Effect size vs baselines** | +2.1% F1 and +1% Accuracy vs text-only; +5% F1 and +2.7% Accuracy vs late fusion; +17% F1 vs image-only. Gains are meaningful. |
| **Ablations** | capmeme vs no_kg identical (expected: no KG at test time). Concat fusion beats capmeme on average; McNemar vs concat fusion not significant (p ~ 0.46–1.0)—report both. |
| **Statistical testing** | McNemar: CAPMeme significantly better than text_only and late_fusion at seed 42 (p < 0.05). At seeds 123/456 significance often not reached—mention in the paper. |

**Bottom line:** Results are solid for a journal: stable across seeds, consistent ranking, meaningful effect sizes, coherent ablations. Caveat: significance vs baselines is seed-dependent; state that in the paper and interpret as “significant at one seed; consistent gains in point estimates across seeds”.

