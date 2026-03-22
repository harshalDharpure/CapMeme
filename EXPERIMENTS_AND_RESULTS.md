# CAPMeme — Models and Experiment Results

This document lists all models, the experiment grid (7 models × 3 seeds), and test-set results where available.  
**Dataset:** EMOFF_MEME (Hindi memes), binary sarcasm (Level1). After `filter_missing`: 7,386 samples; stratified 70/15/15 train/val/test.  
**Seeds:** 42, 123, 456. **Epochs:** 10. **Outputs:** `outputs/`.

---

## 1. Models

| Model | Description |
|-------|-------------|
| **capmeme** | Full model: CLIP vision + BERT text, contrastive affect (\|A_v − A_t\|), emotion supervision (Level 3–5), optional ConceptNet KG. Fusion of [F_contrast; E_kg] → classifier. |
| **capmeme_no_kg** | Same as capmeme but without ConceptNet; E_kg comes only from fallback linear layer on E_t. |
| **capmeme_no_emotion** | CAPMeme with affect weight set to 0 (no emotion supervision); only sarcasm BCE loss. |
| **capmeme_concat_fusion** | Ablation: replace affective contrast with simple concatenation [E_v; E_t] (or similar) before fusion. |
| **text_only** | BERT text encoder only; single modality baseline. |
| **image_only** | CLIP vision encoder only; single modality baseline. |
| **late_fusion** | Separate BERT and CLIP classifiers; fuse logits (e.g. average or concat) for final prediction. |

---

## 2. Experiment Grid and Status

| # | Model | Seed | Status | Test Accuracy | Binary F1 | Macro F1 | ROC-AUC | PR-AUC | Best Val F1 |
|---|-------|------|--------|---------------|-----------|----------|---------|--------|-------------|
| 1 | capmeme | 42 | ✅ Completed | 84.66% | 0.653 | 0.777 | 0.824 | 0.656 | 0.671 |
| 2 | capmeme | 123 | ✅ Completed | 83.94% | 0.648 | 0.772 | 0.827 | 0.654 | 0.671 |
| 3 | capmeme | 456 | ⚠️ Checkpoint only | — | — | — | — | — | — |
| 4 | capmeme_no_kg | 42 | ❌ Not run | — | — | — | — | — | — |
| 5 | capmeme_no_kg | 123 | ❌ Not run | — | — | — | — | — | — |
| 6 | capmeme_no_kg | 456 | ❌ Not run | — | — | — | — | — | — |
| 7 | capmeme_no_emotion | 42 | ❌ Not run | — | — | — | — | — | — |
| 8 | capmeme_no_emotion | 123 | ❌ Not run | — | — | — | — | — | — |
| 9 | capmeme_no_emotion | 456 | ❌ Not run | — | — | — | — | — | — |
| 10 | capmeme_concat_fusion | 42 | ❌ Not run | — | — | — | — | — | — |
| 11 | capmeme_concat_fusion | 123 | ❌ Not run | — | — | — | — | — | — |
| 12 | capmeme_concat_fusion | 456 | ❌ Not run | — | — | — | — | — | — |
| 13 | text_only | 42 | ❌ Not run | — | — | — | — | — | — |
| 14 | text_only | 123 | ❌ Not run | — | — | — | — | — | — |
| 15 | text_only | 456 | ❌ Not run | — | — | — | — | — | — |
| 16 | image_only | 42 | ❌ Not run | — | — | — | — | — | — |
| 17 | image_only | 123 | ❌ Not run | — | — | — | — | — | — |
| 18 | image_only | 456 | ❌ Not run | — | — | — | — | — | — |
| 19 | late_fusion | 42 | ❌ Not run | — | — | — | — | — | — |
| 20 | late_fusion | 123 | ❌ Not run | — | — | — | — | — | — |
| 21 | late_fusion | 456 | ❌ Not run | — | — | — | — | — | — |

**Legend:**  
- **✅ Completed** — `*_best.pt` and `*_metrics.json` present; metrics in table from test set.  
- **⚠️ Checkpoint only** — `*_best.pt` saved but no `*_metrics.json` (run interrupted before metrics write).  
- **❌ Not run** — No outputs for this (model, seed) in this pipeline run.

---

## 3. Detailed Results (Completed Runs)

### 3.1 capmeme — seed 42

| Metric | Value |
|--------|--------|
| Accuracy | 84.66% |
| Binary F1 | 0.653 |
| Macro F1 | 0.777 |
| Weighted F1 | 0.842 |
| Precision | 0.714 |
| Recall | 0.602 |
| ROC-AUC | 0.824 |
| PR-AUC | 0.656 |
| Best val F1 | 0.671 |
| Test loss | 0.846 |

**Confusion matrix (test):** TN=778, FP=64, FN=106, TP=160.

### 3.2 capmeme — seed 123

| Metric | Value |
|--------|--------|
| Accuracy | 83.94% |
| Binary F1 | 0.648 |
| Macro F1 | 0.772 |
| Weighted F1 | 0.836 |
| Precision | 0.683 |
| Recall | 0.617 |
| ROC-AUC | 0.827 |
| PR-AUC | 0.654 |
| Best val F1 | 0.671 |
| Test loss | 1.019 |

**Confusion matrix (test):** TN=766, FP=76, FN=102, TP=164.

---

## 4. Summary

- **Total experiments:** 21 (7 models × 3 seeds).  
- **Fully completed:** 2 (capmeme seed 42, capmeme seed 123).  
- **Checkpoint only:** 1 (capmeme seed 456).  
- **Not run yet:** 18.

Results are read from `outputs/<model>_seed<seed>_metrics.json`. When more runs complete, re-run the pipeline or update this file from the new metrics and `run_all_summary.json`.
