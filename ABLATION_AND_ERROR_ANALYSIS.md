# Ablation Analysis and Error Analysis (Journal Supplement)

## 1. Ablation Analysis

### 1.1 Component ablations (Table 2)

- **CAPMeme (full):** Affective contrast |A_v − A_t| + optional ConceptNet KG + emotion supervision (Level 3–5). This is the proposed model.
- **CAPMeme w/o KG:** Same architecture but KG is replaced by a fallback linear layer on text embeddings. Measures the contribution of external knowledge (ConceptNet).
- **CAPMeme w/o emotion:** Same as full but affect weight = 0 (no auxiliary emotion loss). Measures the benefit of multi-hot emotion supervision.

*Interpretation (fill after all runs):* Compare F1 and Accuracy across the three rows. If full > w/o KG, then KG helps; if full > w/o emotion, then emotion supervision helps. Report mean ± std over seeds and note whether differences are statistically significant (McNemar test between model pairs on the same test set).

### 1.2 Fusion ablation (Table 3)

- **CAPMeme (affective contrast):** Fusion input is [F_contrast; E_kg] with F_contrast = |A_v − A_t|.
- **CAPMeme (concat fusion):** Fusion input is [E_v; E_t; E_kg] (concatenation of raw modalities + KG). No contrastive affect.

*Interpretation (fill after all runs):* If affective contrast outperforms concat fusion, it supports the hypothesis that modelling affective incongruity is better than simple concatenation for sarcasm detection.

### 1.3 Statistical significance

After saving test predictions for all models (`python save_test_predictions.py --output_dir outputs --splits_file outputs/splits.json --filter_missing`), run McNemar tests between the proposed model (e.g. capmeme) and each baseline/ablation:

```bash
python mcnemar_significance.py --pred_dir outputs/predictions --model_a capmeme --model_b text_only
python mcnemar_significance.py --pred_dir outputs/predictions --model_a capmeme --model_b image_only
python mcnemar_significance.py --pred_dir outputs/predictions --model_a capmeme --model_b late_fusion
python mcnemar_significance.py --pred_dir outputs/predictions --model_a capmeme --model_b capmeme_no_kg
python mcnemar_significance.py --pred_dir outputs/predictions --model_a capmeme --model_b capmeme_no_emotion
python mcnemar_significance.py --pred_dir outputs/predictions --model_a capmeme --model_b capmeme_concat_fusion
```

Report p-values in the paper (e.g. "CAPMeme significantly outperforms text-only (McNemar, p < 0.05)").

---

## 2. Error analysis

### 2.1 Confusion matrix (from completed runs)

For **capmeme** (seeds 42 and 123), test-set confusion:

| Seed | TN | FP | FN | TP |
|------|----|----|----|-----|
| 42   | 778| 64 | 106| 160 |
| 123  | 766| 76 | 102| 164 |

- **Observation:** False negatives (sarcastic memes predicted as non-sarcastic) are relatively high (~102–106), suggesting the model is somewhat conservative on the positive class. Precision is higher than recall for seed 42; more balanced for seed 123.
- **Class balance:** Test set has 1108 samples; positive rate ~24% (TP+FN). Macro F1 accounts for both classes.

### 2.2 Example predictions (optional for paper)

To generate a short table of success/failure cases:

1. After all runs, pick the best model (e.g. capmeme, seed 42).
2. Load test set and model; run inference; for each sample store: text (or image name), true label, pred, prob.
3. Select a few **true positives**, **true negatives**, **false positives**, **false negatives** (e.g. 2–3 each) and add them to the paper as a qualitative analysis table.

*Placeholder:* "We leave detailed qualitative analysis and example memes for the camera-ready version."

---

## 3. Checklist for submission

- [ ] All 21 runs completed (7 models × 3 seeds).
- [ ] `aggregate_results.py --update_tables` run; Tables 1–3 filled with mean ± std; best rows bolded.
- [ ] Test predictions saved for all (model, seed) with `save_test_predictions.py`.
- [ ] McNemar tests run for CAPMeme vs each baseline and ablation; p-values reported in text or table.
- [ ] Ablation discussion: contribution of KG, emotion supervision, and fusion strategy.
- [ ] Error analysis: confusion matrix summary and (optional) example success/failure cases.
