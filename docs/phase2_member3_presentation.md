# Member 3 — D-307 final presentation and demonstration

## Slide 1 — Contribution and amendment

- Built an embedding MLP using 49 numeric and 10 categorical features.
- Registered D-307 before amended training to cover dropout and batch size.
- Evaluated the full 48-setting joint grid with seed 42.
- Reused eight verified D-305 runs and trained 40 new settings.

Speaker notes

Sep–Oct was scored once under D-305 (D-306) before this amendment. D-307 replaces D-306 as the Member 3 result because D-305 did not tune batch size and dropout as assigned. Sep–Oct is not used for any choice.

## Slide 2 — Architecture

Use `reports/figures/phase2_member3/d307/embedding_architecture.png`.

- Selected learning rate 0.0001, hidden layers [128, 64], embedding cap 4, dropout 0.2, batch size 1024.
- Concatenate numeric values with trainable categorical embeddings.
- Apply two ReLU hidden layers, dropout, and one output logit.
- Use sigmoid to produce a probability at inference.
- The refit model contains 23,315 parameters.
- Fit vocabulary, imputation and scalers on each stage's training rows only.

## Slide 3 — Joint tuning

Use `reports/figures/phase2_member3/d307/august_grid.png`.

- Learning rate 0.0001/0.0003, hidden layers 64–32/128–64, embedding cap 4/8.
- Dropout 0.1/0.2/0.4 and batch size 256/1024.
- Fit January–July and choose every setting and epoch by August PR-AUC.
- Winner August PR-AUC 0.39747, seed-42 epoch 4.
- Exact ties prefer fewer parameters, lower learning rate, then grid order.
- Adam, weight decay 0.0001, unweighted BCE, maximum 30 epochs, patience 5.

## Slide 4 — Five seeds and final results

- Seeds 42–46 each choose their own epoch on August.
- Reinitialise and refit January–August for the chosen epoch count.
- Freeze every model before the D-307 September–October scoring.
- Use the same 55,628 flights and saved MLP, LR and GRU baselines.

| model | pr_auc_mean | pr_auc_std | roc_auc_mean | brier_score_mean | ece_10_mean |
| --- | --- | --- | --- | --- | --- |
| embedding | 0.27572 | 0.00347 | 0.67483 | 0.12826 | 0.05034 |
| gru | 0.29457 | 0.00552 | 0.67434 | 0.136 | 0.08628 |
| lr | 0.28959 | — | 0.68 | 0.18554 | 0.23341 |
| mlp | 0.28393 | 0.00429 | 0.67187 | 0.13327 | 0.07574 |

## Slide 5 — Historical result next to final result

| result | dropout | batch_size | August tuning PR-AUC | pr_auc | roc_auc | brier_score | ece_10 | pr_auc_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D-306 historical | 0.2 | 256 | 0.39683 | 0.2762 | 0.67528 | 0.12793 | 0.04732 | 0.00242 |
| D-307 final | 0.2 | 1024 | 0.39747 | 0.27572 | 0.67483 | 0.12826 | 0.05034 | 0.00347 |

Speaker notes

D-307 is the final Member 3 result because it completes the assigned joint grid.
The historical D-306 results remain available for review. Final evaluation values
were observed after architecture and epoch choices were frozen.

## Slide 6 — Paired uncertainty and interpretation

Use `reports/figures/phase2_member3/d307/bootstrap_differences.png`.

| comparison | difference | ci_low | ci_high | excludes_zero |
| --- | --- | --- | --- | --- |
| embedding - mlp | -0.00821 | -0.02668 | 0.01158 | False |
| embedding - lr | -0.01387 | -0.02181 | -0.0034 | True |
| embedding - gru | -0.01885 | -0.03969 | 0.00226 | False |

No comparison has a positive 95% interval entirely above zero. Negative intervals support lower PR-AUC for embedding - lr.

The existing Member 2 bootstrap samples whole days 1,000 times, averages PR-AUC
over the five seeds, and computes paired differences on identical sampled flights.
PR-AUC evaluates ranking. Brier and ECE evaluate raw probability quality. Earlier
evaluation exposure is disclosed. One year of NYC data limits generalisation.

## Demonstration

1. Show D-307 and its pre-training commit `90eecd5`.
2. Show all 48 settings and provenance of the eight reused runs.
3. Show the five saved models under `models/phase2/member3/refit_d307/`.
4. Open the executed Member 3 notebook and the historical-versus-final table.
5. Explain the day-bootstrap intervals and the role of August-only selection.

## Viva preparation

- An embedding is a category-specific vector learned jointly with the classifier.
- Dropout randomly removes hidden activations during training to constrain overfitting.
- Batch size controls how many examples contribute to each gradient update.
- Learning rate controls update size. Batch size and learning rate interact,
  which motivates tuning their combinations in the joint grid.
- PR-AUC focuses on retrieving delayed flights in an imbalanced dataset.
- Five seeds expose variation from initialisation and training order.
- Preprocessing fitted on evaluation data would leak distribution information.
