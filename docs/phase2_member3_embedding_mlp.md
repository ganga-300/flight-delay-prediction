# Phase 2 — Member 3 embedding MLP

## Scope and ownership

Aditya Ranjan implemented the categorical-embedding model, its preprocessing, August-only
hyperparameter selection, five-seed refits and paired comparisons. Member 1's seeding utilities,
Member 2's logit metrics and sigmoid, and `scripts/train_gru.py::day_bootstrap` are imported.
Their implementation and saved artifacts retain their original contents.

## Relevant methodological literature

The following primary-source abstracts were verified for this implementation.

- Guo and Berkhahn (2016) learn supervised entity embeddings for categorical fields.
  This motivates replacing sparse one-hot inputs with trainable category representations.
  [Entity Embeddings of Categorical Variables](https://arxiv.org/abs/1604.06737)
- Srivastava et al. (2014) introduce dropout to reduce co-adaptation and overfitting.
  This motivates the fixed hidden-layer dropout in D-305.
  [Dropout, JMLR](https://jmlr.org/papers/v15/srivastava14a.html)
- Gorishniy et al. (2021) compare tabular neural architectures and boosted trees under
  consistent protocols. This motivates retaining strong ML baselines and measuring the
  embedding MLP's value with the same flights and metrics.
  [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959)

## Question and architecture

Can learned categorical representations improve next-day arrival-delay classification compared
with one-hot MLP, Logistic Regression and GRU on identical flights?

The target is arrival delay of at least 15 minutes, conditional on an observed arrival label.
The decision cutoff is the end of D−1. Inputs are the frozen 49 numeric and 10 categorical
features from Phase 1. Each categorical field has a learned embedding with dimension
`max(1, min(cap, ceil(sqrt(training vocabulary size))))`.

Selected architecture settings are learning rate 0.0001, hidden layers [128, 64], embedding cap 4. The January–August refit has
23,315 trainable parameters, with embedding dimensions [4, 2, 4, 4, 3, 4, 4, 4, 4, 4].
Embeddings and scaled numeric values are concatenated, followed by two dense layers, ReLU,
dropout 0.2 after each hidden layer, and one logit. Sigmoid converts logits to probabilities
at inference. Airline and route representations learn jointly with the classification loss.

![Architecture](../reports/figures/phase2_member3/embedding_architecture.png)

## Registered experiment and preprocessing

D-305 was committed as `12f13e8` before Member 3 training. The registered grid has eight
configurations, with learning rates 0.0001/0.0003, widths (64,32)/(128,64), and embedding caps
4/8. Fixed choices are Adam, weight decay 0.0001, batch size 256, unweighted binary
cross-entropy with logits, CPU, maximum 30 epochs and patience 5.

Binary cross-entropy is `-[y log(p) + (1-y) log(1-p)]`. Learning embeddings and network weights
minimises this objective. Dropout and weight decay constrain overfitting. The lower learning
rates follow the earlier models' early August peaks. Small layers and low-dimensional
embeddings limit capacity on a modest tabular dataset.

The January–July vocabulary uses training counts to collapse rare levels to `__OTHER__`.
Numeric medians and StandardScaler statistics use those training rows only. August and later
unseen levels map to the reserved category. For final training, preprocessing is fitted anew
on January–August and model weights are reinitialised. This can change vocabulary sizes.

| Period | Use | Labelled flights |
| --- | --- | --- |
| January–July | Fit all grid candidates | 188,971 |
| August | Select configuration and epochs by PR-AUC | 28,756 |
| January–August | Fixed-epoch final refit | 217,727 |
| September–October | Final comparison | 55,628 |
| November–December | Sealed during Member 3 work | Excluded |

Each configuration uses seed 42. Exact PR-AUC ties prefer fewer parameters, then lower
learning rate, then configuration order. Earliest epochs win equal-score checkpoint ties.
After the architecture is frozen, each seed 42–46 selects its own epoch on August and refits
January–August for exactly that count. September–October has no role in these decisions.

## August selection

| run | learning_rate | hidden_dims | embedding_cap | parameters | best_epoch | august_pr_auc |
| --- | --- | --- | --- | --- | --- | --- |
| run01 | 0.0001 | [64, 32] | 4 | 11523 | 2 | 0.39448 |
| run02 | 0.0001 | [64, 32] | 8 | 16555 | 3 | 0.39372 |
| run03 | 0.0001 | [128, 64] | 4 | 23299 | 2 | 0.39683 |
| run04 | 0.0001 | [128, 64] | 8 | 29739 | 3 | 0.39493 |
| run05 | 0.0003 | [64, 32] | 4 | 11523 | 1 | 0.39262 |
| run06 | 0.0003 | [64, 32] | 8 | 16555 | 2 | 0.39114 |
| run07 | 0.0003 | [128, 64] | 4 | 23299 | 1 | 0.39473 |
| run08 | 0.0003 | [128, 64] | 8 | 29739 | 1 | 0.39323 |

Winner `run03` has August PR-AUC 0.39683 at epoch
2 for seed 42. These are tuning scores.

| seed | best_epoch | august_pr_auc |
| --- | --- | --- |
| 42 | 2 | 0.39683 |
| 43 | 3 | 0.39679 |
| 44 | 2 | 0.39678 |
| 45 | 5 | 0.39712 |
| 46 | 2 | 0.39772 |

![Grid](../reports/figures/phase2_member3/august_grid.png)
![Training](../reports/figures/phase2_member3/training_curves.png)

## Final September–October results

All four model families use the exact same 55,628 flight IDs, dates and labels.
The delay rate is 0.1570. Neural model metrics are means and sample standard
deviations over seeds 42–46. Logistic Regression is the saved Phase 1 model and has one run.
All probabilities are raw. Monthly results are in `comparison.csv`.

| model | pr_auc_mean | pr_auc_std | roc_auc_mean | brier_score_mean | ece_10_mean |
| --- | --- | --- | --- | --- | --- |
| embedding | 0.2762 | 0.00242 | 0.67528 | 0.12793 | 0.04732 |
| gru | 0.29457 | 0.00552 | 0.67434 | 0.136 | 0.08628 |
| lr | 0.28959 | — | 0.68 | 0.18554 | 0.23341 |
| mlp | 0.28393 | 0.00429 | 0.67187 | 0.13327 | 0.07574 |

![Seeds](../reports/figures/phase2_member3/seed_comparison.png)
![PR and reliability](../reports/figures/phase2_member3/pr_and_reliability.png)

## Paired uncertainty analysis

The existing Member 2 bootstrap resamples all 61 evaluation days with replacement 1,000 times.
It computes each model's PR-AUC per seed, averages over seeds, then takes the paired difference
on the same sampled flights. It does not average probabilities into a new ensemble. An
improvement requires the entire 95% interval to be positive, as registered in D-305.

| comparison | difference | ci_low | ci_high | excludes_zero |
| --- | --- | --- | --- | --- |
| embedding - mlp | -0.00773 | -0.02533 | 0.0111 | False |
| embedding - lr | -0.01339 | -0.02121 | -0.00345 | True |
| embedding - gru | -0.01837 | -0.03869 | 0.00197 | False |

No comparison has a positive 95% interval entirely above zero. Negative intervals support lower PR-AUC for embedding - lr.

![Uncertainty](../reports/figures/phase2_member3/bootstrap_differences.png)

## Interpretation and limits

Learned embeddings provide a compact way to model airline, route and time-bucket interactions.
The measured differences and their uncertainty determine the performance conclusion. The
grid and epochs remain fixed after observing evaluation results.

The model comparison changes representation, width and optimisation together. Matched
ablation experiments by Member 4 can isolate the contribution of embeddings. The LR baseline
uses balanced class weights, while the neural models use unweighted BCE. This also influences
the raw probability calibration comparison.

These results cover one year's NYC departures, schedule information and past-known outcomes.
Weather forecasts and same-day operational signals are absent. Early stopping addresses
overfitting but cannot supply missing information. Raw probability reliability is reported;
probability calibration belongs to Member 5's separately designed evaluation work.

The earlier project history includes prior exposure to evaluation periods, disclosed in D-000
and D-301. September–October is a shared evaluation period already viewed by earlier members.
The bootstrap reflects variation across these days, conditional on the fixed model designs and
five selected seeds. It does not establish performance in unseen years or correct historical
evaluation exposure. Additional architectures would need a new registered experiment.

## Reproduction and recovery

Use Python 3.11 and the recorded environment in `environment.txt`. Saved results can be read
without retraining. On a clean development checkout, install requirements and the package,
create the dedicated January–October development file with the existing preparation script,
then execute these stages in order.

```bash
python scripts/train_embedding_mlp.py --stage check
python scripts/train_embedding_mlp.py --stage tune
python scripts/train_embedding_mlp.py --stage refit
python scripts/train_embedding_mlp.py --stage evaluate
python scripts/report_embedding_mlp.py
```

Completed selection, refits and evaluation are protected from overwrite. Interrupted tuning
and refits reuse completed records after checking source and input hashes. If bootstrapping is
interrupted, evaluation resumes from saved probabilities and avoids rescoring frozen models.
Source hashes and the D-305 hash are stored in every stage. Each final checkpoint and the
fitted preprocessor have hashes in `models/phase2/member3/frozen.json`.

## Deliverables

- `src/flight_delay/models/embedding_mlp.py` and `embedding_training.py`
- `scripts/train_embedding_mlp.py` and `report_embedding_mlp.py`
- `tests/test_embedding_mlp.py` and `test_embedding_pipeline.py`
- `models/phase2/member3/` with five seed checkpoints and preprocessing
- `reports/results/phase2_member3/` with all registered runs and final comparisons
- `reports/figures/phase2_member3/` with architecture and diagnostic figures
- `notebooks/phase2_member3_embedding_mlp.ipynb` with read-only results
