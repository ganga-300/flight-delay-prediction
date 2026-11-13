# Member 3 — presentation and demonstration

## Slide 1 — My contribution

- Built an embedding-based MLP for next-day flight-delay classification.
- Used the existing 49 numeric and 10 categorical features.
- Registered eight settings in D-305 before training.
- Delivered five seed models, comparisons, uncertainty estimates and reproducible code.

Speaker notes

I continued the Phase 1 feature pipeline. My question was whether learned category
representations could improve prediction compared with the existing MLP, LR and GRU.

## Slide 2 — How the network works

Use `reports/figures/phase2_member3/embedding_architecture.png`.

- Scale numeric features using training statistics.
- Give each category a trainable vector.
- Combine 49 numeric values and 37 learned embedding values.
- Apply two hidden layers of 128 and 64 neurons, with ReLU and dropout.
- Convert the output logit into a delay probability using sigmoid.

Speaker notes

For example, an airline code receives a small vector of numbers. The model learns
these numbers while predicting delayed and on-time flights. The final network has
23,315 trainable parameters. Unknown categories map to a reserved category.

## Slide 3 — Training protocol

Use `reports/figures/phase2_member3/august_grid.png`.

- Fit on January–July.
- Select settings and epochs using August PR-AUC.
- Selected learning rate 0.0001, hidden layers 128/64 and embedding cap 4.
- Refit January–August using seeds 42–46 and their selected epochs.
- Score September–October after every model is frozen.

Speaker notes

The eight runs combine two learning rates, two network widths and two embedding
caps. Seed 42 selected epoch 2 and August PR-AUC 0.3968. The final seeds selected
2, 3, 2, 5 and 2 epochs. Every stage learns its preprocessing from its training rows.

## Slide 4 — Final results

Use `reports/figures/phase2_member3/seed_comparison.png`.

| Model | Sep–Oct PR-AUC |
| --- | --- |
| Embedding MLP | 0.2762 ± 0.0024 |
| Member 1 MLP | 0.2839 ± 0.0043 |
| Logistic Regression | 0.2896 |
| GRU | 0.2946 ± 0.0055 |

Speaker notes

All models use the same 55,628 flights. My embedding model has lower PR-AUC in
this experiment. Its Brier score is 0.1279 and ECE is 0.0473, the smallest raw
probability errors among these compared models. Loss weighting and model choices
also affect calibration, so these differences need careful interpretation.

## Slide 5 — Uncertainty and conclusion

Use `reports/figures/phase2_member3/bootstrap_differences.png`.

- Resample whole days 1,000 times to preserve within-day dependence.
- Compute paired PR-AUC differences on identical flights.
- The embedding-minus-LR interval is entirely below zero.
- The MLP and GRU difference intervals include zero.
- Record the result under the registered protocol and retain the trained artifacts.

Speaker notes

The LR comparison supports lower ranking performance for my model. This experiment
does not demonstrate a ranking improvement from the chosen embedding architecture.
One year of NYC data and past-known information limits generalisation. Matched
ablations can isolate representation effects in Member 4's work.

## Demonstration

1. Open D-305 and show its commit preceding training.
2. Open the architecture code and the eight August experiment records.
3. Open `notebooks/phase2_member3_embedding_mlp.ipynb` and its executed results.
4. Show the five saved checkpoints and `frozen.json`.
5. Show paired differences and explain the negative LR interval.

## Viva questions

- Why embeddings? They learn compact representations of categorical levels jointly
  with the prediction task. Their value must be established experimentally.
- Why lower learning rates? Earlier models peaked in one or two August epochs.
  Smaller steps were included in the predeclared grid.
- Why five seeds? Random initialisation and batch order can change a neural model's
  performance. Reporting mean and standard deviation makes this variation visible.
- Why PR-AUC? Delayed flights are the minority class, and precision–recall focuses
  on retrieving those flights. The project predeclared it as the primary metric.
- Why raw calibration differs from ranking? PR-AUC measures ordering, while Brier
  and ECE assess probability values. These properties can move differently.
- What is leakage? Using information unavailable at prediction time, or fitting
  preprocessing on evaluation rows, can inflate measured performance.
