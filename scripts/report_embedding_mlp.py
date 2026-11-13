"""Build Member 3 figures, documentation and a read-only notebook from saved results."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nbformat  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / "reports/results/phase2_member3"
RESULT = HISTORICAL / "d307"
MODEL = ROOT / "models/phase2/member3/refit_d307"
FIGURE = ROOT / "reports/figures/phase2_member3/d307"
LABELS = {
    "embedding": "Embedding MLP",
    "mlp": "Member 1 MLP",
    "gru": "Member 2 GRU",
    "lr": "Logistic Regression",
}
COLORS = {"embedding": "#087e8b", "mlp": "#777777", "gru": "#6854a3", "lr": "#d87924"}


def read_json(path):
    return json.loads(path.read_text())


def save(figure, name):
    figure.savefig(FIGURE / name, dpi=160, bbox_inches="tight")
    plt.close(figure)


def markdown_table(frame):
    text = frame.round(5).astype(object).fillna("—").astype(str)
    rows = [
        "| " + " | ".join(text.columns) + " |",
        "| " + " | ".join(["---"] * len(text.columns)) + " |",
    ]
    rows.extend("| " + " | ".join(row) + " |" for row in text.to_numpy().tolist())
    return "\n".join(rows)


def main():
    FIGURE.mkdir(parents=True, exist_ok=True)
    selection = read_json(RESULT / "selection.json")
    frozen = read_json(MODEL / "frozen.json")
    evaluation = read_json(RESULT / "evaluation.json")
    winner = selection["winner"]
    grid = pd.read_csv(RESULT / "grid_august.csv")
    summary = pd.read_csv(RESULT / "comparison.csv")
    overall = summary.query("period == 'sep_oct'").set_index("model")
    per_seed = pd.read_csv(RESULT / "per_seed_metrics.csv")
    bootstrap = pd.read_csv(RESULT / "bootstrap_pr_auc_differences.csv")
    predictions = pd.read_parquet(RESULT / "sep_oct_predictions.parquet")
    epochs = pd.read_csv(RESULT / "august_seed_selection.csv")
    labels = predictions["is_delayed"].to_numpy()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )

    architectures = list(
        dict.fromkeys(
            (row.learning_rate, row.hidden_dims, row.embedding_cap) for row in grid.itertuples()
        )
    )
    columns = [(dropout, batch) for dropout in (0.1, 0.2, 0.4) for batch in (256, 1024)]
    matrix = np.zeros((8, 6))
    for row in grid.itertuples():
        i = architectures.index((row.learning_rate, row.hidden_dims, row.embedding_cap))
        j = columns.index((row.dropout, row.batch_size))
        matrix[i, j] = row.august_pr_auc
    figure, axis = plt.subplots(figsize=(12, 8))
    heatmap = axis.imshow(matrix, cmap="YlGnBu", aspect="auto")
    for row in grid.itertuples():
        i = architectures.index((row.learning_rate, row.hidden_dims, row.embedding_cap))
        j = columns.index((row.dropout, row.batch_size))
        reused = "*" if row.dropout == 0.2 and row.batch_size == 256 else ""
        axis.text(
            j,
            i,
            f"{row.august_pr_auc:.4f}{reused}",
            ha="center",
            va="center",
            color="white" if row.august_pr_auc > np.median(matrix) else "black",
        )
        if row.run == winner["run"]:
            axis.add_patch(
                Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#f07000", linewidth=3)
            )
    axis.set_xticks(range(6), [f"dropout {dropout}\nbatch {batch}" for dropout, batch in columns])
    axis.set_yticks(
        range(8), [f"lr={rate:g}, h={hidden}, cap={cap}" for rate, hidden, cap in architectures]
    )
    axis.set_title("D-307 — all 48 joint settings\n* reused D-305 run; orange box selects winner")
    figure.colorbar(heatmap, ax=axis, label="August PR-AUC (tuning scores)")
    save(figure, "august_grid.png")

    run = read_json(RESULT / "tuning_d307" / f"{winner['run']}.json")
    history = pd.DataFrame(run["history"])
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="Jan–Jul training BCE")
    axes[0].plot(history["epoch"], history["august_loss"], label="August BCE")
    axes[0].set(xlabel="Epoch", ylabel="BCE loss", title="Selected configuration, seed 42")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["august_pr_auc"], marker="o", color=COLORS["embedding"])
    axes[1].axvline(winner["best_epoch"], linestyle="--", color="#333333", label="Selected epoch")
    axes[1].set(xlabel="Epoch", ylabel="August PR-AUC", title="Epoch selection uses August only")
    axes[1].legend()
    save(figure, "training_curves.png")

    figure, axis = plt.subplots(figsize=(8, 5))
    for index, name in enumerate(["embedding", "mlp", "gru"]):
        values = per_seed.query("model == @name and period == 'sep_oct'")["pr_auc"].to_numpy()
        axis.scatter(
            np.arange(42, 47) + (index - 1) * 0.06, values, color=COLORS[name], label=LABELS[name]
        )
    axis.axhline(
        overall.loc["lr", "pr_auc_mean"], color=COLORS["lr"], linestyle="--", label=LABELS["lr"]
    )
    axis.set(
        xticks=list(range(42, 47)),
        xlabel="Random seed",
        ylabel="Sep–Oct PR-AUC",
        title="Five frozen models per neural architecture",
    )
    axis.legend()
    save(figure, "seed_comparison.png")

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name in ["embedding", "mlp", "gru", "lr"]:
        column = "lr" if name == "lr" else f"{name}_seed42"
        probability = predictions[column].to_numpy()
        precision, recall, _ = precision_recall_curve(labels, probability)
        axes[0].plot(recall, precision, color=COLORS[name], label=LABELS[name])
        bins = np.minimum((probability * 10).astype(int), 9)
        observed, predicted = [], []
        for group in range(10):
            mask = bins == group
            if mask.any():
                observed.append(labels[mask].mean())
                predicted.append(probability[mask].mean())
        axes[1].plot(predicted, observed, marker="o", color=COLORS[name], label=LABELS[name])
    axes[0].axhline(labels.mean(), color="#aaaaaa", linestyle="--", label="Delay rate")
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Sep–Oct precision–recall (seed 42)")
    axes[1].plot([0, 1], [0, 1], "--", color="#aaaaaa")
    axes[1].set(
        xlabel="Mean predicted probability",
        ylabel="Observed delay rate",
        title="Raw probability reliability (seed 42)",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    for axis in axes:
        axis.legend(fontsize=8)
    save(figure, "pr_and_reliability.png")

    figure, axis = plt.subplots(figsize=(9, 4))
    position = np.arange(len(bootstrap))
    axis.errorbar(
        bootstrap["difference"],
        position,
        xerr=[
            bootstrap["difference"] - bootstrap["ci_low"],
            bootstrap["ci_high"] - bootstrap["difference"],
        ],
        fmt="o",
        color=COLORS["embedding"],
        capsize=5,
    )
    axis.axvline(0, color="#555555", linestyle="--")
    axis.set(
        yticks=position,
        yticklabels=bootstrap["comparison"],
        xlabel="PR-AUC difference with paired 95% day-bootstrap interval",
        title="Uncertainty across September–October days",
    )
    save(figure, "bootstrap_differences.png")

    embedding_dims = [
        max(1, min(winner["embedding_cap"], int(np.ceil(np.sqrt(size)))))
        for size in frozen["vocabulary_sizes"]
    ]
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.set(xlim=(0, 12), ylim=(0, 4))
    axis.axis("off")
    blocks = [
        (0.1, 2.2, 2.1, 1.2, "49 numeric features\nTraining median + scaler"),
        (
            0.1,
            0.5,
            2.1,
            1.2,
            "10 categorical fields\nTrain-only vocabularies\n"
            f"Cap {winner['embedding_cap']} embeddings",
        ),
        (3.0, 1.3, 2.0, 1.4, f"Concatenate\n{49 + sum(embedding_dims)} values"),
        (5.8, 1.3, 2.5, 1.4, f"Dense {winner['hidden_dims']}\nReLU + dropout {winner['dropout']}"),
        (9.1, 1.3, 2.2, 1.4, "One output logit\nSigmoid at inference\nDelay probability"),
    ]
    for x, y, width, height, text in blocks:
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.1",
                facecolor="#e4f2f3",
                edgecolor=COLORS["embedding"],
            )
        )
        axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=10)
    for start, end in [
        ((2.3, 2.8), (2.9, 2.2)),
        ((2.3, 1.1), (2.9, 1.7)),
        ((5.1, 2), (5.7, 2)),
        ((8.4, 2), (9, 2)),
    ]:
        axis.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->"})
    axis.set_title(
        "Member 3 — selected embedding MLP after January–August refit", fontweight="bold"
    )
    save(figure, "embedding_architecture.png")

    table = overall.reset_index()[
        ["model", "pr_auc_mean", "pr_auc_std", "roc_auc_mean", "brier_score_mean", "ece_10_mean"]
    ]
    winner_description = (
        f"learning rate {winner['learning_rate']}, hidden layers {winner['hidden_dims']}, "
        f"embedding cap {winner['embedding_cap']}, dropout {winner['dropout']}, "
        f"batch size {winner['batch_size']}"
    )
    previous = (
        pd.read_csv(HISTORICAL / "comparison.csv")
        .query("model == 'embedding' and period == 'sep_oct'")
        .iloc[0]
    )
    previous_settings = read_json(HISTORICAL / "selection.json")["winner"]
    versions = pd.DataFrame(
        [
            {
                "result": name,
                "dropout": dropout,
                "batch_size": batch,
                "August tuning PR-AUC": august,
                **{
                    metric: float(scores[metric + "_mean"])
                    for metric in ("pr_auc", "roc_auc", "brier_score", "ece_10")
                },
                "pr_auc_std": float(scores["pr_auc_std"]),
            }
            for name, dropout, batch, august, scores in [
                ("D-306 historical", 0.2, 256, previous_settings["august_pr_auc"], previous),
                (
                    "D-307 final",
                    winner["dropout"],
                    winner["batch_size"],
                    winner["august_pr_auc"],
                    overall.loc["embedding"],
                ),
            ]
        ]
    )
    disclosure = (
        "Sep–Oct was scored once under D-305 (D-306) before this amendment. "
        "D-307 replaces D-306 as the Member 3 result because D-305 did not tune batch size "
        "and dropout as assigned. Sep–Oct is not used for any choice."
    )
    significant = bootstrap.loc[bootstrap["ci_low"] > 0, "comparison"].tolist()
    reading = (
        "Positive 95% intervals support improvements for " + ", ".join(significant) + "."
        if significant
        else "No comparison has a positive 95% interval entirely above zero."
    )
    lower = bootstrap.loc[bootstrap["ci_high"] < 0, "comparison"].tolist()
    if lower:
        reading += " Negative intervals support lower PR-AUC for " + ", ".join(lower) + "."
    weights = torch.load(MODEL / "seed42.pth", weights_only=True)
    parameters = sum(value.numel() for value in weights.values())
    document = f"""# Phase 2 — Member 3 embedding MLP — D-307 final

## Amendment and evaluation disclosure

> {disclosure}

D-307 is the final Member 3 experiment. Historical D-305/D-306 records, checkpoints,
predictions and figures retain their original bytes. The report below shows both versions.

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
  This motivates including dropout in the D-307 tuning grid.
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

Selected architecture settings are {winner_description}. The January–August refit has
{parameters:,} trainable parameters, with embedding dimensions {embedding_dims}.
Embeddings and scaled numeric values are concatenated, followed by two dense layers, ReLU,
dropout {winner["dropout"]} after each hidden layer, and one logit. Sigmoid converts logits
to probabilities at inference. Airline and route representations learn jointly with the loss.

![Architecture](../reports/figures/phase2_member3/d307/embedding_architecture.png)

## Registered experiment and preprocessing

D-307 was committed as `90eecd5` before amended training. The full Cartesian grid has
48 configurations, with learning rates 0.0001/0.0003, widths (64,32)/(128,64), embedding caps
4/8, dropout 0.1/0.2/0.4, and batch size 256/1024. Eight D-305 runs at dropout 0.2 and batch
256 were reused after verifying their original source, input and selection records. Forty
new configurations were trained. Original record paths and hashes are saved with reused runs.
Fixed choices are Adam, weight decay 0.0001, unweighted binary cross-entropy with logits,
CPU, maximum 30 epochs and patience 5.

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

{markdown_table(grid)}

Winner `{winner["run"]}` has August PR-AUC {winner["august_pr_auc"]:.5f} at epoch
{winner["best_epoch"]} for seed 42. These are tuning scores.

{markdown_table(epochs)}

![Grid](../reports/figures/phase2_member3/d307/august_grid.png)
![Training](../reports/figures/phase2_member3/d307/training_curves.png)

## D-306 and D-307 side by side

{markdown_table(versions)}

D-306 remains a historical result. D-307 supersedes it because the amended joint grid covers
the assigned batch-size and dropout tuning. September–October metrics are reported after
the August-only choice is frozen. They do not select a configuration or a result version.

## Final September–October results

All four model families use the exact same {evaluation["rows"]:,} flight IDs, dates and labels.
The delay rate is {labels.mean():.4f}. Neural model metrics are means and sample standard
deviations over seeds 42–46. Logistic Regression is the saved Phase 1 model and has one run.
All probabilities are raw. Monthly results are in `comparison.csv`.

{markdown_table(table)}

![Seeds](../reports/figures/phase2_member3/d307/seed_comparison.png)
![PR and reliability](../reports/figures/phase2_member3/d307/pr_and_reliability.png)

## Paired uncertainty analysis

The existing Member 2 bootstrap resamples all 61 evaluation days with replacement 1,000 times.
It computes each model's PR-AUC per seed, averages over seeds, then takes the paired difference
on the same sampled flights. It does not average probabilities into a new ensemble. An
improvement requires the entire 95% interval to be positive, as registered in D-307.

{markdown_table(bootstrap)}

{reading}

![Uncertainty](../reports/figures/phase2_member3/d307/bootstrap_differences.png)

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
Source hashes and the D-307 hash are stored in every stage. Each final checkpoint and the
fitted preprocessor have hashes in `models/phase2/member3/refit_d307/frozen.json`.

## Deliverables

- `src/flight_delay/models/embedding_mlp.py` and `embedding_training.py`
- `scripts/train_embedding_mlp.py` and `report_embedding_mlp.py`
- `tests/test_embedding_mlp.py` and `test_embedding_pipeline.py`
- `models/phase2/member3/refit_d307/` with five seed checkpoints and preprocessing
- `reports/results/phase2_member3/d307/` with all registered runs and final comparisons
- `reports/figures/phase2_member3/d307/` with architecture and diagnostic figures
- `notebooks/phase2_member3_embedding_mlp.ipynb` with read-only results
"""
    (ROOT / "docs/phase2_member3_embedding_mlp.md").write_text(document)

    presentation = f"""# Member 3 — D-307 final presentation and demonstration

## Slide 1 — Contribution and amendment

- Built an embedding MLP using 49 numeric and 10 categorical features.
- Registered D-307 before amended training to cover dropout and batch size.
- Evaluated the full 48-setting joint grid with seed 42.
- Reused eight verified D-305 runs and trained 40 new settings.

Speaker notes

{disclosure}

## Slide 2 — Architecture

Use `reports/figures/phase2_member3/d307/embedding_architecture.png`.

- Selected {winner_description}.
- Concatenate numeric values with trainable categorical embeddings.
- Apply two ReLU hidden layers, dropout, and one output logit.
- Use sigmoid to produce a probability at inference.
- The refit model contains {parameters:,} parameters.
- Fit vocabulary, imputation and scalers on each stage's training rows only.

## Slide 3 — Joint tuning

Use `reports/figures/phase2_member3/d307/august_grid.png`.

- Learning rate 0.0001/0.0003, hidden layers 64–32/128–64, embedding cap 4/8.
- Dropout 0.1/0.2/0.4 and batch size 256/1024.
- Fit January–July and choose every setting and epoch by August PR-AUC.
- Winner August PR-AUC {winner["august_pr_auc"]:.5f}, seed-42 epoch {winner["best_epoch"]}.
- Exact ties prefer fewer parameters, lower learning rate, then grid order.
- Adam, weight decay 0.0001, unweighted BCE, maximum 30 epochs, patience 5.

## Slide 4 — Five seeds and final results

- Seeds 42–46 each choose their own epoch on August.
- Reinitialise and refit January–August for the chosen epoch count.
- Freeze every model before the D-307 September–October scoring.
- Use the same 55,628 flights and saved MLP, LR and GRU baselines.

{markdown_table(table)}

## Slide 5 — Historical result next to final result

{markdown_table(versions)}

Speaker notes

D-307 is the final Member 3 result because it completes the assigned joint grid.
The historical D-306 results remain available for review. Final evaluation values
were observed after architecture and epoch choices were frozen.

## Slide 6 — Paired uncertainty and interpretation

Use `reports/figures/phase2_member3/d307/bootstrap_differences.png`.

{markdown_table(bootstrap)}

{reading}

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
"""
    (ROOT / "docs/phase2_member3_presentation.md").write_text(presentation)

    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            "# Member 3 — D-307 final embedding MLP\n\n"
            "Read-only summary of the saved artifacts.\n\n" + disclosure
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport json\nimport pandas as pd\n"
            "from IPython.display import Image, display\n"
            "ROOT = Path.cwd()\nif ROOT.name == 'notebooks':\n    ROOT = ROOT.parent\n"
            "RESULT = ROOT / 'reports/results/phase2_member3/d307'\n"
            "selection = json.loads((RESULT / 'selection.json').read_text())\n"
            "display(pd.Series({k: selection['winner'][k] for k in "
            "['run', 'learning_rate', 'hidden_dims', 'embedding_cap', 'dropout', 'batch_size', "
            "'best_epoch', 'august_pr_auc']}))"
        ),
        nbformat.v4.new_markdown_cell(
            "## August tuning and training\n\n"
            "All settings and epochs were selected on August PR-AUC."
        ),
        nbformat.v4.new_code_cell(
            "display(pd.read_csv(RESULT / 'grid_august.csv'))\n"
            "display(pd.read_csv(RESULT / 'august_seed_selection.csv'))\n"
            "FIGURE = ROOT / 'reports/figures/phase2_member3/d307'\n"
            "display(Image(filename=str(FIGURE / 'august_grid.png')))\n"
            "display(Image(filename=str(FIGURE / 'embedding_architecture.png')))\n"
            "display(Image(filename=str(FIGURE / 'training_curves.png')))"
        ),
        nbformat.v4.new_markdown_cell(
            "## Frozen September–October comparison\n\n"
            "The same 55,628 flights are used for all models. "
            "Neural scores are reported across five seeds."
        ),
        nbformat.v4.new_code_cell(
            "comparison = pd.read_csv(RESULT / 'comparison.csv')\n"
            "display(comparison.query(\"period == 'sep_oct'\"))\n"
            "previous = pd.read_csv(RESULT.parent / 'comparison.csv')\n"
            "versions = pd.concat([previous.assign(result='D-306 historical'), "
            "comparison.assign(result='D-307 final')])\n"
            "display(versions.query(\"model == 'embedding' and period == 'sep_oct'\"))\n"
            "display(pd.read_csv(RESULT / 'bootstrap_pr_auc_differences.csv'))\n"
            "display(Image(filename=str(FIGURE / 'seed_comparison.png')))\n"
            "display(Image(filename=str(FIGURE / 'bootstrap_differences.png')))"
        ),
        nbformat.v4.new_markdown_cell(
            "## Limitations\n\nRaw probabilities require calibration. "
            "Evaluation periods were viewed in earlier project iterations. "
            "Confidence intervals describe day-level variation for these fixed designs. "
            "Full protocol, interpretation and source references are in "
            "`docs/phase2_member3_embedding_mlp.md`."
        ),
    ]
    nbformat.write(notebook, ROOT / "notebooks/phase2_member3_embedding_mlp.ipynb")
    print("Generated six figures, Member 3 documentation and read-only notebook")


if __name__ == "__main__":
    main()
