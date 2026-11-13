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
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "reports/results/phase2_member3"
MODEL = ROOT / "models/phase2/member3"
FIGURE = ROOT / "reports/figures/phase2_member3"
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

    figure, axis = plt.subplots(figsize=(11, 6))
    names = [
        f"{row.run}: lr={row.learning_rate:g}, h={row.hidden_dims}, cap={row.embedding_cap}"
        for row in grid.itertuples()
    ]
    axis.barh(
        names,
        grid["august_pr_auc"],
        color=[COLORS["embedding"] if run == winner["run"] else "#bdcbd2" for run in grid["run"]],
    )
    axis.invert_yaxis()
    axis.set(xlabel="August PR-AUC (tuning scores)", title="D-305 — all eight registered settings")
    for index, value in enumerate(grid["august_pr_auc"]):
        axis.text(value + 0.001, index, f"{value:.4f}", va="center")
    axis.set_xlim(0, max(grid["august_pr_auc"]) + 0.045)
    save(figure, "august_grid.png")

    run = read_json(RESULT / "tuning" / f"{winner['run']}.json")
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
        (5.8, 1.3, 2.5, 1.4, f"Dense {winner['hidden_dims']}\nReLU + dropout 0.2"),
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
        f"embedding cap {winner['embedding_cap']}"
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
    document = f"""# Phase 2 — Member 3 embedding MLP

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

Selected architecture settings are {winner_description}. The January–August refit has
{parameters:,} trainable parameters, with embedding dimensions {embedding_dims}.
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

{markdown_table(grid)}

Winner `{winner["run"]}` has August PR-AUC {winner["august_pr_auc"]:.5f} at epoch
{winner["best_epoch"]} for seed 42. These are tuning scores.

{markdown_table(epochs)}

![Grid](../reports/figures/phase2_member3/august_grid.png)
![Training](../reports/figures/phase2_member3/training_curves.png)

## Final September–October results

All four model families use the exact same {evaluation["rows"]:,} flight IDs, dates and labels.
The delay rate is {labels.mean():.4f}. Neural model metrics are means and sample standard
deviations over seeds 42–46. Logistic Regression is the saved Phase 1 model and has one run.
All probabilities are raw. Monthly results are in `comparison.csv`.

{markdown_table(table)}

![Seeds](../reports/figures/phase2_member3/seed_comparison.png)
![PR and reliability](../reports/figures/phase2_member3/pr_and_reliability.png)

## Paired uncertainty analysis

The existing Member 2 bootstrap resamples all 61 evaluation days with replacement 1,000 times.
It computes each model's PR-AUC per seed, averages over seeds, then takes the paired difference
on the same sampled flights. It does not average probabilities into a new ensemble. An
improvement requires the entire 95% interval to be positive, as registered in D-305.

{markdown_table(bootstrap)}

{reading}

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
"""
    (ROOT / "docs/phase2_member3_embedding_mlp.md").write_text(document)

    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            "# Member 3 — embedding MLP\n\n"
            "Read-only summary of the D-305 saved artifacts. "
            "No training or model selection runs here."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport json\nimport pandas as pd\n"
            "from IPython.display import Image, display\n"
            "ROOT = Path.cwd()\nif ROOT.name == 'notebooks':\n    ROOT = ROOT.parent\n"
            "RESULT = ROOT / 'reports/results/phase2_member3'\n"
            "selection = json.loads((RESULT / 'selection.json').read_text())\n"
            "display(pd.Series({k: selection['winner'][k] for k in "
            "['run', 'learning_rate', 'hidden_dims', 'embedding_cap', "
            "'best_epoch', 'august_pr_auc']}))"
        ),
        nbformat.v4.new_markdown_cell(
            "## August tuning and training\n\n"
            "All settings and epochs were selected on August PR-AUC."
        ),
        nbformat.v4.new_code_cell(
            "display(pd.read_csv(RESULT / 'grid_august.csv'))\n"
            "display(pd.read_csv(RESULT / 'august_seed_selection.csv'))\n"
            "FIGURE = ROOT / 'reports/figures/phase2_member3'\n"
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
