"""Train the Phase 2 Member 1 PyTorch MLP baseline under the D-301 protocol.

Same one-job-per-period protocol as Phase 1 (D-201):
- Fit January-July; early-stop and pick the best epoch on August PR-AUC.
- Refit January-August for exactly that many epochs (no early stopping, no later data).
- Score September-October once, next to the Phase 1 Logistic Regression (D-202 final model,
  also trained on January-August) on the same flights.

Encoders and scalers are fitted on the training rows of each stage only. The sealed
November-December data is never read. Refuses to overwrite existing outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure Matplotlib has a writable cache directory
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flight_delay.config import load_config  # noqa: E402
from flight_delay.data.load import load_dev_raw  # noqa: E402
from flight_delay.data.schema import TARGET_COLUMN  # noqa: E402
from flight_delay.evaluation.metrics import (  # noqa: E402
    brier,
    expected_calibration_error,
    pr_auc,
    roc_auc,
)
from flight_delay.features.build import CATEGORICAL, build_feature_table, numeric_features  # noqa: E402
from flight_delay.features.encoding import FeatureEncoder  # noqa: E402
from flight_delay.models.mlp_training import (  # noqa: E402
    MLPPreprocessor,
    evaluate_mlp,
    get_device,
    train_mlp_baseline,
)
from flight_delay.models.persistence import load_model  # noqa: E402


def plot_loss_curve(history: dict, output_path: Path) -> None:
    """Plot Training vs Validation BCE loss across epochs."""
    plt.figure(figsize=(8, 5))
    epochs = history["epoch"]
    plt.plot(epochs, history["train_loss"], label="Training loss (Jan-Jul)", color="#1f77b4", lw=2)
    plt.plot(epochs, history["valid_loss"], label="Tuning loss (August)", color="#ff7f0e", lw=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("BCE Loss", fontsize=12)
    plt.title("MLP Baseline: Training vs August Loss", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_pr_auc_curve(history: dict, best_epoch: int, output_path: Path) -> None:
    """Plot Validation PR-AUC across epochs and mark the best epoch."""
    plt.figure(figsize=(8, 5))
    epochs = history["epoch"]
    pr_aucs = history["valid_pr_auc"]
    best_pr = pr_aucs[best_epoch - 1]

    plt.plot(
        epochs, pr_aucs, label="August PR-AUC", color="#2ca02c", lw=2, marker="o", markersize=4
    )
    plt.axvline(
        best_epoch,
        color="#d62728",
        linestyle="--",
        alpha=0.8,
        label=f"Best Epoch {best_epoch} (PR-AUC={best_pr:.4f})",
    )
    plt.scatter([best_epoch], [best_pr], color="#d62728", s=100, zorder=5)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("PR-AUC (Average Precision)", fontsize=12)
    plt.title("MLP Baseline: August PR-AUC vs Epoch", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_architecture_diagram(input_dim: int, output_path: Path) -> None:
    """Generate an architecture diagram showing dimensions, activations, and dropout."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    layers = [
        {
            "name": "Input Layer",
            "detail": (
                f"Preprocessed Features\nDim: {input_dim}\n(49 numeric + {input_dim - 49} OHE)"
            ),
            "color": "#d9eaf7",
        },
        {
            "name": "Hidden Layer 1",
            "detail": "Linear(128)\nReLU\nDropout(p=0.3)",
            "color": "#b8e2f2",
        },
        {
            "name": "Hidden Layer 2",
            "detail": "Linear(64)\nReLU\nDropout(p=0.2)",
            "color": "#9ed5eb",
        },
        {"name": "Output Layer", "detail": "Linear(1)\n(Raw Logits)", "color": "#82c5e3"},
        {
            "name": "Inference / Eval",
            "detail": "Sigmoid(Logits)\nOutput: Delay Prob in [0, 1]",
            "color": "#c7e9c0",
        },
    ]

    n = len(layers)
    box_width = 1.4
    box_height = 0.8
    y_center = 0.5

    for i, layer in enumerate(layers):
        x = i * 2.1 + 0.8
        rect = FancyBboxPatch(
            (x - box_width / 2, y_center - box_height / 2),
            box_width,
            box_height,
            boxstyle="round,pad=0.2",
            fc=layer["color"],
            ec="#333333",
            lw=1.5,
        )
        ax.add_patch(rect)
        ax.text(
            x,
            y_center + 0.15,
            layer["name"],
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
        ax.text(x, y_center - 0.15, layer["detail"], ha="center", va="center", fontsize=9)

        if i < n - 1:
            ax.annotate(
                "",
                xy=(x + box_width / 2 + 0.7, y_center),
                xytext=(x + box_width / 2, y_center),
                arrowprops=dict(arrowstyle="->", lw=2, color="#333333"),
            )

    ax.set_xlim(-0.2, n * 2.1 + 0.2)
    ax.set_ylim(0, 1.0)
    plt.title(
        f"PyTorch Simple MLP Baseline Architecture (Input Dimension = {input_dim})",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_confusion_matrix(cm: list[list[int]], output_path: Path) -> None:
    """Plot validation confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_arr = np.array(cm)

    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    classes = ["On-Time (<15 min)", "Delayed (>=15 min)"]
    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=classes,
        yticklabels=classes,
        title="MLP: Sep-Oct Confusion Matrix (diagnostic threshold 0.50)",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    thresh = cm_arr.max() / 2.0
    for i in range(2):
        for j in range(2):
            val = cm_arr[i, j]
            ax.text(
                j,
                i,
                f"{val:,}\n({val / cm_arr.sum():.1%})",
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=11,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    probs: np.ndarray,
    pr_auc_score: float,
    output_path: Path,
) -> None:
    """Plot validation Precision-Recall curve with baseline delay rate."""
    precision, recall, _ = precision_recall_curve(y_true, probs)
    baseline_rate = float(np.mean(y_true))

    plt.figure(figsize=(8, 5))
    plt.plot(
        recall,
        precision,
        label=f"MLP Baseline (PR-AUC = {pr_auc_score:.4f})",
        color="#1f77b4",
        lw=2,
    )
    plt.axhline(
        baseline_rate,
        color="navy",
        linestyle="--",
        label=f"Sep-Oct delay rate ({baseline_rate:.3f})",
    )

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("MLP Baseline: Precision-Recall Curve (Sep-Oct)", fontsize=13, fontweight="bold")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


FIT = ("2013-01-01", "2013-07-31")
TUNE = ("2013-08-01", "2013-08-31")
EVAL = ("2013-09-01", "2013-10-31")
LR_MODEL = ROOT / "models" / "d201" / "final_model.joblib"
LR_ENCODER = ROOT / "models" / "d201" / "final_encoder.json"


def period(table: pd.DataFrame, bounds: tuple[str, str]) -> pd.DataFrame:
    dates = pd.to_datetime(table["date"])
    rows = table.loc[dates.between(*bounds)].copy()
    if rows.empty or rows[TARGET_COLUMN].nunique() != 2:
        raise SystemExit(f"period {bounds} is empty or has one class")
    return rows


def scores(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "pr_auc": float(pr_auc(y, p)),
        "roc_auc": float(roc_auc(y, p)),
        "brier_score": float(brier(y, p)),
        "ece_10": float(expected_calibration_error(y, p, n_bins=10)),
        "mean_predicted_probability": float(np.mean(p)),
        "positive_rate": float(np.mean(y)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-pos-weight", action="store_true", help="Use training-derived pos_weight for BCE loss"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="cpu (default, reproducible), cuda or mps"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed

    model_dir = ROOT / "models" / "phase2" / "member1"
    figures_dir = ROOT / "reports" / "figures" / "phase2_member1"
    results_dir = ROOT / "reports" / "results" / "phase2_member1"
    if (model_dir / "mlp_baseline.pth").exists() or (
        results_dir / "validation_metrics.json"
    ).exists():
        raise SystemExit("D-302 outputs already exist; refusing to overwrite")
    for d in (model_dir, figures_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    num_cols = numeric_features(cfg)
    cat_cols = list(CATEGORICAL)
    if len(num_cols) != 49 or len(cat_cols) != 10:
        raise SystemExit("expected the 59 frozen features (49 numeric + 10 categorical)")
    min_count = cfg["features"]["categorical_min_count"]

    # 1. Build features on the January-October development file (the sealed file is not read).
    print("Building features on the January-October development data...")
    table = build_feature_table(load_dev_raw(cfg, ROOT), cfg)
    fit, tune, evaluation = period(table, FIT), period(table, TUNE), period(table, EVAL)
    final = pd.concat([fit, tune], ignore_index=True)
    y = {
        name: df[TARGET_COLUMN].to_numpy(dtype=np.float32)
        for name, df in [("fit", fit), ("tune", tune), ("final", final), ("eval", evaluation)]
    }

    device = get_device(args.device)
    print(f"Selected computation device: {device}")
    common = dict(
        device=device,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        seed=seed,
        use_pos_weight=args.use_pos_weight,
    )

    # 2. Stage A: fit January-July, early-stop on August.
    print("\nStage A: fit January-July, select the epoch on August PR-AUC")
    prep_a = MLPPreprocessor(num_cols, cat_cols, min_count=min_count)
    X_fit = prep_a.fit_transform(fit)
    X_tune = prep_a.transform(tune)
    _, history, best_epoch, august = train_mlp_baseline(
        X_fit,
        y["fit"],
        X_tune,
        y["tune"],
        max_epochs=args.max_epochs,
        patience=args.patience,
        **common,
    )
    del X_fit, X_tune

    # 3. Stage B: refit January-August for exactly best_epoch epochs.
    print(f"\nStage B: refit January-August for {best_epoch} epoch(s), no early stopping")
    preprocessor = MLPPreprocessor(num_cols, cat_cols, min_count=min_count)
    X_final = preprocessor.fit_transform(final)
    model, final_history, _, _ = train_mlp_baseline(
        X_final, y["final"], None, None, max_epochs=best_epoch, patience=best_epoch, **common
    )
    del X_final
    input_dim = preprocessor.input_dim_

    preprocessor.save(model_dir / "mlp_preprocessor.joblib")
    preprocessor.feature_encoder.save(model_dir / "feature_encoder.json")
    torch.save(model.state_dict(), model_dir / "mlp_baseline.pth")

    # 4. Score September-October once, MLP and Phase 1 Logistic Regression on the same flights.
    print("\nScoring September-October (never used for any choice above)...")
    X_eval = preprocessor.transform(evaluation)
    val_eval = evaluate_mlp(model, X_eval, y["eval"], device, threshold=0.5)
    with torch.no_grad():
        logits = model(torch.from_numpy(X_eval).to(device)).squeeze(-1).cpu().numpy()
    probs_eval = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))

    lr_encoder = FeatureEncoder.load(LR_ENCODER)
    lr_probs = load_model(LR_MODEL).predict_proba(
        lr_encoder.transform(evaluation)[num_cols + cat_cols]
    )[:, 1]
    lr_scores = scores(y["eval"].astype(int), lr_probs)
    mlp_scores = scores(y["eval"].astype(int), probs_eval)

    # 5. Save config, histories, metrics and run metadata.
    model_config = {
        "protocol": "D-301",
        "architecture": (
            "Input -> Linear(128) -> ReLU -> Dropout(0.3) -> Linear(64) -> ReLU"
            " -> Dropout(0.2) -> Linear(1)"
        ),
        "input_dim": input_dim,
        "hidden_dim1": 128,
        "hidden_dim2": 64,
        "dropout1": 0.3,
        "dropout2": 0.2,
        "optimizer": "Adam",
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "loss_function": "BCEWithLogitsLoss",
        "use_pos_weight": args.use_pos_weight,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "seed": seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "final_training_epochs": best_epoch,
    }
    (model_dir / "model_config.json").write_text(json.dumps(model_config, indent=2))
    (model_dir / "training_history.json").write_text(
        json.dumps(
            {"stage_a_jan_jul_vs_august": history, "stage_b_jan_aug_refit": final_history}, indent=2
        )
    )
    (model_dir / "august_selection_metrics.json").write_text(json.dumps(august, indent=2))
    for d in (model_dir, results_dir):
        (d / "validation_metrics.json").write_text(json.dumps(val_eval, indent=2))

    pd.DataFrame(
        [
            {
                "model": "MLP_Baseline",
                "evaluation_period": "2013-09-01 to 2013-10-31",
                **mlp_scores,
                "loss": val_eval["loss"],
                "precision_at_0.5": val_eval["precision"],
                "recall_at_0.5": val_eval["recall"],
                "f1_at_0.5": val_eval["f1_score"],
                "best_epoch": best_epoch,
            }
        ]
    ).to_csv(results_dir / "validation_metrics.csv", index=False)

    pd.DataFrame(
        [
            {
                "model": "Logistic Regression (Phase 1 final, raw)",
                **lr_scores,
                "notes": "D-202 model, trained Jan-Aug; never tuned on Sep-Oct",
            },
            {
                "model": "MLP baseline (Phase 2 Member 1, raw)",
                **mlp_scores,
                "notes": f"trained Jan-Aug for {best_epoch} epoch(s) chosen on August",
            },
        ]
    ).to_csv(results_dir / "model_comparison.csv", index=False)

    run_metadata = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "protocol": "D-301",
        "stage_a_fit_period": "2013-01-01 to 2013-07-31",
        "stage_a_tuning_period": "2013-08-01 to 2013-08-31",
        "final_training_period": "2013-01-01 to 2013-08-31",
        "evaluation_period": "2013-09-01 to 2013-10-31",
        "holdout_accessed": False,
        "rows": {"fit": len(fit), "tune": len(tune), "final": len(final), "eval": len(evaluation)},
        "positive_rate": {k: float(np.mean(v)) for k, v in y.items()},
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "one_hot_feature_count": len(preprocessor.ohe_feature_names_),
        "total_input_dim": input_dim,
        "best_epoch": best_epoch,
        "august_pr_auc_at_best_epoch": august["pr_auc"],
        "training_device": str(device),
        "torch_version": torch.__version__,
    }
    (model_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2))

    # 6. Figures.
    plot_loss_curve(history, figures_dir / "loss_curve.png")
    plot_pr_auc_curve(history, best_epoch, figures_dir / "august_pr_auc_vs_epoch.png")
    plot_architecture_diagram(input_dim, figures_dir / "mlp_architecture.png")
    plot_confusion_matrix(val_eval["confusion_matrix"], figures_dir / "confusion_matrix.png")
    plot_precision_recall_curve(
        y["eval"], probs_eval, val_eval["pr_auc"], figures_dir / "precision_recall_curve.png"
    )

    print("\n=== Phase 2 Member 1: MLP baseline (D-302) ===")
    print(f"Best epoch on August: {best_epoch} (August PR-AUC {august['pr_auc']:.4f})")
    for name, sc in (("MLP", mlp_scores), ("Logistic Regression", lr_scores)):
        print(
            f"{name:20s} Sep-Oct PR-AUC {sc['pr_auc']:.4f}  ROC-AUC {sc['roc_auc']:.4f}  "
            f"Brier {sc['brier_score']:.4f}  ECE {sc['ece_10']:.4f}  "
            f"mean p {sc['mean_predicted_probability']:.3f}"
        )
    print(f"Saved artifacts in {model_dir}")


if __name__ == "__main__":
    main()
