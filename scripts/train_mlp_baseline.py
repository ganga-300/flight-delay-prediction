"""Train the Phase 2 Member 1 PyTorch MLP Baseline.

Features:
- Reads train and validation data from data/processed/{train,valid}.parquet (or generates if missing)
- Fits FeatureEncoder and MLPPreprocessor on training data only
- Transforms validation data unchanged
- Trains MLP with Adam (lr=0.001), batch_size=256, max_epochs=30, patience=5, seed=42
- Tracks train loss, valid loss, PR-AUC, ROC-AUC, Brier score, and ECE
- Saves best checkpoint based on validation PR-AUC
- Generates all required figures and results in reports/
- Saves model and preprocessing artifacts in models/phase2/member1/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

# Ensure Matplotlib has a writable cache directory
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib
matplotlib.use("Agg")
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flight_delay.config import load_config
from flight_delay.data.schema import TARGET_COLUMN
from flight_delay.features.build import CATEGORICAL, numeric_features
from flight_delay.features.encoding import FeatureEncoder
from flight_delay.models.mlp_training import (
    MLPPreprocessor,
    evaluate_mlp,
    get_device,
    train_mlp_baseline,
)


def plot_loss_curve(history: dict, output_path: Path) -> None:
    """Plot Training vs Validation BCE loss across epochs."""
    plt.figure(figsize=(8, 5))
    epochs = history["epoch"]
    plt.plot(epochs, history["train_loss"], label="Training Loss", color="#1f77b4", lw=2)
    plt.plot(epochs, history["valid_loss"], label="Validation Loss", color="#ff7f0e", lw=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("BCE Loss", fontsize=12)
    plt.title("MLP Baseline: Training vs Validation Loss Curve", fontsize=13, fontweight="bold")
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

    plt.plot(epochs, pr_aucs, label="Validation PR-AUC", color="#2ca02c", lw=2, marker="o", markersize=4)
    plt.axvline(best_epoch, color="#d62728", linestyle="--", alpha=0.8, label=f"Best Epoch {best_epoch} (PR-AUC={best_pr:.4f})")
    plt.scatter([best_epoch], [best_pr], color="#d62728", s=100, zorder=5)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("PR-AUC (Average Precision)", fontsize=12)
    plt.title("MLP Baseline: Validation PR-AUC vs Epoch", fontsize=13, fontweight="bold")
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
        {"name": "Input Layer", "detail": f"Preprocessed Features\nDim: {input_dim}\n(49 numeric + {input_dim - 49} OHE)", "color": "#d9eaf7"},
        {"name": "Hidden Layer 1", "detail": "Linear(128)\nReLU\nDropout(p=0.3)", "color": "#b8e2f2"},
        {"name": "Hidden Layer 2", "detail": "Linear(64)\nReLU\nDropout(p=0.2)", "color": "#9ed5eb"},
        {"name": "Output Layer", "detail": "Linear(1)\n(Raw Logits)", "color": "#82c5e3"},
        {"name": "Inference / Eval", "detail": "Sigmoid(Logits)\nOutput: Delay Prob in [0, 1]", "color": "#c7e9c0"},
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
        ax.text(x, y_center + 0.15, layer["name"], ha="center", va="center", fontsize=11, fontweight="bold")
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
    plt.title(f"PyTorch Simple MLP Baseline Architecture (Input Dimension = {input_dim})", fontsize=13, fontweight="bold", pad=20)
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
        title="MLP Baseline: Validation Confusion Matrix (Threshold=0.50)",
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
    plt.plot(recall, precision, label=f"MLP Baseline (PR-AUC = {pr_auc_score:.4f})", color="#1f77b4", lw=2)
    plt.axhline(baseline_rate, color="navy", linestyle="--", label=f"Validation Delay Rate Baseline ({baseline_rate:.3f})")

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("MLP Baseline: Precision-Recall Curve (Validation Set)", fontsize=13, fontweight="bold")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-pos-weight", action="store_true", help="Use training-derived pos_weight for BCE loss")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu, cuda, mps, or auto)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed

    # Directories
    model_dir = ROOT / "models" / "phase2" / "member1"
    figures_dir = ROOT / "reports" / "figures" / "phase2_member1"
    results_dir = ROOT / "reports" / "results" / "phase2_member1"

    model_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    train_path = ROOT / cfg["paths"]["processed"] / "train.parquet"
    valid_path = ROOT / cfg["paths"]["processed"] / "valid.parquet"
    encoder_path = ROOT / cfg["paths"]["processed"] / "encoder.json"

    if not train_path.exists() or not valid_path.exists() or not encoder_path.exists():
        raise FileNotFoundError(
            f"Processed data files not found in {cfg['paths']['processed']}. Run python scripts/build_features.py first."
        )

    print("Loading pre-split data from Parquet files...")
    train_df = pd.read_parquet(train_path)
    valid_df = pd.read_parquet(valid_path)
    feature_encoder = FeatureEncoder.load(encoder_path)

    num_cols = numeric_features(cfg)
    cat_cols = CATEGORICAL
    target_col = TARGET_COLUMN

    y_train = train_df[target_col].to_numpy(dtype=np.float32)
    y_valid = valid_df[target_col].to_numpy(dtype=np.float32)

    # 2. Fit Preprocessor on Training data only
    print("Fitting MLPPreprocessor (StandardScaler + OneHotEncoder) on training split only...")
    preprocessor = MLPPreprocessor(
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        feature_encoder=feature_encoder,
        min_count=cfg.get("features", {}).get("categorical_min_count", 50),
    )
    X_train = preprocessor.fit_transform(train_df)
    X_valid = preprocessor.transform(valid_df)

    input_dim = preprocessor.input_dim_
    print(f"Preprocessor fit complete.")
    print(f"Input dimensions: {input_dim} features ({len(num_cols)} numerical + {len(preprocessor.ohe_feature_names_)} one-hot categorical).")
    print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}, size: {X_train.nbytes / (1024**2):.1f} MB")
    print(f"X_valid shape: {X_valid.shape}, dtype: {X_valid.dtype}, size: {X_valid.nbytes / (1024**2):.1f} MB")

    # Save preprocessor and feature encoder state
    preprocessor.save(model_dir / "mlp_preprocessor.joblib")
    feature_encoder.save(model_dir / "feature_encoder.json")
    print(f"Saved preprocessor to {model_dir / 'mlp_preprocessor.joblib'}")

    # 3. Detect device
    device = get_device(args.device)
    print(f"Selected computation device: {device}")

    # 4. Train Model
    model, history, best_epoch, best_val_metrics = train_mlp_baseline(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        device=device,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=seed,
        use_pos_weight=args.use_pos_weight,
    )

    # Save model weights (best checkpoint)
    torch.save(model.state_dict(), model_dir / "mlp_baseline.pth")
    print(f"Saved best model weights to {model_dir / 'mlp_baseline.pth'}")

    # 5. Evaluate best model on Validation set
    print("\nEvaluating best checkpoint on September-October validation set...")
    val_eval = evaluate_mlp(model, X_valid, y_valid, device, threshold=0.5)

    # 6. Save Metrics, Config, History, and Run Metadata
    model_config = {
        "architecture": "Input -> Linear(128) -> ReLU -> Dropout(0.3) -> Linear(64) -> ReLU -> Dropout(0.2) -> Linear(1)",
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
    }
    with open(model_dir / "model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    with open(model_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(model_dir / "validation_metrics.json", "w") as f:
        json.dump(val_eval, f, indent=2)

    # Also save to results dir
    with open(results_dir / "validation_metrics.json", "w") as f:
        json.dump(val_eval, f, indent=2)

    # Metrics CSV for easy comparison
    val_metrics_df = pd.DataFrame([{
        "model": "MLP_Baseline",
        "validation_period": "2013-09-01 to 2013-10-31",
        "pr_auc": val_eval["pr_auc"],
        "roc_auc": val_eval["roc_auc"],
        "brier_score": val_eval["brier_score"],
        "ece_10": val_eval["ece_10"],
        "loss": val_eval["loss"],
        "precision": val_eval["precision"],
        "recall": val_eval["recall"],
        "f1_score": val_eval["f1_score"],
        "positive_rate": val_eval["positive_rate"],
        "mean_predicted_probability": val_eval["mean_predicted_probability"],
        "best_epoch": best_epoch,
    }])
    val_metrics_df.to_csv(results_dir / "validation_metrics.csv", index=False)

    # Comparison CSV showing protocol context
    comparison_df = pd.DataFrame([
        {
            "model": "Logistic_Regression (Phase 1 Winner)",
            "eval_period": "August 2013 (Tuning Selection)",
            "pr_auc": 0.4007,
            "roc_auc": 0.6756,
            "notes": "Historical D-201 August selection score (not Sept-Oct validation)",
        },
        {
            "model": "XGBoost (Phase 1 Candidate)",
            "eval_period": "August 2013 (Tuning Selection)",
            "pr_auc": 0.3960,
            "roc_auc": 0.6770,
            "notes": "Historical D-201 August selection score (not Sept-Oct validation)",
        },
        {
            "model": "MLP_Baseline (Phase 2 Member 1)",
            "eval_period": "Sept-Oct 2013 (Validation Set)",
            "pr_auc": val_eval["pr_auc"],
            "roc_auc": val_eval["roc_auc"],
            "notes": f"Best epoch {best_epoch}, Adam lr=0.001, OHE input dim={input_dim}",
        },
    ])
    comparison_df.to_csv(results_dir / "model_comparison.csv", index=False)

    run_metadata = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "split": cfg["split"],
        "training_rows": len(train_df),
        "validation_rows": len(valid_df),
        "training_positive_rate": float(np.mean(y_train)),
        "validation_positive_rate": float(np.mean(y_valid)),
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "one_hot_feature_count": len(preprocessor.ohe_feature_names_),
        "total_input_dim": input_dim,
        "train_parquet_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "valid_parquet_sha256": hashlib.sha256(valid_path.read_bytes()).hexdigest(),
        "best_epoch": best_epoch,
        "training_device": str(device),
    }
    with open(model_dir / "run_metadata.json", "w") as f:
        json.dump(run_metadata, f, indent=2)

    # 7. Generate Visualizations
    print("\nGenerating figures in reports/figures/phase2_member1/...")

    # A. Training vs Validation Loss Curve
    loss_fig = figures_dir / "loss_curve.png"
    plot_loss_curve(history, loss_fig)
    print(f"Generated {loss_fig}")

    # B. Validation PR-AUC vs Epoch
    pr_fig = figures_dir / "valid_pr_auc_vs_epoch.png"
    plot_pr_auc_curve(history, best_epoch, pr_fig)
    print(f"Generated {pr_fig}")

    # C. MLP Architecture Diagram
    arch_fig = figures_dir / "mlp_architecture.png"
    plot_architecture_diagram(input_dim, arch_fig)
    print(f"Generated {arch_fig}")

    # D. Confusion Matrix
    cm_fig = figures_dir / "confusion_matrix.png"
    plot_confusion_matrix(val_eval["confusion_matrix"], cm_fig)
    print(f"Generated {cm_fig}")

    # E. Precision-Recall Curve
    with torch.no_grad():
        x_val_t = torch.from_numpy(X_valid).to(device)
        logits_val = model(x_val_t).squeeze(-1).cpu().numpy()
        probs_val = 1.0 / (1.0 + np.exp(-np.clip(logits_val, -50.0, 50.0)))
    pr_curve_fig = figures_dir / "precision_recall_curve.png"
    plot_precision_recall_curve(y_valid, probs_val, val_eval["pr_auc"], pr_curve_fig)
    print(f"Generated {pr_curve_fig}")

    print("\n=== Phase 2 Member 1: MLP Baseline Complete ===")
    print(f"Best Epoch: {best_epoch}")
    print(f"Validation PR-AUC: {val_eval['pr_auc']:.5f} (Primary Metric)")
    print(f"Validation ROC-AUC: {val_eval['roc_auc']:.5f}")
    print(f"Validation Brier Score: {val_eval['brier_score']:.5f}")
    print(f"Validation ECE: {val_eval['ece_10']:.5f}")
    print(f"Validation Loss: {val_eval['loss']:.5f}")
    print(f"Confusion Matrix: {val_eval['confusion_matrix']}")
    print(f"Saved artifacts in {model_dir}")


if __name__ == "__main__":
    main()
