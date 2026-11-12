"""Training, preprocessing, evaluation, and inference pipelines for PyTorch MLP baseline."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from flight_delay.evaluation.metrics import (
    brier,
    expected_calibration_error,
    pr_auc,
    recall_at_precision,
    roc_auc,
)
from flight_delay.features.encoding import FeatureEncoder
from flight_delay.models.mlp import MLPBaseline
from flight_delay.utils.seed import set_seed

logger = logging.getLogger(__name__)


def get_device(requested: str | None = None) -> torch.device:
    """Detect and return the appropriate PyTorch device.

    Auto-detection order:
    1. CUDA if available
    2. Apple MPS if available
    3. CPU otherwise
    """
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def set_torch_seed(seed: int = 42) -> None:
    """Ensure reproducibility across random, numpy, and torch."""
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS determinism where supported
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


class MLPPreprocessor:
    """Unified preprocessor for MLP baseline.

    Applies:
    1. FeatureEncoder (rare-level collapsing to __OTHER__, median imputation)
    2. StandardScaler on numeric features (fit on train only)
    3. OneHotEncoder(handle_unknown='ignore') on categorical features (fit on train only)
    4. Concatenation into float32 array
    """

    def __init__(
        self,
        numeric_cols: list[str],
        categorical_cols: list[str],
        feature_encoder: FeatureEncoder | None = None,
        min_count: int = 50,
    ) -> None:
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.min_count = min_count
        self.feature_encoder = feature_encoder
        self.scaler = StandardScaler()
        self.one_hot_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32,
        )
        self.ohe_feature_names_: list[str] = []
        self.all_feature_names_: list[str] = []
        self.input_dim_: int = 0
        self.is_fitted_: bool = False

    def fit(self, train_df: pd.DataFrame) -> MLPPreprocessor:
        """Fit preprocessor components on training data only."""
        if self.feature_encoder is None:
            self.feature_encoder = FeatureEncoder(
                categorical=self.categorical_cols,
                numeric=self.numeric_cols,
                min_count=self.min_count,
            )
            encoded_df = self.feature_encoder.fit_transform(train_df)
        else:
            # Transform if encoder already provided/fitted
            encoded_df = self.feature_encoder.transform(train_df)

        # 1. Fit standard scaler on numeric features
        X_num = encoded_df[self.numeric_cols].to_numpy(dtype=np.float32)
        self.scaler.fit(X_num)

        # 2. Fit one-hot encoder on categorical features
        X_cat = encoded_df[self.categorical_cols]
        self.one_hot_encoder.fit(X_cat)

        self.ohe_feature_names_ = list(
            self.one_hot_encoder.get_feature_names_out(self.categorical_cols)
        )
        self.all_feature_names_ = self.numeric_cols + self.ohe_feature_names_
        self.input_dim_ = len(self.all_feature_names_)
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform input DataFrame to dense float32 array."""
        if not self.is_fitted_:
            raise RuntimeError("MLPPreprocessor must be fitted before transforming data.")

        # Ensure FeatureEncoder transformations are applied
        encoded_df = self.feature_encoder.transform(df)

        # Numeric standard scaling
        X_num = encoded_df[self.numeric_cols].to_numpy(dtype=np.float32)
        X_num_scaled = self.scaler.transform(X_num).astype(np.float32)

        # Categorical one-hot encoding
        X_cat = encoded_df[self.categorical_cols]
        X_cat_ohe = self.one_hot_encoder.transform(X_cat).astype(np.float32)

        # Combine
        X_combined = np.hstack([X_num_scaled, X_cat_ohe])
        return X_combined.astype(np.float32)

    def fit_transform(self, train_df: pd.DataFrame) -> np.ndarray:
        """Fit on train and transform."""
        return self.fit(train_df).transform(train_df)

    def save(self, path: str | Path) -> None:
        """Persist preprocessor to disk using joblib."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @classmethod
    def load(cls, path: str | Path) -> MLPPreprocessor:
        """Load fitted preprocessor from disk."""
        return joblib.load(path)


def evaluate_mlp(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
    batch_size: int = 1024,
) -> dict[str, Any]:
    """Evaluate MLP model and return all required performance metrics.

    Metrics reported:
    1. PR-AUC (Average Precision) - primary metric
    2. ROC-AUC
    3. Brier Score
    4. Expected Calibration Error (ECE)
    5. Validation loss (BCEWithLogitsLoss)
    6. Precision (at threshold)
    7. Recall (at threshold)
    8. F1-score (at threshold)
    9. Confusion matrix [[TN, FP], [FN, TP]]
    10. Positive-class rate
    """
    model.eval()
    y_true = np.asarray(y, dtype=np.float32).ravel()

    logits_list: list[np.ndarray] = []
    num_samples = len(X)
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_x = torch.from_numpy(X[i : i + batch_size]).to(device)
            batch_logits = model(batch_x).squeeze(-1).cpu().numpy()
            logits_list.append(batch_logits)

    logits = np.concatenate(logits_list, axis=0) if logits_list else np.array([], dtype=np.float32)
    # Probability prediction using sigmoid
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))

    # Loss computation (unweighted BCE for standard evaluation)
    criterion = nn.BCEWithLogitsLoss()
    loss_val = float(criterion(torch.from_numpy(logits), torch.from_numpy(y_true)).item())

    # Classification at decision threshold
    preds = (probs >= threshold).astype(int)
    y_int = y_true.astype(int)

    cm = confusion_matrix(y_int, preds, labels=[0, 1]).tolist()
    prec = float(precision_score(y_int, preds, zero_division=0))
    rec = float(recall_score(y_int, preds, zero_division=0))
    f1 = float(f1_score(y_int, preds, zero_division=0))
    pos_rate = float(np.mean(y_true))
    mean_prob = float(np.mean(probs))

    prauc = pr_auc(y_true, probs)
    rocauc = roc_auc(y_true, probs)
    brier_score = brier(y_true, probs)
    ece_10 = expected_calibration_error(y_true, probs, n_bins=10)

    # Operational point feasibility check at min_precision=0.60
    op_recall, op_thresh = recall_at_precision(y_true, probs, min_precision=0.60)

    return {
        "pr_auc": float(prauc),
        "roc_auc": float(rocauc),
        "brier_score": float(brier_score),
        "ece_10": float(ece_10),
        "loss": loss_val,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "threshold": threshold,
        "confusion_matrix": cm,
        "positive_rate": pos_rate,
        "mean_predicted_probability": mean_prob,
        "op_precision_0.60_recall": float(op_recall),
        "op_precision_0.60_threshold": float(op_thresh) if not np.isnan(op_thresh) else None,
        "sample_count": int(num_samples),
    }


def train_mlp_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray | None,
    y_valid: np.ndarray | None,
    device: torch.device,
    learning_rate: float = 0.001,
    batch_size: int = 256,
    max_epochs: int = 30,
    patience: int = 5,
    seed: int = 42,
    use_pos_weight: bool = False,
) -> tuple[MLPBaseline, dict[str, Any], int, dict[str, Any]]:
    """Train MLP baseline with Adam, BCEWithLogitsLoss, and early stopping on validation PR-AUC.

    With ``X_valid=None`` the model trains for exactly ``max_epochs`` epochs with no early
    stopping. This is the final refit, whose epoch count was chosen on the tuning month.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        X_valid: Tuning-period feature matrix, or None for a fixed-epoch refit.
        y_valid: Tuning-period labels, or None for a fixed-epoch refit.
        device: PyTorch device.
        learning_rate: Learning rate for Adam (default 0.001).
        batch_size: Mini-batch size (default 256).
        max_epochs: Maximum epochs (default 30).
        patience: Early stopping patience epochs (default 5).
        seed: Random seed for reproducibility.
        use_pos_weight: Whether to use training-derived pos_weight for loss.

    Returns:
        tuple of (best_model, history, best_epoch, final_val_metrics)
    """
    set_torch_seed(seed)
    input_dim = X_train.shape[1]

    # Model architecture as specified:
    # Linear(128) -> ReLU -> Dropout(0.3) -> Linear(64) -> ReLU -> Dropout(0.2) -> Linear(1)
    model = MLPBaseline(
        input_dim=input_dim,
        hidden_dim1=128,
        hidden_dim2=64,
        dropout1=0.3,
        dropout2=0.2,
    ).to(device)

    # Class counts derived strictly from training labels
    pos_count = int(np.sum(y_train == 1))
    neg_count = int(np.sum(y_train == 0))
    pos_weight_value = neg_count / max(1, pos_count)

    if use_pos_weight:
        pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Prepare PyTorch datasets and DataLoaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train.astype(np.float32)).unsqueeze(-1),
    )
    # Generator for reproducible batch shuffling
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        drop_last=False,
    )

    history: dict[str, list[Any]] = {
        "epoch": [],
        "train_loss": [],
        "valid_loss": [],
        "valid_pr_auc": [],
        "valid_roc_auc": [],
        "valid_brier": [],
        "valid_ece": [],
    }

    best_pr_auc = -1.0
    best_epoch = 0
    best_model_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] = {}
    epochs_no_improve = 0

    n_valid = "none (fixed epochs)" if X_valid is None else f"{len(X_valid):,}"
    print(
        f"Starting training on {device} (Input dim: {input_dim}, "
        f"Train: {len(X_train):,}, Valid: {n_valid})"
    )
    print(
        f"Class imbalance: Train Positive={pos_count:,} ({pos_count / len(y_train):.4f}), "
        f"Negative={neg_count:,}"
    )
    print(f"pos_weight={pos_weight_value:.4f} (applied to loss: {use_pos_weight})")

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            batches += 1

        train_loss = running_loss / max(1, batches)

        if X_valid is None:
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            print(f"Epoch {epoch:2d}/{max_epochs:2d} | Train Loss: {train_loss:.5f}")
            best_epoch = epoch
            continue

        # Validation after every epoch
        val_metrics = evaluate_mlp(model, X_valid, y_valid, device, threshold=0.5)
        val_loss = val_metrics["loss"]
        val_pr_auc = val_metrics["pr_auc"]
        val_roc_auc = val_metrics["roc_auc"]
        val_brier = val_metrics["brier_score"]
        val_ece = val_metrics["ece_10"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(val_loss)
        history["valid_pr_auc"].append(val_pr_auc)
        history["valid_roc_auc"].append(val_roc_auc)
        history["valid_brier"].append(val_brier)
        history["valid_ece"].append(val_ece)

        is_best = val_pr_auc > best_pr_auc
        marker = " *" if is_best else ""
        print(
            f"Epoch {epoch:2d}/{max_epochs:2d} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Valid Loss: {val_loss:.5f} | "
            f"Valid PR-AUC: {val_pr_auc:.5f} | "
            f"Valid ROC-AUC: {val_roc_auc:.5f} | "
            f"ECE: {val_ece:.4f}{marker}"
        )

        if is_best:
            best_pr_auc = val_pr_auc
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            best_metrics = val_metrics
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(no PR-AUC improvement for {patience} epochs)."
                )
                break

    # Load best checkpoint weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(
            f"Restored best checkpoint from Epoch {best_epoch} with Valid PR-AUC: {best_pr_auc:.5f}"
        )

    return model, history, best_epoch, best_metrics


def predict_flight_delays(
    artifacts_dir: str | Path,
    df: pd.DataFrame,
    threshold: float = 0.5,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reusable prediction function for new flight features.

    1. Loads saved preprocessing artifacts and model configuration
    2. Applies exactly the same preprocessing pipeline
    3. Returns delay probabilities and predicted classes

    Args:
        artifacts_dir: Directory containing saved model artifacts.
        df: DataFrame containing flight features.
        threshold: Classification probability threshold.
        device: PyTorch device ('cpu', 'cuda', 'mps', or auto if None).

    Returns:
        tuple of (probabilities, predicted_classes) as numpy arrays.
    """
    model_path = Path(artifacts_dir) / "mlp_baseline.pth"
    preprocessor_path = Path(artifacts_dir) / "mlp_preprocessor.joblib"
    config_path = Path(artifacts_dir) / "model_config.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")

    with open(config_path) as f:
        cfg = json.load(f)

    preprocessor = MLPPreprocessor.load(preprocessor_path)
    X = preprocessor.transform(df)

    dev = get_device(device)
    model = MLPBaseline(
        input_dim=cfg["input_dim"],
        hidden_dim1=cfg.get("hidden_dim1", 128),
        hidden_dim2=cfg.get("hidden_dim2", 64),
        dropout1=cfg.get("dropout1", 0.3),
        dropout2=cfg.get("dropout2", 0.2),
    )
    model.load_state_dict(torch.load(model_path, map_location=dev))
    model.to(dev)
    model.eval()

    with torch.no_grad():
        x_tensor = torch.from_numpy(X).to(dev)
        logits = model(x_tensor).squeeze(-1).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
        preds = (probs >= threshold).astype(int)

    return probs, preds
