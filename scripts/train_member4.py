"""Member 4 execution pipeline under D-309.

Executes:
1. Four systematic ablations against Member 3 D-307 Embedding MLP reference:
   - Ablation 1: Remove historical features (29 features)
   - Ablation 2: Remove congestion features (4 features)
   - Ablation 3: Replace embeddings with one-hot encoding
   - Ablation 4: Disable dropout and weight decay
2. Plain XGBoost baseline retrained with D-201 settings.
3. Hybrid Deep Learning + XGBoost model (64-d neural representation + 49 numeric features).
4. Five seeds (42-46) per model family.
5. Tuning on August, refit on Jan-Aug, evaluation on Sep-Oct (55,628 flights).
6. 1,000 day-level bootstrap resamples for paired difference testing.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Prevent OpenMP runtime conflicts between PyTorch and XGBoost on macOS Apple Silicon
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_gru import day_bootstrap  # noqa: E402

from flight_delay.config import load_config  # noqa: E402
from flight_delay.data.load import load_dev_raw  # noqa: E402
from flight_delay.evaluation.metrics import (  # noqa: E402
    brier,
    expected_calibration_error,
    pr_auc,
    roc_auc,
)
from flight_delay.features.build import (  # noqa: E402
    CATEGORICAL,
    build_feature_table,
    feature_groups,
    numeric_features,
)
from flight_delay.features.encoding import FeatureEncoder  # noqa: E402
from flight_delay.models.ablation import (  # noqa: E402
    OneHotMLP,
    get_ablation_numeric_features,
    predict_one_hot_logits,
    train_one_hot_ablation,
)
from flight_delay.models.embedding_mlp import (  # noqa: E402
    EmbeddingMLP,
    EmbeddingPreprocessor,
)
from flight_delay.models.embedding_training import (  # noqa: E402
    EmbeddingData,
    predict_logits,
    train_embedding,
)
from flight_delay.models.gru_training import evaluate_logits, sigmoid  # noqa: E402
from flight_delay.models.hybrid import (  # noqa: E402
    build_hybrid_feature_matrix,
    extract_neural_representations,
    get_default_xgboost_params,
    train_hybrid_xgboost,
    train_plain_xgboost,
)
from flight_delay.models.mlp_training import MLPPreprocessor, set_torch_seed  # noqa: E402
from flight_delay.models.persistence import save_model  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]
MODEL_DIR = ROOT / "models/phase2/member4"
RESULT_DIR = ROOT / "reports/results/phase2_member4"
FIGURE_DIR = ROOT / "reports/figures/phase2_member4"
BASELINE_PATH = ROOT / "reports/results/phase2_member3/d307/sep_oct_predictions.parquet"


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "pr_auc": pr_auc(y_true, y_prob),
        "roc_auc": roc_auc(y_true, y_prob),
        "brier_score": brier(y_true, y_prob),
        "ece_10": expected_calibration_error(y_true, y_prob, n_bins=10),
        "mean_probability": float(np.mean(y_prob)),
    }


def make_embedding_data(frame: pd.DataFrame, prep: EmbeddingPreprocessor) -> EmbeddingData:
    num, cat = prep.transform(frame)
    return EmbeddingData(num, cat, frame["is_delayed"].to_numpy(np.float32))


def plot_ablation_comparison(comparison_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    models = comparison_df["model"].tolist()
    praucs = comparison_df["pr_auc_mean"].tolist()
    stds = comparison_df["pr_auc_std"].tolist()

    y_pos = np.arange(len(models))
    colors = ["#1f77b4" if "Ablation" in m else "#2ca02c" if "Hybrid" in m else "#7f7f7f" for m in models]

    plt.barh(y_pos, praucs, xerr=stds, align="center", color=colors, alpha=0.85, capsize=4)
    plt.yticks(y_pos, models, fontsize=10)
    plt.xlabel("September-October PR-AUC (mean ± sd)", fontsize=11)
    plt.title("Phase 2 Model & Ablation Comparison (Sep-Oct)", fontsize=13, fontweight="bold")
    plt.xlim(0.24, 0.32)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_bootstrap_differences(bootstrap_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    comps = bootstrap_df["comparison"].tolist()
    diffs = bootstrap_df["difference"].tolist()
    ci_lows = bootstrap_df["ci_low"].tolist()
    ci_highs = bootstrap_df["ci_high"].tolist()

    y_pos = np.arange(len(comps))
    xerr = [np.array(diffs) - np.array(ci_lows), np.array(ci_highs) - np.array(diffs)]

    plt.errorbar(diffs, y_pos, xerr=xerr, fmt="o", color="#d62728", ecolor="#1f77b4", elinewidth=2, capsize=5)
    plt.axvline(0.0, color="black", linestyle="--", alpha=0.7)
    plt.yticks(y_pos, comps, fontsize=10)
    plt.xlabel("Paired PR-AUC Difference (95% Day-Bootstrap CI)", fontsize=11)
    plt.title("Member 4 Paired PR-AUC Differences", fontsize=13, fontweight="bold")
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/config.yaml"))
    args = parser.parse_args()

    start_time = time.time()
    cfg = load_config(args.config)
    print("Loading Jan-Oct development dataset...", flush=True)
    raw = load_dev_raw(cfg, ROOT)
    table = build_feature_table(raw, cfg)

    # Validate split counts
    fit_mask = table["date"] <= "2013-07-31"
    tune_mask = table["date"].between("2013-08-01", "2013-08-31")
    train_mask = table["date"] <= "2013-08-31"
    eval_mask = table["date"].between("2013-09-01", "2013-10-31")

    fit_df = table.loc[fit_mask].reset_index(drop=True)
    tune_df = table.loc[tune_mask].reset_index(drop=True)
    train_df = table.loc[train_mask].reset_index(drop=True)
    eval_df = table.loc[eval_mask].reset_index(drop=True)

    print(
        f"Data splits: Fit={len(fit_df):,}, Tune={len(tune_df):,}, "
        f"Refit(Jan-Aug)={len(train_df):,}, Eval(Sep-Oct)={len(eval_df):,}",
        flush=True,
    )
    if (len(fit_df), len(tune_df), len(eval_df)) != (188971, 28756, 55628):
        raise ValueError("Flight counts do not match pre-registered protocol")

    # Ensure output directories exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    y_eval = eval_df["is_delayed"].to_numpy(int)
    eval_dates = pd.to_datetime(eval_df["date"]).dt.strftime("%Y-%m-%d").to_numpy()

    # Load baseline predictions (LR, M1 MLP, M2 GRU, M3 D-307)
    baseline_df = pd.read_parquet(BASELINE_PATH)
    if len(baseline_df) != len(eval_df) or not (baseline_df["id"].to_numpy() == eval_df["id"].to_numpy()).all():
        raise ValueError("Evaluation IDs do not align with Member 3 baseline predictions")

    prediction_records = baseline_df[["id", "date", "is_delayed", "lr"]].copy()
    for s in SEEDS:
        prediction_records[f"mlp_seed{s}"] = baseline_df[f"mlp_seed{s}"]
        prediction_records[f"gru_seed{s}"] = baseline_df[f"gru_seed{s}"]
        prediction_records[f"embedding_seed{s}"] = baseline_df[f"embedding_seed{s}"]

    selection_records = []
    per_seed_records = []

    # =========================================================================
    # 1. ABLATION 1: No Historical Features (20 numeric + 10 categorical)
    # =========================================================================
    print("\n" + "=" * 70)
    print("RUNNING ABLATION 1: NO HISTORICAL FEATURES")
    print("=" * 70, flush=True)

    num_no_hist = get_ablation_numeric_features(cfg, "no_history")
    prep_a1_fit = EmbeddingPreprocessor(num_no_hist, list(CATEGORICAL)).fit(fit_df)
    data_a1_fit = make_embedding_data(fit_df, prep_a1_fit)
    data_a1_tune = make_embedding_data(tune_df, prep_a1_fit)

    prep_a1_refit = EmbeddingPreprocessor(num_no_hist, list(CATEGORICAL)).fit(train_df)
    data_a1_train = make_embedding_data(train_df, prep_a1_refit)
    data_a1_eval = make_embedding_data(eval_df, prep_a1_refit)
    prep_a1_refit.save(MODEL_DIR / "ablation_no_history_preprocessor.joblib")

    a1_settings = {
        "learning_rate": 0.0001,
        "hidden_dims": [128, 64],
        "embedding_cap": 4,
        "dropout": 0.2,
        "batch_size": 1024,
    }

    for s in SEEDS:
        print(f"\n--- Ablation 1 (No History) | Seed {s} ---", flush=True)
        # Stage A: Tune on August
        _, _, best_epoch, best_metrics = train_embedding(
            data_a1_fit, data_a1_tune, prep_a1_fit.vocabulary_sizes, a1_settings, seed=s
        )
        selection_records.append(
            {
                "experiment": "ablation_no_history",
                "seed": s,
                "best_epoch": best_epoch,
                "august_pr_auc": best_metrics["pr_auc"],
                "august_roc_auc": best_metrics["roc_auc"],
            }
        )

        # Stage B: Refit on Jan-Aug
        model_a1, _, _, _ = train_embedding(
            data_a1_train,
            None,
            prep_a1_refit.vocabulary_sizes,
            a1_settings,
            seed=s,
            max_epochs=best_epoch,
        )
        torch.save(model_a1.state_dict(), MODEL_DIR / f"ablation_no_history_seed{s}.pth")

        # Evaluate on Sep-Oct
        eval_logits = predict_logits(model_a1, data_a1_eval)
        eval_probs = sigmoid(eval_logits)
        prediction_records[f"ablation_no_history_seed{s}"] = eval_probs

        metrics = compute_metrics(y_eval, eval_probs)
        per_seed_records.append({"experiment": "ablation_no_history", "seed": s, **metrics})
        print(
            f"  Ablation 1 (No History) Seed {s} -> Sep-Oct PR-AUC={metrics['pr_auc']:.5f}, "
            f"ROC-AUC={metrics['roc_auc']:.5f}, Brier={metrics['brier_score']:.5f}",
            flush=True,
        )

    # =========================================================================
    # 2. ABLATION 2: No Congestion Features (45 numeric + 10 categorical)
    # =========================================================================
    print("\n" + "=" * 70)
    print("RUNNING ABLATION 2: NO CONGESTION FEATURES")
    print("=" * 70, flush=True)

    num_no_cong = get_ablation_numeric_features(cfg, "no_congestion")
    prep_a2_fit = EmbeddingPreprocessor(num_no_cong, list(CATEGORICAL)).fit(fit_df)
    data_a2_fit = make_embedding_data(fit_df, prep_a2_fit)
    data_a2_tune = make_embedding_data(tune_df, prep_a2_fit)

    prep_a2_refit = EmbeddingPreprocessor(num_no_cong, list(CATEGORICAL)).fit(train_df)
    data_a2_train = make_embedding_data(train_df, prep_a2_refit)
    data_a2_eval = make_embedding_data(eval_df, prep_a2_refit)
    prep_a2_refit.save(MODEL_DIR / "ablation_no_congestion_preprocessor.joblib")

    a2_settings = a1_settings.copy()

    for s in SEEDS:
        print(f"\n--- Ablation 2 (No Congestion) | Seed {s} ---", flush=True)
        _, _, best_epoch, best_metrics = train_embedding(
            data_a2_fit, data_a2_tune, prep_a2_fit.vocabulary_sizes, a2_settings, seed=s
        )
        selection_records.append(
            {
                "experiment": "ablation_no_congestion",
                "seed": s,
                "best_epoch": best_epoch,
                "august_pr_auc": best_metrics["pr_auc"],
                "august_roc_auc": best_metrics["roc_auc"],
            }
        )

        model_a2, _, _, _ = train_embedding(
            data_a2_train,
            None,
            prep_a2_refit.vocabulary_sizes,
            a2_settings,
            seed=s,
            max_epochs=best_epoch,
        )
        torch.save(model_a2.state_dict(), MODEL_DIR / f"ablation_no_congestion_seed{s}.pth")

        eval_logits = predict_logits(model_a2, data_a2_eval)
        eval_probs = sigmoid(eval_logits)
        prediction_records[f"ablation_no_congestion_seed{s}"] = eval_probs

        metrics = compute_metrics(y_eval, eval_probs)
        per_seed_records.append({"experiment": "ablation_no_congestion", "seed": s, **metrics})
        print(
            f"  Ablation 2 (No Congestion) Seed {s} -> Sep-Oct PR-AUC={metrics['pr_auc']:.5f}, "
            f"ROC-AUC={metrics['roc_auc']:.5f}, Brier={metrics['brier_score']:.5f}",
            flush=True,
        )

    # =========================================================================
    # 3. ABLATION 3: One-Hot Instead of Embeddings (1,015 inputs)
    # =========================================================================
    print("\n" + "=" * 70)
    print("RUNNING ABLATION 3: ONE-HOT ENCODING INSTEAD OF EMBEDDINGS")
    print("=" * 70, flush=True)

    all_num = numeric_features(cfg)
    prep_a3_fit = MLPPreprocessor(all_num, list(CATEGORICAL)).fit(fit_df)
    X_a3_fit = prep_a3_fit.transform(fit_df)
    X_a3_tune = prep_a3_fit.transform(tune_df)
    y_fit = fit_df["is_delayed"].to_numpy(np.float32)
    y_tune = tune_df["is_delayed"].to_numpy(np.float32)

    prep_a3_refit = MLPPreprocessor(all_num, list(CATEGORICAL)).fit(train_df)
    X_a3_train = prep_a3_refit.transform(train_df)
    X_a3_eval = prep_a3_refit.transform(eval_df)
    y_train = train_df["is_delayed"].to_numpy(np.float32)
    prep_a3_refit.save(MODEL_DIR / "ablation_one_hot_preprocessor.joblib")

    for s in SEEDS:
        print(f"\n--- Ablation 3 (One-Hot) | Seed {s} ---", flush=True)
        _, _, best_epoch, best_metrics = train_one_hot_ablation(
            X_a3_fit,
            y_fit,
            X_a3_tune,
            y_tune,
            seed=s,
            learning_rate=0.0001,
            hidden_dims=(128, 64),
            dropout=0.2,
            weight_decay=0.0001,
            batch_size=1024,
            max_epochs=30,
            patience=5,
        )
        selection_records.append(
            {
                "experiment": "ablation_one_hot",
                "seed": s,
                "best_epoch": best_epoch,
                "august_pr_auc": best_metrics["pr_auc"],
                "august_roc_auc": best_metrics["roc_auc"],
            }
        )

        model_a3, _, _, _ = train_one_hot_ablation(
            X_a3_train,
            y_train,
            None,
            None,
            seed=s,
            learning_rate=0.0001,
            hidden_dims=(128, 64),
            dropout=0.2,
            weight_decay=0.0001,
            batch_size=1024,
            max_epochs=best_epoch,
        )
        torch.save(model_a3.state_dict(), MODEL_DIR / f"ablation_one_hot_seed{s}.pth")

        eval_logits = predict_one_hot_logits(model_a3, X_a3_eval)
        eval_probs = sigmoid(eval_logits)
        prediction_records[f"ablation_one_hot_seed{s}"] = eval_probs

        metrics = compute_metrics(y_eval, eval_probs)
        per_seed_records.append({"experiment": "ablation_one_hot", "seed": s, **metrics})
        print(
            f"  Ablation 3 (One-Hot) Seed {s} -> Sep-Oct PR-AUC={metrics['pr_auc']:.5f}, "
            f"ROC-AUC={metrics['roc_auc']:.5f}, Brier={metrics['brier_score']:.5f}",
            flush=True,
        )

    # =========================================================================
    # 4. ABLATION 4: No Regularization (Dropout=0.0, Weight Decay=0.0)
    # =========================================================================
    print("\n" + "=" * 70)
    print("RUNNING ABLATION 4: NO REGULARIZATION (DROPOUT=0, WEIGHT_DECAY=0)")
    print("=" * 70, flush=True)

    prep_refit_std = EmbeddingPreprocessor(all_num, list(CATEGORICAL)).fit(train_df)
    prep_fit_std = EmbeddingPreprocessor(all_num, list(CATEGORICAL)).fit(fit_df)
    prep_refit_std.save(MODEL_DIR / "ablation_no_regularization_preprocessor.joblib")

    data_std_fit = make_embedding_data(fit_df, prep_fit_std)
    data_std_tune = make_embedding_data(tune_df, prep_fit_std)
    data_std_train = make_embedding_data(train_df, prep_refit_std)
    data_std_eval = make_embedding_data(eval_df, prep_refit_std)

    a4_settings = {
        "learning_rate": 0.0001,
        "hidden_dims": [128, 64],
        "embedding_cap": 4,
        "dropout": 0.0,
        "batch_size": 1024,
    }

    for s in SEEDS:
        print(f"\n--- Ablation 4 (No Regularization) | Seed {s} ---", flush=True)
        # Custom training with weight_decay=0.0
        set_torch_seed(s)
        model_a4_cand = EmbeddingMLP(
            data_std_fit.numeric.shape[1],
            prep_fit_std.vocabulary_sizes,
            a4_settings["embedding_cap"],
            tuple(a4_settings["hidden_dims"]),
            dropout=0.0,
        )
        opt_a4 = torch.optim.Adam(model_a4_cand.parameters(), lr=0.0001, weight_decay=0.0)
        crit = torch.nn.BCEWithLogitsLoss()
        gen = torch.Generator().manual_seed(s)
        ldr = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(data_std_fit.numeric),
                torch.from_numpy(data_std_fit.categories),
                torch.from_numpy(data_std_fit.labels).unsqueeze(-1),
            ),
            batch_size=1024,
            shuffle=True,
            generator=gen,
        )

        best_ep, best_pr, stale, best_st = 0, -1.0, 0, None
        for ep in range(1, 31):
            model_a4_cand.train()
            for n_b, c_b, l_b in ldr:
                opt_a4.zero_grad(set_to_none=True)
                loss = crit(model_a4_cand(n_b, c_b), l_b)
                loss.backward()
                opt_a4.step()

            m_tune = evaluate_logits(data_std_tune.labels, predict_logits(model_a4_cand, data_std_tune))
            if m_tune["pr_auc"] > best_pr:
                best_pr, best_ep, stale = m_tune["pr_auc"], ep, 0
                best_st = copy.deepcopy(model_a4_cand.state_dict())
            else:
                stale += 1
            print(
                f"  epoch {ep:02d}: loss={loss.item():.5f}, August PR-AUC={m_tune['pr_auc']:.5f}, best_epoch={best_ep}",
                flush=True,
            )
            if stale >= 5:
                break

        selection_records.append(
            {
                "experiment": "ablation_no_regularization",
                "seed": s,
                "best_epoch": best_ep,
                "august_pr_auc": best_pr,
                "august_roc_auc": m_tune["roc_auc"],
            }
        )

        # Refit on Jan-Aug
        set_torch_seed(s)
        model_a4_final = EmbeddingMLP(
            data_std_train.numeric.shape[1],
            prep_refit_std.vocabulary_sizes,
            a4_settings["embedding_cap"],
            tuple(a4_settings["hidden_dims"]),
            dropout=0.0,
        )
        opt_a4_f = torch.optim.Adam(model_a4_final.parameters(), lr=0.0001, weight_decay=0.0)
        gen_f = torch.Generator().manual_seed(s)
        ldr_f = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(data_std_train.numeric),
                torch.from_numpy(data_std_train.categories),
                torch.from_numpy(data_std_train.labels).unsqueeze(-1),
            ),
            batch_size=1024,
            shuffle=True,
            generator=gen_f,
        )

        for ep in range(1, best_ep + 1):
            model_a4_final.train()
            for n_b, c_b, l_b in ldr_f:
                opt_a4_f.zero_grad(set_to_none=True)
                loss = crit(model_a4_final(n_b, c_b), l_b)
                loss.backward()
                opt_a4_f.step()
            print(f"  refit epoch {ep:02d}: loss={loss.item():.5f}", flush=True)

        torch.save(model_a4_final.state_dict(), MODEL_DIR / f"ablation_no_regularization_seed{s}.pth")

        eval_logits = predict_logits(model_a4_final, data_std_eval)
        eval_probs = sigmoid(eval_logits)
        prediction_records[f"ablation_no_regularization_seed{s}"] = eval_probs

        metrics = compute_metrics(y_eval, eval_probs)
        per_seed_records.append({"experiment": "ablation_no_regularization", "seed": s, **metrics})
        print(
            f"  Ablation 4 (No Reg) Seed {s} -> Sep-Oct PR-AUC={metrics['pr_auc']:.5f}, "
            f"ROC-AUC={metrics['roc_auc']:.5f}, Brier={metrics['brier_score']:.5f}",
            flush=True,
        )

    # =========================================================================
    # 5. FRESH PLAIN XGBOOST BASELINE (D-201 Settings)
    # =========================================================================
    print("\n" + "=" * 70)
    print("RUNNING FRESH PLAIN XGBOOST BASELINE (D-201 SETTINGS)")
    print("=" * 70, flush=True)

    # Encode with FeatureEncoder
    enc_fit = FeatureEncoder(list(CATEGORICAL), all_num, min_count=cfg["features"]["categorical_min_count"]).fit(fit_df)
    features_59 = all_num + list(CATEGORICAL)

    X_xgb_fit = enc_fit.transform(fit_df)[features_59]
    X_xgb_tune = enc_fit.transform(tune_df)[features_59]

    enc_refit = FeatureEncoder(list(CATEGORICAL), all_num, min_count=cfg["features"]["categorical_min_count"]).fit(train_df)
    enc_refit.save(MODEL_DIR / "plain_xgboost_encoder.json")
    X_xgb_train = enc_refit.transform(train_df)[features_59]
    X_xgb_eval = enc_refit.transform(eval_df)[features_59]

    for s in SEEDS:
        print(f"\n--- Plain XGBoost Baseline | Seed {s} ---", flush=True)
        xgb_params = get_default_xgboost_params(
            max_depth=5,
            learning_rate=0.10,
            n_estimators=1000,
            early_stopping_rounds=50,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1.0,
            seed=s,
            enable_categorical=True,
        )
        xgb_cand, effective_trees = train_plain_xgboost(X_xgb_fit, y_fit, X_xgb_tune, y_tune, xgb_params)
        tune_probs = xgb_cand.predict_proba(X_xgb_tune)[:, 1]
        m_tune = compute_metrics(y_tune, tune_probs)

        selection_records.append(
            {
                "experiment": "plain_xgboost",
                "seed": s,
                "effective_trees": effective_trees,
                "august_pr_auc": m_tune["pr_auc"],
                "august_roc_auc": m_tune["roc_auc"],
            }
        )

        # Refit on Jan-Aug
        xgb_final_params = get_default_xgboost_params(
            max_depth=5,
            learning_rate=0.10,
            n_estimators=effective_trees,
            early_stopping_rounds=None,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1.0,
            seed=s,
            enable_categorical=True,
        )
        xgb_final, _ = train_plain_xgboost(X_xgb_train, y_train, None, None, xgb_final_params)
        save_model(xgb_final, MODEL_DIR / f"plain_xgboost_seed{s}.joblib")

        eval_probs = xgb_final.predict_proba(X_xgb_eval)[:, 1]
        prediction_records[f"plain_xgboost_seed{s}"] = eval_probs

        metrics = compute_metrics(y_eval, eval_probs)
        per_seed_records.append({"experiment": "plain_xgboost", "seed": s, **metrics})
        print(
            f"  Plain XGBoost Seed {s} -> Trees={effective_trees}, Sep-Oct PR-AUC={metrics['pr_auc']:.5f}, "
            f"ROC-AUC={metrics['roc_auc']:.5f}, Brier={metrics['brier_score']:.5f}",
            flush=True,
        )

    # =========================================================================
    # 6. HYBRID EMBEDDING MLP + XGBOOST
    # =========================================================================
    print("\n" + "=" * 70)
    print("RUNNING HYBRID EMBEDDING MLP + XGBOOST")
    print("=" * 70, flush=True)

    d307_settings = {
        "learning_rate": 0.0001,
        "hidden_dims": [128, 64],
        "embedding_cap": 4,
        "dropout": 0.2,
        "batch_size": 1024,
    }
    prep_refit_std.save(MODEL_DIR / "hybrid_preprocessor.joblib")

    for s in SEEDS:
        print(f"\n--- Hybrid DL + XGBoost | Seed {s} ---", flush=True)

        # Stage A: Train Candidate Neural Model on Jan-Jul, early stopping on Aug
        print("  Training Candidate Embedding MLP on Jan-Jul...", flush=True)
        nn_cand, _, nn_best_ep, _ = train_embedding(
            data_std_fit, data_std_tune, prep_fit_std.vocabulary_sizes, d307_settings, seed=s
        )

        # Extract 64-d representations from Candidate Neural Model
        rep_fit = extract_neural_representations(nn_cand, data_std_fit)
        rep_tune = extract_neural_representations(nn_cand, data_std_tune)

        Z_fit = build_hybrid_feature_matrix(data_std_fit.numeric, rep_fit)
        Z_tune = build_hybrid_feature_matrix(data_std_tune.numeric, rep_tune)

        # Tune XGBoost on August hybrid representations
        print(f"  Tuning Hybrid XGBoost on August (Input Dim={Z_fit.shape[1]})...", flush=True)
        xgb_h_params = get_default_xgboost_params(
            max_depth=5,
            learning_rate=0.10,
            n_estimators=1000,
            early_stopping_rounds=50,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1.0,
            seed=s,
            enable_categorical=False,
        )
        xgb_h_cand, h_effective_trees = train_hybrid_xgboost(Z_fit, y_fit, Z_tune, y_tune, xgb_h_params)
        m_tune_h = compute_metrics(y_tune, xgb_h_cand.predict_proba(Z_tune)[:, 1])

        selection_records.append(
            {
                "experiment": "hybrid_dl_xgboost",
                "seed": s,
                "nn_selected_epoch": nn_best_ep,
                "effective_trees": h_effective_trees,
                "august_pr_auc": m_tune_h["pr_auc"],
                "august_roc_auc": m_tune_h["roc_auc"],
            }
        )

        # Stage B: Train Final Neural Model on Jan-Aug
        print(f"  Refitting Neural Network on Jan-Aug for {nn_best_ep} epochs...", flush=True)
        nn_final, _, _, _ = train_embedding(
            data_std_train, None, prep_refit_std.vocabulary_sizes, d307_settings, seed=s, max_epochs=nn_best_ep
        )
        torch.save(nn_final.state_dict(), MODEL_DIR / f"hybrid_nn_seed{s}.pth")

        # Extract 64-d representations from Final Neural Model
        rep_train = extract_neural_representations(nn_final, data_std_train)
        rep_eval = extract_neural_representations(nn_final, data_std_eval)

        Z_train = build_hybrid_feature_matrix(data_std_train.numeric, rep_train)
        Z_eval = build_hybrid_feature_matrix(data_std_eval.numeric, rep_eval)

        # Retrain Hybrid XGBoost on Jan-Aug representations
        print(f"  Retraining Hybrid XGBoost for {h_effective_trees} trees...", flush=True)
        xgb_h_final_params = get_default_xgboost_params(
            max_depth=5,
            learning_rate=0.10,
            n_estimators=h_effective_trees,
            early_stopping_rounds=None,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1.0,
            seed=s,
            enable_categorical=False,
        )
        xgb_h_final, _ = train_hybrid_xgboost(Z_train, y_train, None, None, xgb_h_final_params)
        save_model(xgb_h_final, MODEL_DIR / f"hybrid_xgboost_seed{s}.joblib")

        # Evaluate Hybrid on Sep-Oct
        eval_probs = xgb_h_final.predict_proba(Z_eval)[:, 1]
        prediction_records[f"hybrid_dl_xgboost_seed{s}"] = eval_probs

        metrics = compute_metrics(y_eval, eval_probs)
        per_seed_records.append({"experiment": "hybrid_dl_xgboost", "seed": s, **metrics})
        print(
            f"  Hybrid DL+XGBoost Seed {s} -> Trees={h_effective_trees}, Sep-Oct PR-AUC={metrics['pr_auc']:.5f}, "
            f"ROC-AUC={metrics['roc_auc']:.5f}, Brier={metrics['brier_score']:.5f}",
            flush=True,
        )

    # Save all Sep-Oct predictions to Parquet
    prediction_records.to_parquet(RESULT_DIR / "sep_oct_predictions.parquet", index=False)
    print(f"\nSaved all aligned Sep-Oct predictions to {RESULT_DIR / 'sep_oct_predictions.parquet'}")

    # =========================================================================
    # 7. CONSOLIDATED SUMMARY & BOOTSTRAP SIGNIFICANCE TESTING
    # =========================================================================
    print("\n" + "=" * 70)
    print("CALCULATING CONSOLIDATED METRICS & BOOTSTRAP SIGNIFICANCE")
    print("=" * 70, flush=True)

    # Save per-seed metrics CSV
    per_seed_df = pd.DataFrame(per_seed_records)
    per_seed_df.to_csv(RESULT_DIR / "per_seed_metrics.csv", index=False)

    # Save selection records CSV
    selection_df = pd.DataFrame(selection_records)
    selection_df.to_csv(RESULT_DIR / "august_seed_selection.csv", index=False)

    # Model comparison table
    models_to_compare = [
        ("Logistic Regression (Phase 1)", "lr", None),
        ("Member 1 MLP (One-Hot)", "mlp", [f"mlp_seed{s}" for s in SEEDS]),
        ("Member 2 GRU (Sequences)", "gru", [f"gru_seed{s}" for s in SEEDS]),
        ("Member 3 Embedding MLP (D-307 Reference)", "embedding_d307", [f"embedding_seed{s}" for s in SEEDS]),
        ("Plain XGBoost Baseline", "plain_xgboost", [f"plain_xgboost_seed{s}" for s in SEEDS]),
        ("Ablation 1 (No Historical)", "ablation_no_history", [f"ablation_no_history_seed{s}" for s in SEEDS]),
        ("Ablation 2 (No Congestion)", "ablation_no_congestion", [f"ablation_no_congestion_seed{s}" for s in SEEDS]),
        ("Ablation 3 (One-Hot Encoded)", "ablation_one_hot", [f"ablation_one_hot_seed{s}" for s in SEEDS]),
        ("Ablation 4 (No Regularization)", "ablation_no_regularization", [f"ablation_no_regularization_seed{s}" for s in SEEDS]),
        ("Hybrid DL + XGBoost", "hybrid_dl_xgboost", [f"hybrid_dl_xgboost_seed{s}" for s in SEEDS]),
    ]

    comparison_rows = []
    probs_dict = {}

    for label, key, cols in models_to_compare:
        if cols is None:
            # Single baseline (LR)
            p = prediction_records[key].to_numpy()
            probs_dict[key] = [p]
            m = compute_metrics(y_eval, p)
            comparison_rows.append(
                {
                    "model": label,
                    "key": key,
                    "pr_auc_mean": m["pr_auc"],
                    "pr_auc_std": 0.0,
                    "roc_auc_mean": m["roc_auc"],
                    "roc_auc_std": 0.0,
                    "brier_score_mean": m["brier_score"],
                    "ece_10_mean": m["ece_10"],
                }
            )
        else:
            seed_probs = [prediction_records[c].to_numpy() for c in cols]
            probs_dict[key] = seed_probs
            seed_praucs = [pr_auc(y_eval, p) for p in seed_probs]
            seed_rocs = [roc_auc(y_eval, p) for p in seed_probs]
            seed_briers = [brier(y_eval, p) for p in seed_probs]
            seed_eces = [expected_calibration_error(y_eval, p, n_bins=10) for p in seed_probs]

            comparison_rows.append(
                {
                    "model": label,
                    "key": key,
                    "pr_auc_mean": float(np.mean(seed_praucs)),
                    "pr_auc_std": float(np.std(seed_praucs, ddof=1)),
                    "roc_auc_mean": float(np.mean(seed_rocs)),
                    "roc_auc_std": float(np.std(seed_rocs, ddof=1)),
                    "brier_score_mean": float(np.mean(seed_briers)),
                    "ece_10_mean": float(np.mean(seed_eces)),
                }
            )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(RESULT_DIR / "comparison.csv", index=False)
    print("\nSummary Comparison Table (September-October 2013):")
    print(comparison_df.to_string(index=False))

    # Day-level block bootstrap (1,000 resamples)
    print("\nRunning Day-Level Block Bootstrap (1,000 resamples)...", flush=True)
    pairs = [
        ("hybrid_dl_xgboost", "plain_xgboost"),
        ("hybrid_dl_xgboost", "embedding_d307"),
        ("hybrid_dl_xgboost", "lr"),
        ("hybrid_dl_xgboost", "mlp"),
        ("hybrid_dl_xgboost", "gru"),
        ("ablation_no_history", "embedding_d307"),
        ("ablation_no_congestion", "embedding_d307"),
        ("ablation_one_hot", "embedding_d307"),
        ("ablation_no_regularization", "embedding_d307"),
        ("plain_xgboost", "lr"),
        ("plain_xgboost", "embedding_d307"),
    ]

    bootstrap_df = day_bootstrap(eval_dates, y_eval, probs_dict, pairs, seed=42)
    bootstrap_df.to_csv(RESULT_DIR / "bootstrap_pr_auc_differences.csv", index=False)
    print("\nBootstrap Significance Results:")
    print(bootstrap_df.to_string(index=False))

    # Generate charts
    plot_ablation_comparison(comparison_df, FIGURE_DIR / "ablation_comparison.png")
    plot_bootstrap_differences(bootstrap_df, FIGURE_DIR / "bootstrap_differences.png")
    print(f"\nGenerated charts in {FIGURE_DIR}")

    total_time = time.time() - start_time
    print(f"\nMember 4 pipeline completed successfully in {total_time:.1f}s ({total_time / 60:.2f} min).")


if __name__ == "__main__":
    main()
