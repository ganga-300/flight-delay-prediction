"""Member 2: GRU over ordered daily delay sequences, under the pre-registered D-303 design.

Protocol (D-301, same as Phase 1):
1. Grid (GRU, seed 42): fit January-July, early-stop on August PR-AUC, for L in {7, 14, 28} and
   hidden size h in {16, 32}. Highest August PR-AUC wins; ties within 0.001 go to the smaller.
2. Final runs, seeds 42-46: the selected GRU, the control (same inputs, no recurrence), the
   Member 1 MLP, and the diagnostic GRU without the 29 historical features. Each seed chooses
   its epoch on August, is refit on January-August for that many epochs, and scores
   September-October once.
3. September-October comparison with the Phase 1 Logistic Regression on the same flights,
   overall and per month, with day-level bootstrap intervals for PR-AUC differences.

Static features use Member 1's ``MLPPreprocessor``; sequence scaling uses ``SequenceScaler``.
Both are fitted on each stage's training rows only. The sealed November-December file is never
read. Refuses to overwrite existing outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flight_delay.config import load_config  # noqa: E402
from flight_delay.data.clean import build_labelled_frame  # noqa: E402
from flight_delay.data.load import load_dev_raw  # noqa: E402
from flight_delay.data.schema import TARGET_COLUMN  # noqa: E402
from flight_delay.features.build import (  # noqa: E402
    CATEGORICAL,
    build_feature_table,
    feature_groups,
    numeric_features,
)
from flight_delay.features.encoding import FeatureEncoder  # noqa: E402
from flight_delay.features.sequences import (  # noqa: E402
    SEQUENCE_CHANNELS,
    SequenceScaler,
    build_sequence_tables,
)
from flight_delay.models.gru_training import (  # noqa: E402
    SequenceData,
    evaluate_logits,
    predict_logits,
    sigmoid,
    train_sequence_model,
)
from flight_delay.models.mlp_training import (  # noqa: E402
    MLPPreprocessor,
    train_mlp_baseline,
)
from flight_delay.models.persistence import load_model  # noqa: E402

FIT = ("2013-01-01", "2013-07-31")
TUNE = ("2013-08-01", "2013-08-31")
EVAL = ("2013-09-01", "2013-10-31")
MONTHS = {"september": ("2013-09-01", "2013-09-30"), "october": ("2013-10-01", "2013-10-31")}
GRID_LENGTHS = [7, 14, 28]
GRID_HIDDEN = [16, 32]
SEEDS = [42, 43, 44, 45, 46]
TIE = 0.001
N_BOOT = 1000
LR_MODEL = ROOT / "models" / "d201" / "final_model.joblib"
LR_ENCODER = ROOT / "models" / "d201" / "final_encoder.json"

MODEL_DIR = ROOT / "models" / "phase2" / "member2"
FIG_DIR = ROOT / "reports" / "figures" / "phase2_member2"
RES_DIR = ROOT / "reports" / "results" / "phase2_member2"

LABELS = {
    "gru": "GRU (sequence, in order)",
    "control": "Control (same inputs, no recurrence)",
    "mlp": "Member 1 MLP (no sequence)",
    "gru_no_history": "GRU without the 29 historical features",
    "lr": "Logistic Regression (Phase 1)",
}


def period(table: pd.DataFrame, bounds: tuple[str, str]) -> np.ndarray:
    return pd.to_datetime(table["date"]).between(*bounds).to_numpy()


class Stage:
    """Training rows of one stage plus the rows it is scored on; caches fitted preprocessing."""

    def __init__(self, table, tables, train_mask, score_mask, cfg):
        self.table, self.tables, self.cfg = table, tables, cfg
        self.train = table.loc[train_mask].reset_index(drop=True)
        self.score = table.loc[score_mask].reset_index(drop=True)
        self.idx_train = tables.flight_indices(self.train)
        self.idx_score = tables.flight_indices(self.score)
        self._static: dict[str, tuple] = {}
        self._seq: dict[int, list[np.ndarray]] = {}

    def static(self, name: str, numeric: list[str]):
        if name not in self._static:
            prep = MLPPreprocessor(
                numeric, list(CATEGORICAL), min_count=self.cfg["features"]["categorical_min_count"]
            )
            self._static[name] = (prep, prep.fit_transform(self.train), prep.transform(self.score))
        return self._static[name]

    def scaled_tables(self, length: int):
        if length not in self._seq:
            scaler = SequenceScaler.fit(self.tables, self.idx_train, length)
            self._seq[length] = (scaler, scaler.transform_tables(self.tables))
        return self._seq[length]

    def data(self, static_name: str, numeric: list[str], length: int):
        _, x_train, x_score = self.static(static_name, numeric)
        _, scaled = self.scaled_tables(length)
        y_tr = self.train[TARGET_COLUMN].to_numpy(np.float32)
        y_sc = self.score[TARGET_COLUMN].to_numpy(np.float32)
        return (
            SequenceData(x_train, self.idx_train, y_tr, scaled, length),
            SequenceData(x_score, self.idx_score, y_sc, scaled, length),
        )


def plot_architecture(length: int, hidden: int, static_dim: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis("off")
    boxes = [
        (
            0.08,
            0.72,
            f"Daily sequence\n{length} days x {len(SEQUENCE_CHANNELS)} values\n"
            "(D-L ... D-1, oldest first)",
            "#d9eaf7",
        ),
        (
            0.08,
            0.25,
            f"Flight features\n59 Phase 1 features\n({static_dim} after one-hot)",
            "#e5f5e0",
        ),
        (0.36, 0.72, f"GRU, 1 layer\nhidden {hidden}\nfinal state", "#9ecae1"),
        (0.6, 0.5, f"Concatenate\n{static_dim} + {hidden}", "#fdd0a2"),
        (
            0.88,
            0.5,
            "Linear(64) -> ReLU\nDropout(0.2)\nLinear(1) -> sigmoid\nP(delay >= 15 min)",
            "#c7e9c0",
        ),
    ]
    for x, y, text, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x - 0.1, y - 0.12),
                0.2,
                0.24,
                boxstyle="round,pad=0.02",
                fc=color,
                ec="#333",
                lw=1.2,
            )
        )
        ax.text(x, y, text, ha="center", va="center", fontsize=9)
    arrows = [
        ((0.18, 0.72), (0.26, 0.72)),
        ((0.46, 0.72), (0.5, 0.6)),
        ((0.18, 0.25), (0.5, 0.4)),
        ((0.7, 0.5), (0.78, 0.5)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.set_xlim(-0.05, 1.03)
    ax.set_ylim(0, 1)
    ax.set_title(
        "Member 2 GRU model (control: GRU replaced by Flatten -> Linear -> ReLU)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_training(history: dict, best_epoch: int, path: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.5))
    e = history["epoch"]
    a.plot(e, history["train_loss"], marker="o", label="training loss (Jan-Jul)")
    a.plot(e, history["valid_loss"], marker="o", label="tuning loss (August)")
    a.set_xlabel("epoch")
    a.set_ylabel("BCE loss")
    a.legend()
    a.grid(alpha=0.4)
    b.plot(e, history["valid_pr_auc"], marker="o", color="#2ca02c")
    b.axvline(best_epoch, color="#d62728", ls="--", label=f"selected epoch {best_epoch}")
    b.set_xlabel("epoch")
    b.set_ylabel("August PR-AUC")
    b.legend()
    b.grid(alpha=0.4)
    fig.suptitle("GRU training curves (seed 42, Stage A)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_grid(grid: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for h, part in grid.groupby("hidden_size"):
        ax.plot(part["length"], part["august_pr_auc"], marker="o", label=f"hidden {h}")
    ax.axhline(0.4007, color="grey", ls="--", label="Phase 1 LR on August (0.4007)")
    ax.set_xticks(GRID_LENGTHS)
    ax.set_xlabel("sequence length L (days)")
    ax.set_ylabel("August PR-AUC")
    ax.set_title("GRU grid (seed 42)", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_seeds(per_seed: pd.DataFrame, lr_pr: float, path: Path) -> None:
    overall = per_seed[per_seed["period"] == "sep_oct"]
    order = ["mlp", "control", "gru", "gru_no_history"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, model in enumerate(order):
        vals = overall.loc[overall["model"] == model, "pr_auc"]
        ax.scatter(np.full(len(vals), i), vals, alpha=0.7)
        ax.errorbar(i + 0.15, vals.mean(), yerr=vals.std(ddof=1), fmt="D", color="black")
    ax.axhline(lr_pr, color="grey", ls="--", label=f"Phase 1 LR ({lr_pr:.3f})")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["Member 1 MLP", "Control", "GRU", "GRU, no history\nfeatures"])
    ax.set_ylabel("September-October PR-AUC")
    ax.set_title("Five seeds per model (dots) and mean +/- sd (diamond)", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.4, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_pr_curves(y: np.ndarray, probs: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, p in probs.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, label=f"{LABELS[name]} (AP {average_precision_score(y, p):.3f})")
    ax.axhline(y.mean(), color="grey", ls=":", label=f"delay rate {y.mean():.3f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("September-October precision-recall (seed 42)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def day_bootstrap(dates: np.ndarray, y: np.ndarray, probs: dict, pairs: list, seed: int = 42):
    """95% intervals for PR-AUC differences, resampling whole days; each model's PR-AUC is
    the mean over its seeds."""
    days, inverse = np.unique(dates, return_inverse=True)
    rows_by_day = [np.flatnonzero(inverse == i) for i in range(len(days))]
    rng = np.random.default_rng(seed)

    def mean_ap(name, rows):
        return np.mean([average_precision_score(y[rows], p[rows]) for p in probs[name]])

    point = {name: mean_ap(name, np.arange(len(y))) for name in probs}
    draws = {pair: [] for pair in pairs}
    for _ in range(N_BOOT):
        rows = np.concatenate([rows_by_day[i] for i in rng.integers(0, len(days), len(days))])
        ap = {name: mean_ap(name, rows) for name in probs}
        for a, b in pairs:
            draws[(a, b)].append(ap[a] - ap[b])
    out = []
    for a, b in pairs:
        lo, hi = np.percentile(draws[(a, b)], [2.5, 97.5])
        out.append(
            {
                "comparison": f"{a} - {b}",
                "difference": point[a] - point[b],
                "ci_low": lo,
                "ci_high": hi,
                "excludes_zero": bool(lo > 0 or hi < 0),
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny run for checking the pipeline; writes to a scratch folder",
    )
    parser.add_argument("--out", default=None, help="output root for --smoke runs")
    args = parser.parse_args()

    global MODEL_DIR, FIG_DIR, RES_DIR, SEEDS, GRID_LENGTHS, GRID_HIDDEN, N_BOOT
    if args.smoke:
        out = Path(args.out or "/tmp/gru_smoke")
        MODEL_DIR, FIG_DIR, RES_DIR = out / "models", out / "figures", out / "results"
        SEEDS, GRID_LENGTHS, GRID_HIDDEN, N_BOOT = [42, 43], [7], [16], 20
    if (MODEL_DIR / "gru_final.pth").exists() or (RES_DIR / "comparison.csv").exists():
        raise SystemExit("D-304 outputs already exist; refusing to overwrite")
    for d in (MODEL_DIR, FIG_DIR, RES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cfg = load_config(args.config)
    device = torch.device(args.device)
    numeric_all = numeric_features(cfg)
    historical = feature_groups(cfg)["historical"]
    numeric_no_hist = [c for c in numeric_all if c not in set(historical)]
    if len(numeric_all) != 49 or len(historical) != 29:
        raise SystemExit("expected the frozen 49 numeric features, 29 of them historical")

    print("Building features and daily sequence tables (January-October only)...", flush=True)
    raw = load_dev_raw(cfg, ROOT)
    table = build_feature_table(raw, cfg)
    labelled = build_labelled_frame(
        raw, cfg["target"]["threshold_minutes"], cfg["target"]["cancelled_policy"]
    )
    labelled[TARGET_COLUMN] = labelled[TARGET_COLUMN].astype(int)
    tables = build_sequence_tables(
        labelled,
        table["date"].max(),
        max_len=max(GRID_LENGTHS),
        prior_strength=cfg["features"]["history_prior_strength"],
    )
    if args.smoke:
        keep = table["date"].dt.day <= 3  # three days per month
        table = table.loc[keep].reset_index(drop=True)

    stage_a = Stage(table, tables, period(table, FIT), period(table, TUNE), cfg)
    stage_b = Stage(
        table, tables, period(table, FIT) | period(table, TUNE), period(table, EVAL), cfg
    )
    common = dict(device=device, max_epochs=args.max_epochs, patience=args.patience)

    # 1. Grid on August (GRU, seed 42).
    grid_rows, grid_hist = [], {}
    for length in GRID_LENGTHS:
        for hidden in GRID_HIDDEN:
            print(f"\nGrid: L={length}, hidden={hidden}", flush=True)
            tr, tu = stage_a.data("all", numeric_all, length)
            _, hist, best_epoch, m = train_sequence_model(tr, tu, "gru", hidden, seed=42, **common)
            grid_rows.append(
                {
                    "length": length,
                    "hidden_size": hidden,
                    "best_epoch": best_epoch,
                    "august_pr_auc": m["pr_auc"],
                    "august_roc_auc": m["roc_auc"],
                }
            )
            grid_hist[f"L{length}_h{hidden}"] = hist
    grid = pd.DataFrame(grid_rows)
    top = grid["august_pr_auc"].max()
    chosen = grid[grid["august_pr_auc"] >= top - TIE].sort_values(["length", "hidden_size"]).iloc[0]
    length, hidden = int(chosen["length"]), int(chosen["hidden_size"])
    grid["selected"] = (grid["length"] == length) & (grid["hidden_size"] == hidden)
    grid.to_csv(RES_DIR / "grid_august.csv", index=False)
    print(f"\nSelected L={length}, hidden={hidden} (August PR-AUC {chosen['august_pr_auc']:.4f})")

    # 2. Final runs: each seed picks its epoch on August, then refits on January-August.
    eval_rows = stage_b.score
    y_eval = eval_rows[TARGET_COLUMN].to_numpy(np.float32)
    probs: dict[str, list[np.ndarray]] = {
        m: [] for m in ["gru", "control", "mlp", "gru_no_history"]
    }
    runs, histories, saved = [], {}, {}
    specs = [
        ("gru", "gru", "all", numeric_all),
        ("control", "flat", "all", numeric_all),
        ("gru_no_history", "gru", "no_hist", numeric_no_hist),
        ("mlp", None, "all", numeric_all),
    ]
    for name, encoder, static_name, numeric in specs:
        for seed in SEEDS:
            print(f"\n{name}, seed {seed}: Stage A (Jan-Jul, epoch chosen on August)", flush=True)
            if encoder is None:  # Member 1 MLP, unchanged
                _, x_fit, x_tune = stage_a.static(static_name, numeric)
                _, hist, best_epoch, aug = train_mlp_baseline(
                    x_fit,
                    stage_a.train[TARGET_COLUMN].to_numpy(np.float32),
                    x_tune,
                    stage_a.score[TARGET_COLUMN].to_numpy(np.float32),
                    device=device,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    seed=seed,
                )
                prep, x_final, x_eval = stage_b.static(static_name, numeric)
                print(f"{name}, seed {seed}: Stage B ({best_epoch} epoch(s) on Jan-Aug)")
                model, _, _, _ = train_mlp_baseline(
                    x_final,
                    stage_b.train[TARGET_COLUMN].to_numpy(np.float32),
                    None,
                    None,
                    device=device,
                    max_epochs=best_epoch,
                    patience=best_epoch,
                    seed=seed,
                )
                p = _mlp_probs(model, x_eval, device)
            else:
                tr, tu = stage_a.data(static_name, numeric, length)
                _, hist, best_epoch, aug = train_sequence_model(
                    tr, tu, encoder, hidden, seed=seed, **common
                )
                fin, ev = stage_b.data(static_name, numeric, length)
                print(f"{name}, seed {seed}: Stage B ({best_epoch} epoch(s) on Jan-Aug)")
                model, refit_hist, _, _ = train_sequence_model(
                    fin,
                    None,
                    encoder,
                    hidden,
                    device=device,
                    max_epochs=best_epoch,
                    patience=best_epoch,
                    seed=seed,
                    verbose=False,
                )
                p = sigmoid(predict_logits(model, ev, device))
                if name == "gru" and seed == 42:
                    saved = {"model": model, "refit_history": refit_hist}
            probs[name].append(p)
            histories[f"{name}_seed{seed}"] = hist
            runs.append(
                {
                    "model": name,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "august_pr_auc": aug["pr_auc"],
                    "august_roc_auc": aug["roc_auc"],
                }
            )

    # 3. Phase 1 Logistic Regression on the same flights.
    lr_enc = FeatureEncoder.load(LR_ENCODER)
    lr_p = load_model(LR_MODEL).predict_proba(
        lr_enc.transform(eval_rows)[numeric_all + list(CATEGORICAL)]
    )[:, 1]

    month_mask = {"sep_oct": np.ones(len(eval_rows), bool)}
    for m, b in MONTHS.items():
        month_mask[m] = period(eval_rows, b)
    rows = []
    for name, seeds_p in probs.items():
        for seed, p in zip(SEEDS, seeds_p, strict=True):
            for per, mask in month_mask.items():
                rows.append(
                    {
                        "model": name,
                        "seed": seed,
                        "period": per,
                        **evaluate_logits(y_eval[mask], to_logit(p[mask])),
                    }
                )
    for per, mask in month_mask.items():
        rows.append(
            {
                "model": "lr",
                "seed": None,
                "period": per,
                **evaluate_logits(y_eval[mask], to_logit(lr_p[mask])),
            }
        )
    per_seed = pd.DataFrame(rows)
    per_seed["seed"] = per_seed["seed"].astype("Int64")
    metric_cols = ["pr_auc", "roc_auc", "brier_score", "ece_10", "mean_predicted_probability"]
    summary = per_seed.groupby(["model", "period"])[metric_cols].agg(["mean", "std"]).round(4)
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary["label"] = summary["model"].map(LABELS)

    print("\nBootstrapping PR-AUC differences by day...", flush=True)
    boot_probs = {**{k: v for k, v in probs.items()}, "lr": [lr_p]}
    pairs = [
        ("gru", "lr"),
        ("gru", "control"),
        ("gru", "mlp"),
        ("control", "lr"),
        ("mlp", "lr"),
        ("gru_no_history", "gru"),
    ]
    boot = day_bootstrap(eval_rows["date"].to_numpy(), y_eval, boot_probs, pairs)

    # 4. Save artifacts and results.
    prep_b = stage_b.static("all", numeric_all)[0]
    scaler_b = stage_b.scaled_tables(length)[0]
    torch.save(saved["model"].state_dict(), MODEL_DIR / "gru_final.pth")
    joblib.dump(prep_b, MODEL_DIR / "static_preprocessor.joblib")
    np.savez(MODEL_DIR / "sequence_scaler.npz", mean=scaler_b.mean, std=scaler_b.std)
    config = {
        "protocol": "D-303",
        "selected_length": length,
        "selected_hidden_size": hidden,
        "sequence_channels": SEQUENCE_CHANNELS,
        "static_dim": int(prep_b.input_dim_),
        "head": "Linear(64) -> ReLU -> Dropout(0.2) -> Linear(1)",
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "batch_size": 256,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "seeds": SEEDS,
        "saved_model": "seed 42, refit January-August",
        "device": str(device),
        "torch_version": torch.__version__,
        "holdout_accessed": False,
    }
    (MODEL_DIR / "model_config.json").write_text(json.dumps(config, indent=2))
    (MODEL_DIR / "training_histories.json").write_text(
        json.dumps(
            {
                "grid": grid_hist,
                "final_runs": histories,
                "gru_seed42_refit": saved["refit_history"],
            },
            indent=2,
        )
    )
    pd.DataFrame(runs).to_csv(RES_DIR / "august_selection_runs.csv", index=False)
    per_seed.to_csv(RES_DIR / "per_seed_metrics.csv", index=False)
    summary.to_csv(RES_DIR / "comparison.csv", index=False)
    boot.to_csv(RES_DIR / "bootstrap_pr_auc_differences.csv", index=False)
    preds = pd.DataFrame(
        {
            "id": eval_rows["id"],
            "date": eval_rows["date"],
            TARGET_COLUMN: y_eval.astype(int),
            "lr": lr_p.astype(np.float32),
        }
    )
    for name, seeds_p in probs.items():
        for seed, p in zip(SEEDS, seeds_p, strict=True):
            preds[f"{name}_seed{seed}"] = p.astype(np.float32)
    preds.to_parquet(RES_DIR / "sep_oct_predictions.parquet", index=False)

    plot_architecture(length, hidden, int(prep_b.input_dim_), FIG_DIR / "gru_architecture.png")
    plot_training(
        histories["gru_seed42"],
        int(pd.DataFrame(runs).query("model == 'gru' and seed == 42")["best_epoch"].iloc[0]),
        FIG_DIR / "gru_training_curves.png",
    )
    plot_grid(grid, FIG_DIR / "grid_august_pr_auc.png")
    lr_overall = float(per_seed.query("model == 'lr' and period == 'sep_oct'")["pr_auc"].iloc[0])
    plot_seeds(per_seed, lr_overall, FIG_DIR / "seed_comparison.png")
    plot_pr_curves(
        y_eval,
        {
            "lr": lr_p,
            "mlp": probs["mlp"][0],
            "control": probs["control"][0],
            "gru": probs["gru"][0],
        },
        FIG_DIR / "pr_curves_sep_oct.png",
    )

    print("\n=== D-304: September-October (mean +/- sd over seeds) ===")
    view = summary[summary["period"] == "sep_oct"]
    for _, r in view.iterrows():
        print(
            f"{LABELS[r['model']]:45s} PR-AUC {r['pr_auc_mean']:.4f} +/- "
            f"{0 if pd.isna(r['pr_auc_std']) else r['pr_auc_std']:.4f}  "
            f"ROC {r['roc_auc_mean']:.4f}  Brier {r['brier_score_mean']:.4f}  "
            f"ECE {r['ece_10_mean']:.4f}"
        )
    print(boot.to_string(index=False))
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Outputs in {MODEL_DIR.parent}")


def to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def _mlp_probs(model, x, device) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(x), 4096):
            out.append(
                torch.sigmoid(model(torch.from_numpy(x[s : s + 4096]).to(device)))
                .squeeze(-1)
                .cpu()
                .numpy()
            )
    return np.concatenate(out)


if __name__ == "__main__":
    main()
