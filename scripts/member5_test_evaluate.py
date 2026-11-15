"""Member 5, PR 2: single November-December test of the D-312 final model (Member 2 GRU, seed 42).

Two modes:

``--check`` (Stage 1; reads only the January-October development file, writes nothing)
    1. Frozen state: the ``phase2-model-frozen`` tag exists, the checkpoint files are
       byte-identical to the tagged versions, and no test outputs exist.
    2. Builds features and daily sequences on January-October, scores September-October with the
       frozen checkpoint and asserts the probabilities equal the saved ``gru_seed42`` column of
       ``reports/results/phase2_member2/sep_oct_predictions.parquet`` exactly.
    3. Rebuilds the September sigmoid and the October threshold from those probabilities and
       asserts they equal D-312 (slope 0.601708, intercept -1.106954, threshold 0.394665).

``--run-test --confirm-single-test-run`` (Stage 2; run once)
    1. Same frozen-state checks, then the full-year file's SHA-256 (D-002).
    2. Writes a run-started marker into the test results folder before the full-year file is
       parsed, so a crashed run cannot be repeated either.
    3. Builds features and sequences over January-December (past-only history, as D-205), repeats
       checks 2 and 3 above on the full-year build, and only then scores November-December
       (``split == "test"``) rows. Nothing is fitted except re-deriving the frozen September
       calibrator, which is asserted equal to D-312 before use.
    4. Reports raw and calibrated PR-AUC, ROC-AUC, Brier, ECE-10 and, at the frozen threshold,
       precision, recall and flights flagged, next to the Logistic Regression's D-205 result.

Outputs: ``reports/results/phase2_member5/test_nov_dec/`` and
``reports/figures/phase2_member5/test_nov_dec/``. The script refuses to run if either exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flight_delay.config import load_config  # noqa: E402
from flight_delay.data.clean import build_labelled_frame  # noqa: E402
from flight_delay.data.load import load_dev_raw, load_raw  # noqa: E402
from flight_delay.data.schema import TARGET_COLUMN  # noqa: E402
from flight_delay.evaluation.evaluate import FixedProbabilities  # noqa: E402
from flight_delay.evaluation.metrics import (  # noqa: E402
    brier,
    expected_calibration_error,
    pr_auc,
    roc_auc,
)
from flight_delay.features.build import build_feature_table  # noqa: E402
from flight_delay.features.sequences import (  # noqa: E402
    SEQUENCE_CHANNELS,
    SequenceScaler,
    build_sequence_tables,
)
from flight_delay.models.calibration import calibrate  # noqa: E402
from flight_delay.models.gru import SequenceDelayModel  # noqa: E402
from flight_delay.models.gru_training import SequenceData, predict_logits, sigmoid  # noqa: E402
from flight_delay.models.operating_point import choose_threshold  # noqa: E402

# --- Frozen state (D-002, D-312) ----------------------------------------------------------------
SEALED_SHA256 = "8bdc6abbf07ac0cbb1036bc14a787cab3eea616be46c31899d2d8048963b08aa"
FROZEN_TAG = "phase2-model-frozen"
MODEL_DIR = "models/phase2/member2"
FROZEN_FILES = [
    f"{MODEL_DIR}/gru_final.pth",
    f"{MODEL_DIR}/static_preprocessor.joblib",
    f"{MODEL_DIR}/sequence_scaler.npz",
    f"{MODEL_DIR}/model_config.json",
]
D312 = {"slope": 0.601708, "intercept": -1.106954, "threshold": 0.394665}
D312_DECIMALS = 6
MIN_PRECISION = 0.60  # D-015
SEQUENCE_TABLE_DAYS = 28  # max_len used when the checkpoint was trained (scripts/train_gru.py)

# --- Inputs and outputs ---------------------------------------------------------------------------
SAVED_PREDICTIONS = ROOT / "reports/results/phase2_member2/sep_oct_predictions.parquet"
SAVED_COLUMN = "gru_seed42"
LR_TEST_RESULTS = ROOT / "reports/results/test_metrics.json"  # D-205
LR_D205_PR_AUC = 0.3979
D205_TEST_FLIGHTS = 53_991
RESULTS_DIR = ROOT / "reports/results/phase2_member5/test_nov_dec"
FIGURES_DIR = ROOT / "reports/figures/phase2_member5/test_nov_dec"
MARKER = "RUN_STARTED.json"

SEPTEMBER = ("2013-09-01", "2013-09-30")
OCTOBER = ("2013-10-01", "2013-10-31")
SEP_OCT = (SEPTEMBER[0], OCTOBER[1])
NOV_DEC = ("2013-11-01", "2013-12-31")


# ================================================================================================
# Safety checks (each raises SystemExit; nothing after a failed check runs)
# ================================================================================================


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_sealed_hash(path: Path, expected: str = SEALED_SHA256) -> None:
    if not path.exists():
        raise SystemExit(f"{path} not found")
    if sha256_file(path) != expected:
        raise SystemExit(f"{path} does not match the SHA-256 recorded in D-002")


def check_no_test_outputs(*dirs: Path) -> None:
    """Refuse if any test output folder exists (a finished or a started run)."""
    for d in dirs:
        if d.exists():
            raise SystemExit(
                f"{d} already exists: the November-December test has been run (or started) once. "
                "It is not run again."
            )


def check_frozen_files(root: Path, tag: str, files: list[str]) -> str:
    """Return the tagged commit; exit unless every file equals its tagged version byte-for-byte."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--verify", f"{tag}^{{commit}}"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        raise SystemExit(f"Tag {tag!r} not found") from None
    for rel in files:
        shown = subprocess.run(["git", "show", f"{tag}:{rel}"], cwd=root, capture_output=True)
        if shown.returncode != 0:
            raise SystemExit(f"{rel} is not in the {tag} commit")
        path = root / rel
        if not path.exists() or path.read_bytes() != shown.stdout:
            raise SystemExit(f"{rel} differs from the {tag} version")
    return commit


def check_frozen_state(root: Path = ROOT) -> str:
    check_no_test_outputs(RESULTS_DIR, FIGURES_DIR)
    return check_frozen_files(root, FROZEN_TAG, FROZEN_FILES)


def check_reproduces(ids: np.ndarray, p: np.ndarray, saved: pd.DataFrame, column: str) -> None:
    """The pipeline's September-October probabilities must equal the saved ones exactly."""
    expected = saved.set_index("id")[column]
    if len(expected) != len(ids) or not pd.Index(ids).equals(expected.index):
        raise SystemExit(
            f"September-October flights differ from the saved predictions "
            f"({len(ids)} vs {len(expected)} flights)"
        )
    want = expected.to_numpy(np.float32)
    got = np.asarray(p, dtype=np.float32)
    if not np.array_equal(got, want):
        diff = np.abs(got.astype(np.float64) - want.astype(np.float64))
        raise SystemExit(
            f"Pipeline does not reproduce saved {column}: {int((diff > 0).sum())} of {len(diff)} "
            f"flights differ, max |difference| {diff.max():.3e}"
        )


def check_frozen_operating_point(
    params: dict, expected: dict = D312, decimals: int = D312_DECIMALS
) -> None:
    for key, value in expected.items():
        if params.get(key) is None or round(params[key], decimals) != value:
            raise SystemExit(f"Rebuilt {key} {params.get(key)} does not equal D-312 ({value})")


# ================================================================================================
# Model, scoring, calibration
# ================================================================================================


def load_frozen_model(root: Path = ROOT):
    cfg = json.loads((root / MODEL_DIR / "model_config.json").read_text())
    if cfg["sequence_channels"] != SEQUENCE_CHANNELS:
        raise SystemExit("sequence channels differ from the checkpoint")
    prep = joblib.load(root / MODEL_DIR / "static_preprocessor.joblib")
    if int(prep.input_dim_) != cfg["static_dim"]:
        raise SystemExit("static preprocessor does not match model_config.json")
    sc = np.load(root / MODEL_DIR / "sequence_scaler.npz")
    scaler = SequenceScaler(mean=sc["mean"], std=sc["std"])
    model = SequenceDelayModel(
        static_dim=cfg["static_dim"],
        seq_channels=len(SEQUENCE_CHANNELS),
        seq_len=cfg["selected_length"],
        hidden_size=cfg["selected_hidden_size"],
        encoder="gru",
    )
    state = torch.load(root / MODEL_DIR / "gru_final.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, prep, scaler, cfg


def build_inputs(raw: pd.DataFrame, cfg: dict):
    """Feature table and daily sequence tables over every date in ``raw`` (past-only history)."""
    table = build_feature_table(raw, cfg)
    labelled = build_labelled_frame(
        raw, cfg["target"]["threshold_minutes"], cfg["target"]["cancelled_policy"]
    )
    labelled[TARGET_COLUMN] = labelled[TARGET_COLUMN].astype(int)
    tables = build_sequence_tables(
        labelled,
        table["date"].max(),
        max_len=SEQUENCE_TABLE_DAYS,
        prior_strength=cfg["features"]["history_prior_strength"],
    )
    return table, tables


def score(rows: pd.DataFrame, tables, model, prep, scaler, length: int) -> np.ndarray:
    """Raw GRU probabilities (float32, as saved by scripts/train_gru.py)."""
    data = SequenceData(
        prep.transform(rows),
        tables.flight_indices(rows),
        rows[TARGET_COLUMN].to_numpy(np.float32),
        scaler.transform_tables(tables),
        length,
    )
    return sigmoid(predict_logits(model, data, torch.device("cpu"))).astype(np.float32)


def rows_between(table: pd.DataFrame, bounds: tuple[str, str]) -> pd.DataFrame:
    return table.loc[pd.to_datetime(table["date"]).between(*bounds)].reset_index(drop=True)


def frozen_operating_point(p_sep_oct: np.ndarray, frame: pd.DataFrame):
    """September sigmoid and October threshold, exactly as D-311 / D-312 built them."""
    y = frame[TARGET_COLUMN].to_numpy(int)
    p = np.asarray(p_sep_oct, dtype=np.float32).astype(float)  # D-311 read float32 from parquet
    dates = pd.to_datetime(frame["date"])
    sep, octo = dates.between(*SEPTEMBER).to_numpy(), dates.between(*OCTOBER).to_numpy()
    calibrated = calibrate(FixedProbabilities().fit(), p[sep].reshape(-1, 1), y[sep], "sigmoid")
    sig = calibrated.calibrated_classifiers_[0].calibrators[0]
    # sklearn: P = 1 / (1 + exp(a * f + b)), so calibrated logit = -a * logit(p) - b.
    params = {"slope": float(-sig.a_), "intercept": float(-sig.b_)}

    def calibrator(q):
        return calibrated.predict_proba(np.asarray(q, dtype=float).reshape(-1, 1))[:, 1]

    op = choose_threshold(y[octo], calibrator(p[octo]), min_precision=MIN_PRECISION)
    params["threshold"] = op["threshold"] if op["feasible"] else None
    params["october"] = {
        "precision": op["precision"],
        "recall": op["recall"],
        "flagged": int((calibrator(p[octo]) >= op["threshold"]).sum()) if op["feasible"] else 0,
        "flights": int(octo.sum()),
    }
    return calibrator, params


def probability_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "flights": int(len(y)),
        "delay_rate": float(np.mean(y)),
        "mean_probability": float(np.mean(p)),
        "pr_auc": float(pr_auc(y, p)),
        "roc_auc": float(roc_auc(y, p)),
        "brier": float(brier(y, p)),
        "ece_10": float(expected_calibration_error(y, p, n_bins=10)),
    }


def threshold_metrics(y: np.ndarray, q: np.ndarray, threshold: float) -> dict:
    flagged = q >= threshold
    return {
        "threshold": float(threshold),
        "flagged": int(flagged.sum()),
        "flagged_share": float(flagged.mean()),
        "precision": float(y[flagged].mean()) if flagged.any() else None,
        "recall": float(flagged[y == 1].mean()),
    }


def verify_sep_oct(table, tables, model, prep, scaler, length, cfg_frame_label: str):
    """Checks 2 and 3: exact reproduction and frozen calibrator/threshold. Returns calibrator."""
    sep_oct = rows_between(table, SEP_OCT)
    p = score(sep_oct, tables, model, prep, scaler, length)
    saved = pd.read_parquet(SAVED_PREDICTIONS, columns=["id", SAVED_COLUMN])
    check_reproduces(sep_oct["id"].to_numpy(), p, saved, SAVED_COLUMN)
    print(
        f"[ok] {cfg_frame_label}: {len(p):,} September-October probabilities equal the saved "
        f"{SAVED_COLUMN} exactly"
    )
    calibrator, params = frozen_operating_point(p, sep_oct)
    check_frozen_operating_point(params)
    print(
        f"[ok] {cfg_frame_label}: September sigmoid slope {params['slope']:.6f}, intercept "
        f"{params['intercept']:.6f}; October threshold {params['threshold']:.6f} "
        f"(precision {params['october']['precision']:.4f}, recall "
        f"{params['october']['recall']:.4f}, {params['october']['flagged']} flagged) = D-312"
    )
    return calibrator, params


# ================================================================================================
# Stages
# ================================================================================================


def stage1() -> None:
    commit = check_frozen_state()
    print(
        f"[ok] tag {FROZEN_TAG} -> {commit[:7]}; {len(FROZEN_FILES)} checkpoint files "
        "byte-identical to the tagged versions"
    )
    print(
        f"[ok] no test outputs at {RESULTS_DIR.relative_to(ROOT)} or "
        f"{FIGURES_DIR.relative_to(ROOT)}"
    )
    cfg = load_config()
    model, prep, scaler, mcfg = load_frozen_model()
    print(f"torch {torch.__version__} (checkpoint trained with {mcfg['torch_version']})")
    raw = load_dev_raw(cfg, ROOT)  # January-October only; raises on any later row
    table, tables = build_inputs(raw, cfg)
    verify_sep_oct(table, tables, model, prep, scaler, mcfg["selected_length"], "Jan-Oct build")
    print("\nStage 1 passed. November-December was not read. Nothing was written.")


def stage2() -> None:
    commit = check_frozen_state()
    cfg = load_config()
    full_path = ROOT / cfg["paths"]["raw_sealed"]
    check_sealed_hash(full_path)
    print(f"[ok] frozen state ({FROZEN_TAG} -> {commit[:7]}), no test outputs, D-002 SHA-256")
    lr = json.loads(LR_TEST_RESULTS.read_text())
    lr_rows = {r["probabilities"]: r for r in lr["metrics"]}
    if round(lr_rows["raw"]["pr_auc"], 4) != LR_D205_PR_AUC:
        raise SystemExit("D-205 results file does not match the decision log")

    # From here on the run counts as done, even if it crashes.
    RESULTS_DIR.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (RESULTS_DIR / MARKER).write_text(
        json.dumps(
            {"started_utc": started, "frozen_commit": commit, "torch": torch.__version__}, indent=2
        )
        + "\n"
    )

    model, prep, scaler, mcfg = load_frozen_model()
    length = mcfg["selected_length"]
    table, tables = build_inputs(load_raw(full_path), cfg)
    calibrator, params = verify_sep_oct(table, tables, model, prep, scaler, length, "Jan-Dec build")

    test = table[table["split"] == "test"].reset_index(drop=True)
    dates = pd.to_datetime(test["date"])
    if test.empty or not dates.between(*NOV_DEC).all() or len(test) != D205_TEST_FLIGHTS:
        raise SystemExit(f"unexpected test rows ({len(test)}; D-205 scored {D205_TEST_FLIGHTS})")
    y = test[TARGET_COLUMN].to_numpy(int)
    p = score(test, tables, model, prep, scaler, length)
    q = calibrator(p.astype(float))

    metrics = [
        {"model": "gru_seed42", "probabilities": "raw", **probability_metrics(y, p)},
        {
            "model": "gru_seed42",
            "probabilities": "sigmoid (frozen, D-312)",
            **probability_metrics(y, q),
        },
    ]
    at_threshold = threshold_metrics(y, q, params["threshold"])
    lr_compare = [
        {
            "model": "lr_d205",
            "probabilities": name,
            **{
                k: lr_rows[name][k]
                for k in [
                    "flights",
                    "delay_rate",
                    "mean_probability",
                    "pr_auc",
                    "roc_auc",
                    "brier",
                    "ece_10",
                ]
            },
        }
        for name in ("raw", "sigmoid (frozen)")
    ]
    result = {
        "protocol": "D-312 (D-203 test procedure)",
        "frozen_tag": FROZEN_TAG,
        "frozen_commit": commit,
        "run_started_utc": started,
        "torch": torch.__version__,
        "test_period": list(NOV_DEC),
        "frozen_operating_point": {k: params[k] for k in ("slope", "intercept", "threshold")},
        "metrics": metrics,
        "at_threshold": at_threshold,
        "logistic_regression_d205": {
            "metrics": lr_compare,
            "at_threshold": lr["at_threshold"],
            "source": str(LR_TEST_RESULTS.relative_to(ROOT)),
        },
    }
    (RESULTS_DIR / "test_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    pd.DataFrame(metrics + lr_compare).to_csv(RESULTS_DIR / "test_metrics.csv", index=False)
    pd.DataFrame(
        {
            "id": test["id"],
            "date": test["date"],
            TARGET_COLUMN: y,
            "gru_seed42_raw": p,
            "gru_seed42_calibrated": q.astype(np.float32),
        }
    ).to_parquet(RESULTS_DIR / "test_predictions.parquet", index=False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect calibration")
    for name, prob in (("raw", p), ("sigmoid (frozen, D-312)", q)):
        observed, predicted = calibration_curve(y, prob, n_bins=10, strategy="uniform")
        ax.plot(predicted, observed, marker="o", label=name)
    ax.axvline(params["threshold"], color="#d62728", lw=1, ls=":", label="frozen threshold")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed delay rate")
    ax.set_title("GRU seed 42 reliability — test period (Nov–Dec 2013)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "test_reliability.png", dpi=150)
    plt.close(fig)

    print("\n" + pd.DataFrame(metrics + lr_compare).round(4).to_string(index=False))
    print("at threshold:", at_threshold)
    print(f"Saved {RESULTS_DIR.relative_to(ROOT)}/ and {FIGURES_DIR.relative_to(ROOT)}/")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Stage 1 (January-October only)")
    mode.add_argument("--run-test", action="store_true", help="Stage 2: the single test run")
    parser.add_argument("--confirm-single-test-run", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        stage1()
    elif not args.confirm_single_test_run:
        raise SystemExit("--run-test also needs --confirm-single-test-run")
    else:
        stage2()


if __name__ == "__main__":
    main()
