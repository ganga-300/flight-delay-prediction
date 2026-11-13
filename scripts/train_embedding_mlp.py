"""Member 3 D-307 pipeline: check, tune on August, refit, then evaluate once.

No stage reads the sealed raw file. The evaluation stage reuses saved predictions
if interrupted during bootstrapping, so recovery does not rescore the models.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

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
    numeric_features,
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
from flight_delay.models.gru_training import sigmoid  # noqa: E402

MODEL = ROOT / "models/phase2/member3/refit_d307"
RESULT = ROOT / "reports/results/phase2_member3/d307"
HISTORICAL = ROOT / "reports/results/phase2_member3"
TUNING = "tuning_d307"
REFITS = "refit_d307"
BASELINE = ROOT / "reports/results/phase2_member2/sep_oct_predictions.parquet"
SEEDS = [42, 43, 44, 45, 46]
METRICS = ["pr_auc", "roc_auc", "brier_score", "ece_10", "mean_probability"]
SOURCE_FILES = [
    "src/flight_delay/models/embedding_mlp.py",
    "src/flight_delay/models/embedding_training.py",
    "scripts/train_embedding_mlp.py",
    "src/flight_delay/models/mlp_training.py",
    "src/flight_delay/models/gru_training.py",
    "src/flight_delay/features/encoding.py",
    "scripts/train_gru.py",
    "configs/config.yaml",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def provenance() -> dict:
    log = (ROOT / "docs/decision_log.md").read_text()
    decision = log.split("### D-307", 1)[1].split("\n### ", 1)[0].strip()
    cfg = load_config()
    dev = ROOT / cfg["paths"]["raw"]
    if dev.resolve() != (ROOT / "data/interim/flights_dev.csv").resolve():
        raise ValueError("D-307 requires the dedicated January-October development file")
    return {
        "decision_sha256": hashlib.sha256(decision.encode()).hexdigest(),
        "development_sha256": sha256(dev),
        "baseline_predictions_sha256": sha256(BASELINE),
        "source_sha256": {path: sha256(ROOT / path) for path in SOURCE_FILES},
    }


def metadata(audit: dict) -> dict:
    return {
        "protocol": "D-307",
        "audit": audit,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "torch_version": str(torch.__version__),
        "device": "cpu",
        "holdout_accessed": False,
    }


def checked_record(path: Path, audit: dict) -> dict:
    record = read_json(path)
    if record["audit"] != audit:
        raise ValueError(f"Inputs or source changed since {path}; preserve the registered run")
    return record


def grid() -> list[dict]:
    return [
        {
            "run": f"d307_{index:03d}",
            "learning_rate": rate,
            "hidden_dims": list(hidden),
            "embedding_cap": cap,
            "dropout": dropout,
            "batch_size": batch,
        }
        for index, (rate, hidden, cap, dropout, batch) in enumerate(
            itertools.product(
                [0.0001, 0.0003], [(64, 32), (128, 64)], [4, 8], [0.1, 0.2, 0.4], [256, 1024]
            ),
            start=1,
        )
    ]


def choose_winner(records: list[dict]) -> dict:
    return min(
        records,
        key=lambda row: (
            -row["august_pr_auc"],
            row["parameters"],
            row["learning_rate"],
            row["run"],
        ),
    )


def feature_table(end: str) -> tuple[pd.DataFrame, dict]:
    cfg = load_config()
    raw = load_dev_raw(cfg, ROOT)
    raw = raw.loc[raw["date"] <= pd.Timestamp(end)].copy()
    return build_feature_table(raw, cfg), cfg


def make_data(frame: pd.DataFrame, prep: EmbeddingPreprocessor) -> EmbeddingData:
    numeric, categories = prep.transform(frame)
    return EmbeddingData(numeric, categories, frame["is_delayed"].to_numpy(np.float32))


def stage_a(table: pd.DataFrame, cfg: dict):
    fit = table.loc[table["date"] <= "2013-07-31"]
    tune = table.loc[table["date"].between("2013-08-01", "2013-08-31")]
    if (len(fit), len(tune)) != (188971, 28756):
        raise ValueError("January-July / August flight counts differ from D-301")
    prep = EmbeddingPreprocessor(
        numeric_features(cfg), list(CATEGORICAL), cfg["features"]["categorical_min_count"]
    ).fit(fit)
    return prep, make_data(fit, prep), make_data(tune, prep)


def historical_run(settings: dict, audit: dict) -> tuple[dict, Path]:
    """Validate and locate one compatible, unmodified D-305 August run."""
    matches = []
    for path in sorted((HISTORICAL / "tuning").glob("run*.json")):
        record = read_json(path)
        if all(
            record[key] == settings[key]
            for key in ("learning_rate", "hidden_dims", "embedding_cap")
        ):
            matches.append((record, path))
    if len(matches) != 1 or settings["dropout"] != 0.2 or settings["batch_size"] != 256:
        raise ValueError("Expected exactly one compatible D-305 run")
    record, path = matches[0]
    scores = [row["august_pr_auc"] for row in record["history"]]
    if (
        record["august_pr_auc"] != max(scores)
        or record["best_epoch"] != scores.index(max(scores)) + 1
    ):
        raise ValueError("Historical August score or selected epoch is inconsistent")
    for key in ("development_sha256", "baseline_predictions_sha256"):
        if record["audit"][key] != audit[key]:
            raise ValueError("Historical run input hashes differ")
    log = (ROOT / "docs/decision_log.md").read_text()
    original_decision = log.split("### D-305", 1)[1].split("\n### ", 1)[0].strip()
    if hashlib.sha256(original_decision.encode()).hexdigest() != record["audit"]["decision_sha256"]:
        raise ValueError("Historical D-305 decision changed")
    edited = {"scripts/train_embedding_mlp.py", "src/flight_delay/models/embedding_training.py"}
    for source, expected in record["audit"]["source_sha256"].items():
        historical = subprocess.check_output(
            ["git", "show", f"{record['git_commit']}:{source}"], cwd=ROOT
        )
        if hashlib.sha256(historical).hexdigest() != expected:
            raise ValueError(f"Historical source mismatch for {source}")
        if source not in edited and audit["source_sha256"][source] != expected:
            raise ValueError(f"Shared source changed for reused run: {source}")
    return record, path


def check() -> None:
    cfg = load_config()
    if cfg["split"]["valid"] != ["2013-09-01", "2013-10-31"]:
        raise ValueError("Expected the D-301 September-October evaluation period")
    schema = pq.read_schema(BASELINE)
    required = {"id", "date", "is_delayed", "lr"} | {
        f"{model}_seed{seed}" for model in ("mlp", "gru") for seed in SEEDS
    }
    if not required.issubset(schema.names):
        raise ValueError("Member 2 prediction schema is incomplete")
    assert len(numeric_features(cfg)) == 49 and len(CATEGORICAL) == 10
    audit = provenance()
    reused = [
        settings
        for settings in grid()
        if settings["dropout"] == 0.2 and settings["batch_size"] == 256
    ]
    for settings in reused:
        historical_run(settings, audit)
    print("PASS: D-307 present, 59 features, 48 settings, seeds 42-46, baseline schema")
    print("PASS: all 8 reusable D-305 records passed source, input and checkpoint-selection checks")
    print("PASS: development file and provenance verified; training not started")
    print(json.dumps(audit, indent=2))


def tune(audit: dict) -> None:
    selection_path = RESULT / "selection.json"
    if selection_path.exists():
        raise FileExistsError("August selection is already frozen")
    table, cfg = feature_table("2013-08-31")
    prep, fit, august = stage_a(table, cfg)
    records = []
    for settings in grid():
        path = RESULT / TUNING / (settings["run"] + ".json")
        if path.exists():
            record = checked_record(path, audit)
            print(f"Reusing completed August run {settings['run']}", flush=True)
        elif settings["dropout"] == 0.2 and settings["batch_size"] == 256:
            original, original_path = historical_run(settings, audit)
            record = {
                **original,
                **metadata(audit),
                **settings,
                "reused_from": str(original_path.relative_to(ROOT)),
                "reused_record_sha256": sha256(original_path),
                "original_training_commit": original["git_commit"],
            }
            write_json(path, record)
            print(f"Reused D-305 {original['run']} as {settings['run']}", flush=True)
        else:
            print(f"\nAugust grid {settings}", flush=True)
            model, history, epoch, scores = train_embedding(
                fit, august, prep.vocabulary_sizes, settings, seed=42
            )
            record = {
                **metadata(audit),
                **settings,
                "best_epoch": epoch,
                "august_pr_auc": scores["pr_auc"],
                "august_metrics": scores,
                "parameters": sum(p.numel() for p in model.parameters()),
                "history": history,
            }
            write_json(path, record)
        records.append(record)
    winner = choose_winner(records)
    write_json(
        selection_path,
        {
            **metadata(audit),
            "winner": {
                key: value for key, value in winner.items() if key not in ("history", "audit")
            },
            "selection_metric": "August PR-AUC only",
            "grid_size": len(records),
            "reused_runs": sum("reused_from" in row for row in records),
            "new_runs": sum("reused_from" not in row for row in records),
        },
    )
    pd.DataFrame(
        [
            {
                key: row[key]
                for key in (
                    "run",
                    "learning_rate",
                    "hidden_dims",
                    "embedding_cap",
                    "dropout",
                    "batch_size",
                    "parameters",
                    "best_epoch",
                    "august_pr_auc",
                )
            }
            for row in records
        ]
    ).to_csv(RESULT / "grid_august.csv", index=False)
    print("\nFROZEN AUGUST WINNER", winner["run"], winner["august_pr_auc"], flush=True)


def refit(audit: dict) -> None:
    if (MODEL / "frozen.json").exists():
        raise FileExistsError("Five final seed models are already frozen")
    settings = checked_record(RESULT / "selection.json", audit)["winner"]
    table, cfg = feature_table("2013-08-31")
    prep_a, fit, august = stage_a(table, cfg)
    prep_b = EmbeddingPreprocessor(
        numeric_features(cfg), list(CATEGORICAL), cfg["features"]["categorical_min_count"]
    ).fit(table)
    final = make_data(table, prep_b)
    MODEL.mkdir(parents=True, exist_ok=True)
    prep_path = MODEL / "preprocessor.joblib"
    records = []
    for seed in SEEDS:
        path = MODEL / f"seed{seed}.pth"
        record_path = RESULT / REFITS / f"seed{seed}.json"
        if record_path.exists() and path.exists():
            record = checked_record(record_path, audit)
            if sha256(path) != record["checkpoint_sha256"]:
                raise ValueError("Saved seed checkpoint changed")
            print(f"Reusing frozen seed {seed}", flush=True)
        else:
            print(f"\nSeed {seed}: choose epochs on August", flush=True)
            if seed == 42:
                original = checked_record(RESULT / TUNING / (settings["run"] + ".json"), audit)
                epoch, august_scores = original["best_epoch"], original["august_metrics"]
                history_a = original["history"]
            else:
                _, history_a, epoch, august_scores = train_embedding(
                    fit, august, prep_a.vocabulary_sizes, settings, seed
                )
            print(f"Seed {seed}: refit January-August for {epoch} epoch(s)", flush=True)
            model, history_b, _, _ = train_embedding(
                final, None, prep_b.vocabulary_sizes, settings, seed, max_epochs=epoch
            )
            temporary = path.with_suffix(".pth.tmp")
            torch.save(model.state_dict(), temporary)
            temporary.replace(path)
            record = {
                **metadata(audit),
                "seed": seed,
                "best_epoch": epoch,
                "august_metrics": august_scores,
                "selection_history": history_a,
                "refit_history": history_b,
                "checkpoint_sha256": sha256(path),
            }
            write_json(record_path, record)
        records.append(record)
    prep_b.save(prep_path)
    write_json(
        MODEL / "frozen.json",
        {
            **metadata(audit),
            "settings": settings,
            "seeds": SEEDS,
            "numeric_dim": len(prep_b.numeric),
            "vocabulary_sizes": prep_b.vocabulary_sizes,
            "preprocessor_sha256": sha256(prep_path),
            "training_rows": len(table),
            "models": [
                {
                    "seed": row["seed"],
                    "epochs": row["best_epoch"],
                    "sha256": row["checkpoint_sha256"],
                }
                for row in records
            ],
        },
    )
    pd.DataFrame(
        [
            {
                "seed": row["seed"],
                "best_epoch": row["best_epoch"],
                "august_pr_auc": row["august_metrics"]["pr_auc"],
            }
            for row in records
        ]
    ).to_csv(RESULT / "august_seed_selection.csv", index=False)
    print("FROZEN: all five January-August models; evaluation pending", flush=True)


def align_baselines(rows: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    if not rows["id"].is_unique or not baseline["id"].is_unique:
        raise ValueError("Duplicate flight IDs")
    if len(rows) != len(baseline) or set(rows["id"]) != set(baseline["id"]):
        raise ValueError("Evaluation flight IDs differ from the saved baseline")
    aligned = baseline.set_index("id").loc[rows["id"]].reset_index()
    for column in ("date", "is_delayed"):
        if not np.array_equal(rows[column].to_numpy(), aligned[column].to_numpy()):
            raise ValueError(f"Baseline {column} is misaligned")
    return aligned


def score(labels: np.ndarray, probability: np.ndarray) -> dict:
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Invalid probabilities")
    return {
        "pr_auc": pr_auc(labels, probability),
        "roc_auc": roc_auc(labels, probability),
        "brier_score": brier(labels, probability),
        "ece_10": expected_calibration_error(labels, probability),
        "mean_probability": float(probability.mean()),
    }


def evaluate(audit: dict) -> None:
    if (RESULT / "evaluation.json").exists():
        raise FileExistsError("Final evaluation is complete; use the saved results")
    frozen = checked_record(MODEL / "frozen.json", audit)
    prep_path = MODEL / "preprocessor.joblib"
    if sha256(prep_path) != frozen["preprocessor_sha256"]:
        raise ValueError("Frozen preprocessing changed")
    for model_record in frozen["models"]:
        if sha256(MODEL / f"seed{model_record['seed']}.pth") != model_record["sha256"]:
            raise ValueError("Frozen model changed")

    predictions_path = RESULT / "sep_oct_predictions.parquet"
    if predictions_path.exists():
        print(
            "Resuming comparison from saved predictions; model scoring already complete", flush=True
        )
        predictions = pd.read_parquet(predictions_path)
    else:
        table, _ = feature_table("2013-10-31")
        rows = table.loc[table["date"].between("2013-09-01", "2013-10-31")].reset_index(drop=True)
        if len(rows) != 55628:
            raise ValueError("Expected the same 55,628 September-October flights")
        baseline = align_baselines(rows, pd.read_parquet(BASELINE))
        columns = ["id", "date", "is_delayed", "lr"] + [
            f"{name}_seed{seed}" for name in ("mlp", "gru") for seed in SEEDS
        ]
        predictions = baseline[columns].copy()
        prep = EmbeddingPreprocessor.load(prep_path)
        data = make_data(rows, prep)
        settings = frozen["settings"]
        for seed in SEEDS:
            model = EmbeddingMLP(
                frozen["numeric_dim"],
                frozen["vocabulary_sizes"],
                settings["embedding_cap"],
                tuple(settings["hidden_dims"]),
                dropout=settings["dropout"],
            )
            model.load_state_dict(torch.load(MODEL / f"seed{seed}.pth", weights_only=True))
            predictions[f"embedding_seed{seed}"] = sigmoid(predict_logits(model, data))
        temporary = predictions_path.with_suffix(".parquet.tmp")
        predictions.to_parquet(temporary, index=False)
        temporary.replace(predictions_path)

    labels = predictions["is_delayed"].to_numpy()
    probabilities = {
        name: [predictions[f"{name}_seed{seed}"].to_numpy() for seed in SEEDS]
        for name in ("embedding", "mlp", "gru")
    }
    probabilities["lr"] = [predictions["lr"].to_numpy()]
    records = []
    for name, values in probabilities.items():
        for index, probability in enumerate(values):
            for period, mask in {
                "sep_oct": np.ones(len(labels), dtype=bool),
                "september": predictions["date"].dt.month.to_numpy() == 9,
                "october": predictions["date"].dt.month.to_numpy() == 10,
            }.items():
                records.append(
                    {
                        "model": name,
                        "seed": SEEDS[index] if name != "lr" else None,
                        "period": period,
                        **score(labels[mask], probability[mask]),
                    }
                )
    metrics = pd.DataFrame(records)
    metrics.to_csv(RESULT / "per_seed_metrics.csv", index=False)
    summary = metrics.groupby(["model", "period"])[METRICS].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(RESULT / "comparison.csv", index=False)
    print("\nFINAL SCORES (saved before bootstrap)", flush=True)
    print(summary.xs("sep_oct", level="period").round(5).to_string(), flush=True)
    print("Computing 1,000 paired whole-day bootstrap resamples...", flush=True)
    bootstrap = day_bootstrap(
        predictions["date"].to_numpy(),
        labels,
        probabilities,
        [("embedding", name) for name in ("mlp", "lr", "gru")],
        seed=42,
    )
    bootstrap.to_csv(RESULT / "bootstrap_pr_auc_differences.csv", index=False)
    write_json(
        RESULT / "evaluation.json",
        {
            **metadata(audit),
            "rows": len(predictions),
            "days": predictions["date"].nunique(),
            "prediction_sha256": sha256(predictions_path),
            "bootstrap_resamples": 1000,
            "bootstrap": bootstrap.to_dict(orient="records"),
        },
    )
    print(bootstrap.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["check", "tune", "refit", "evaluate"], default="check")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage == "check":
        check()
        return
    RESULT.mkdir(parents=True, exist_ok=True)
    audit = provenance()
    {"tune": tune, "refit": refit, "evaluate": evaluate}[args.stage](audit)


if __name__ == "__main__":
    main()
