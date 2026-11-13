"""Verify the amended Member 3 artifacts and preserve the historical experiment."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import train_embedding_mlp as pipeline
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_COMMIT = "2dcb02e"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def main():
    original_paths = (
        git(
            "ls-tree",
            "-r",
            "--name-only",
            HISTORICAL_COMMIT,
            "models/phase2/member3",
            "reports/results/phase2_member3",
            "reports/figures/phase2_member3",
        )
        .decode()
        .splitlines()
    )
    historical_hashes = {}
    for path in original_paths:
        expected = hashlib.sha256(git("show", f"{HISTORICAL_COMMIT}:{path}")).hexdigest()
        assert pipeline.sha256(ROOT / path) == expected, f"Historical artifact changed: {path}"
        historical_hashes[path] = expected

    old_log = git("show", f"{HISTORICAL_COMMIT}:docs/decision_log.md").decode()
    log = (ROOT / "docs/decision_log.md").read_text()
    for name in ("D-305", "D-306"):
        old = old_log.split(f"### {name}", 1)[1].split("\n### ", 1)[0].strip()
        current = log.split(f"### {name}", 1)[1].split("\n### ", 1)[0].strip()
        assert current == old, f"Historical decision changed: {name}"

    audit = pipeline.provenance()
    rows = []
    for settings in pipeline.grid():
        record = pipeline.checked_record(
            pipeline.RESULT / pipeline.TUNING / f"{settings['run']}.json", audit
        )
        assert all(record[key] == value for key, value in settings.items())
        values = [row["august_pr_auc"] for row in record["history"]]
        assert record["best_epoch"] == values.index(max(values)) + 1
        assert record["august_pr_auc"] == max(values)
        if "reused_from" in record:
            source = ROOT / record["reused_from"]
            assert pipeline.sha256(source) == record["reused_record_sha256"]
            original = pipeline.read_json(source)
            assert record["history"] == original["history"]
            assert record["august_metrics"] == original["august_metrics"]
            assert settings["dropout"] == 0.2 and settings["batch_size"] == 256
        rows.append(record)
    assert len(rows) == 48 and sum("reused_from" in row for row in rows) == 8
    selection = pipeline.checked_record(pipeline.RESULT / "selection.json", audit)
    assert pipeline.choose_winner(rows)["run"] == selection["winner"]["run"]
    assert selection["grid_size"] == 48 and selection["new_runs"] == 40

    frozen = pipeline.checked_record(pipeline.MODEL / "frozen.json", audit)
    assert frozen["seeds"] == [42, 43, 44, 45, 46]
    assert pipeline.sha256(pipeline.MODEL / "preprocessor.joblib") == frozen["preprocessor_sha256"]
    for item in frozen["models"]:
        assert pipeline.sha256(pipeline.MODEL / f"seed{item['seed']}.pth") == item["sha256"]
        run = pipeline.checked_record(
            pipeline.RESULT / pipeline.REFITS / f"seed{item['seed']}.json", audit
        )
        values = [row["august_pr_auc"] for row in run["selection_history"]]
        assert item["epochs"] == run["best_epoch"] == values.index(max(values)) + 1
        assert len(run["refit_history"]) == run["best_epoch"]

    predictions = pd.read_parquet(pipeline.RESULT / "sep_oct_predictions.parquet")
    baseline = pipeline.align_baselines(
        predictions[["id", "date", "is_delayed"]], pd.read_parquet(pipeline.BASELINE)
    )
    assert len(predictions) == 55628 and predictions["date"].nunique() == 61
    assert predictions["date"].between("2013-09-01", "2013-10-31").all()
    for column in baseline:
        if column in predictions:
            np.testing.assert_array_equal(predictions[column], baseline[column])
    evaluation = pipeline.checked_record(pipeline.RESULT / "evaluation.json", audit)
    assert evaluation["prediction_sha256"] == pipeline.sha256(
        pipeline.RESULT / "sep_oct_predictions.parquet"
    )
    metrics = pd.read_csv(pipeline.RESULT / "per_seed_metrics.csv")
    differences = []
    for row in metrics.itertuples():
        mask = np.ones(len(predictions), dtype=bool)
        if row.period != "sep_oct":
            mask = predictions["date"].dt.month.to_numpy() == (
                9 if row.period == "september" else 10
            )
        column = "lr" if row.model == "lr" else f"{row.model}_seed{int(row.seed)}"
        p = predictions.loc[mask, column].to_numpy()
        y = predictions.loc[mask, "is_delayed"].to_numpy()
        bins = np.minimum((p * 10).astype(int), 9)
        ece = sum(
            float((bins == i).mean()) * abs(float(y[bins == i].mean()) - float(p[bins == i].mean()))
            for i in range(10)
            if (bins == i).any()
        )
        expected = {
            "pr_auc": average_precision_score(y, p),
            "roc_auc": roc_auc_score(y, p),
            "brier_score": brier_score_loss(y, p),
            "ece_10": ece,
            "mean_probability": float(p.mean()),
        }
        differences.extend(abs(value - getattr(row, key)) for key, value in expected.items())
    assert len(differences) == 240 and max(differences) < 1e-12
    boot = pd.read_csv(pipeline.RESULT / "bootstrap_pr_auc_differences.csv")
    means = {
        name: np.mean(
            [
                average_precision_score(
                    predictions["is_delayed"], predictions[f"{name}_seed{seed}"]
                )
                for seed in pipeline.SEEDS
            ]
        )
        for name in ("embedding", "mlp", "gru")
    }
    means["lr"] = average_precision_score(predictions["is_delayed"], predictions["lr"])
    for row in boot.itertuples():
        a, b = row.comparison.split(" - ")
        assert abs(row.difference - (means[a] - means[b])) < 1e-12

    notebook = nbformat.read(ROOT / "notebooks/phase2_member3_embedding_mlp.ipynb", as_version=4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in code)
    assert not any(output.output_type == "error" for cell in code for output in cell.outputs)
    original = set(git("ls-tree", "-r", "--name-only", "3ca1852").decode().splitlines())
    changed = set(git("diff", "--name-only", "3ca1852").decode().splitlines())
    assert original & changed == {"docs/decision_log.md"}

    verification = {
        "protocol": "D-307",
        "joint_grid_configurations": 48,
        "reused_d305_runs": 8,
        "new_seed42_runs": 40,
        "final_seeds": pipeline.SEEDS,
        "historical_artifacts_unchanged": len(historical_hashes),
        "historical_artifact_sha256": historical_hashes,
        "historical_decisions_unchanged": ["D-305", "D-306"],
        "matched_evaluation_flights": len(predictions),
        "evaluation_days": 61,
        "recomputed_metric_values": len(differences),
        "maximum_metric_difference": max(differences),
        "frozen_checkpoint_hashes_verified": True,
        "baseline_probabilities_identical": True,
        "executed_notebook_code_cells": len(code),
        "member1_member2_unchanged": True,
        "sealed_holdout_accessed": False,
    }
    pipeline.write_json(pipeline.RESULT / "verification.json", verification)
    print(
        json.dumps(
            {
                key: value
                for key, value in verification.items()
                if key != "historical_artifact_sha256"
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
