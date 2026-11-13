"""Protocol checks that do not access real evaluation or sealed data."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("torch")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_embedding_mlp.py"
spec = importlib.util.spec_from_file_location("member3_pipeline", SCRIPT)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def test_grid_is_the_registered_joint_48_configurations():
    grid = pipeline.grid()
    assert len(grid) == 48
    assert {item["learning_rate"] for item in grid} == {0.0001, 0.0003}
    assert {tuple(item["hidden_dims"]) for item in grid} == {(64, 32), (128, 64)}
    assert {item["embedding_cap"] for item in grid} == {4, 8}
    assert {item["dropout"] for item in grid} == {0.1, 0.2, 0.4}
    assert {item["batch_size"] for item in grid} == {256, 1024}
    reused = [item for item in grid if item["dropout"] == 0.2 and item["batch_size"] == 256]
    assert len(reused) == 8
    assert len(grid) - len(reused) == 40


def test_inconsistent_historical_scores_are_rejected(tmp_path, monkeypatch):
    record = pipeline.read_json(pipeline.HISTORICAL / "tuning/run01.json")
    record["august_pr_auc"] += 0.01
    pipeline.write_json(tmp_path / "tuning/run01.json", record)
    monkeypatch.setattr(pipeline, "HISTORICAL", tmp_path)
    settings = next(
        row for row in pipeline.grid() if row["dropout"] == 0.2 and row["batch_size"] == 256
    )
    with pytest.raises(ValueError, match="Historical August score"):
        pipeline.historical_run(settings, record["audit"])


def test_provenance_rejects_a_non_development_path(monkeypatch):
    monkeypatch.setattr(pipeline, "load_config", lambda: {"paths": {"raw": "wrong.csv"}})
    with pytest.raises(ValueError, match="dedicated January-October"):
        pipeline.provenance()


def test_completed_evaluation_cannot_be_scored_again(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "RESULT", tmp_path)
    (tmp_path / "evaluation.json").write_text("{}")
    with pytest.raises(FileExistsError, match="Final evaluation is complete"):
        pipeline.evaluate({})


def test_resumed_run_rejects_changed_inputs(tmp_path):
    path = tmp_path / "record.json"
    pipeline.write_json(path, {"audit": {"source": "original"}})
    with pytest.raises(ValueError, match="Inputs or source changed"):
        pipeline.checked_record(path, {"source": "changed"})


def test_winner_uses_august_and_registered_tie_breaks():
    records = [
        {"run": "run01", "august_pr_auc": 0.4, "parameters": 200, "learning_rate": 0.0001},
        {"run": "run02", "august_pr_auc": 0.4, "parameters": 100, "learning_rate": 0.0003},
        {"run": "run03", "august_pr_auc": 0.4, "parameters": 100, "learning_rate": 0.0001},
    ]
    assert pipeline.choose_winner(records)["run"] == "run03"
    records[0]["august_pr_auc"] = 0.4000001
    assert pipeline.choose_winner(records)["run"] == "run01"


def rows():
    return pd.DataFrame(
        {"id": [1, 2], "date": pd.to_datetime(["2013-09-01", "2013-10-01"]), "is_delayed": [0, 1]}
    )


def test_baselines_align_by_flight_id():
    expected = rows()
    actual = pipeline.align_baselines(expected, expected.iloc[::-1])
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("change", ["duplicate", "missing", "wrong_label", "wrong_date"])
def test_baseline_mismatches_are_rejected(change):
    expected, baseline = rows(), rows()
    if change == "duplicate":
        baseline.loc[1, "id"] = 1
    elif change == "missing":
        baseline.loc[1, "id"] = 3
    elif change == "wrong_label":
        baseline.loc[1, "is_delayed"] = 0
    else:
        baseline.loc[1, "date"] = pd.Timestamp("2013-10-02")
    with pytest.raises(ValueError):
        pipeline.align_baselines(expected, baseline)
