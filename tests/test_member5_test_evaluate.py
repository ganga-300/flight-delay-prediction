"""Safety checks of the single November-December test run (scripts/member5_test_evaluate.py, D-312).

Synthetic data only: no real data file is read and no real output folder is touched.
"""

import importlib.util
import json
import subprocess

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from flight_delay.config import REPO_ROOT  # noqa: E402
from flight_delay.features.sequences import SequenceScaler  # noqa: E402
from flight_delay.models.gru import SequenceDelayModel  # noqa: E402
from flight_delay.models.mlp_training import MLPPreprocessor, set_torch_seed  # noqa: E402
from tests.conftest import make_raw  # noqa: E402

SCRIPT = REPO_ROOT / "scripts" / "member5_test_evaluate.py"
spec = importlib.util.spec_from_file_location("member5_test_evaluate", SCRIPT)
m5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m5)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point every output and input path of the script into tmp_path."""
    results, figures = tmp_path / "results" / "test_nov_dec", tmp_path / "figures" / "test_nov_dec"
    lr = tmp_path / "test_metrics.json"
    lr.write_text(json.dumps({
        "metrics": [{"probabilities": "raw", "pr_auc": 0.39791}],
        "at_threshold": "not available",
    }))
    monkeypatch.setattr(m5, "RESULTS_DIR", results)
    monkeypatch.setattr(m5, "FIGURES_DIR", figures)
    monkeypatch.setattr(m5, "LR_TEST_RESULTS", lr)
    monkeypatch.setattr(m5, "check_frozen_files", lambda *a, **k: "f" * 40)
    monkeypatch.setattr(m5, "check_sealed_hash", lambda *a, **k: None)
    return results, figures


# --- D-002 hash ---------------------------------------------------------------------------------


def test_hash_check_accepts_matching_file_and_rejects_others(tmp_path):
    f = tmp_path / "flights.csv"
    f.write_bytes(b"id,month\n1,11\n")
    recorded = m5.sha256_file(f)
    m5.check_sealed_hash(f, recorded)
    with pytest.raises(SystemExit, match="SHA-256"):
        m5.check_sealed_hash(f, m5.SEALED_SHA256)
    f.write_bytes(b"id,month\n1,12\n")  # one byte changed after the hash was recorded
    with pytest.raises(SystemExit, match="SHA-256"):
        m5.check_sealed_hash(f, recorded)
    with pytest.raises(SystemExit, match="not found"):
        m5.check_sealed_hash(tmp_path / "missing.csv")


def test_recorded_hash_is_the_d002_value():
    assert m5.SEALED_SHA256 == "8bdc6abbf07ac0cbb1036bc14a787cab3eea616be46c31899d2d8048963b08aa"


# --- Frozen checkpoint files vs tag -------------------------------------------------------------


@pytest.fixture
def tagged_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "m").mkdir(parents=True)
    (repo / "m" / "a.bin").write_bytes(b"\x00weights")
    (repo / "m" / "b.json").write_text("{}")

    def git(*args):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

    git("init", "-q")
    git("-c", "user.name=t", "-c", "user.email=t@t", "add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "freeze")
    git("tag", "frozen")
    return repo


def test_frozen_files_pass_when_unchanged(tagged_repo):
    commit = m5.check_frozen_files(tagged_repo, "frozen", ["m/a.bin", "m/b.json"])
    assert len(commit) == 40


def test_frozen_files_fail_when_a_file_changed_after_the_tag(tagged_repo):
    (tagged_repo / "m" / "a.bin").write_bytes(b"\x00weightz")
    with pytest.raises(SystemExit, match="m/a.bin differs"):
        m5.check_frozen_files(tagged_repo, "frozen", ["m/a.bin", "m/b.json"])


def test_frozen_files_fail_when_a_file_is_missing(tagged_repo):
    (tagged_repo / "m" / "b.json").unlink()
    with pytest.raises(SystemExit, match="m/b.json differs"):
        m5.check_frozen_files(tagged_repo, "frozen", ["m/a.bin", "m/b.json"])


def test_frozen_files_fail_when_tag_is_missing(tagged_repo):
    with pytest.raises(SystemExit, match="not found"):
        m5.check_frozen_files(tagged_repo, "phase2-model-frozen", ["m/a.bin"])


def test_frozen_files_fail_when_file_was_not_tagged(tagged_repo):
    (tagged_repo / "m" / "c.npz").write_bytes(b"new")
    with pytest.raises(SystemExit, match="not in the frozen commit"):
        m5.check_frozen_files(tagged_repo, "frozen", ["m/c.npz"])


def test_script_freezes_the_d312_checkpoint_under_the_d312_tag():
    assert m5.FROZEN_TAG == "phase2-model-frozen"
    assert set(m5.FROZEN_FILES) >= {
        "models/phase2/member2/gru_final.pth",
        "models/phase2/member2/static_preprocessor.joblib",
        "models/phase2/member2/sequence_scaler.npz",
    }


# --- Refuse to run twice ------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["results", "figures"])
def test_refuses_when_any_test_output_folder_exists(tmp_path, which):
    results, figures = tmp_path / "results", tmp_path / "figures"
    {"results": results, "figures": figures}[which].mkdir()
    with pytest.raises(SystemExit, match="already exists"):
        m5.check_no_test_outputs(results, figures)


def test_stage2_refuses_before_touching_data_when_results_exist(sandbox, monkeypatch):
    results, _ = sandbox
    results.mkdir(parents=True)

    def must_not_run(*a, **k):
        raise AssertionError("data or model was loaded")

    for name in ("check_sealed_hash", "load_raw", "load_frozen_model", "build_inputs"):
        monkeypatch.setattr(m5, name, must_not_run)
    with pytest.raises(SystemExit, match="already exists"):
        m5.main(["--run-test", "--confirm-single-test-run"])


def test_crashed_run_cannot_be_repeated(sandbox, monkeypatch):
    results, _ = sandbox

    def crash(*a, **k):
        raise RuntimeError("simulated crash after the run started")

    monkeypatch.setattr(m5, "load_frozen_model", crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        m5.main(["--run-test", "--confirm-single-test-run"])
    marker = json.loads((results / m5.MARKER).read_text())
    assert marker["frozen_commit"] == "f" * 40
    with pytest.raises(SystemExit, match="already exists"):
        m5.main(["--run-test", "--confirm-single-test-run"])


def test_failed_hash_leaves_no_marker(sandbox, monkeypatch):
    results, _ = sandbox

    def bad_hash(*a, **k):
        raise SystemExit("does not match the SHA-256 recorded in D-002")

    monkeypatch.setattr(m5, "check_sealed_hash", bad_hash)
    with pytest.raises(SystemExit, match="SHA-256"):
        m5.main(["--run-test", "--confirm-single-test-run"])
    assert not results.exists()


def test_run_test_needs_explicit_confirmation(sandbox, monkeypatch):
    monkeypatch.setattr(m5, "stage2", lambda: (_ for _ in ()).throw(AssertionError("ran")))
    with pytest.raises(SystemExit, match="confirm-single-test-run"):
        m5.main(["--run-test"])


def test_a_mode_is_required():
    with pytest.raises(SystemExit):
        m5.main([])


# --- Reproduction and frozen operating point ----------------------------------------------------


def _saved(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"id": np.arange(100, 100 + n), "gru_seed42": rng.random(n).astype(np.float32)})


def test_reproduction_check_accepts_identical_probabilities():
    saved = _saved()
    m5.check_reproduces(saved["id"].to_numpy(), saved["gru_seed42"].to_numpy(), saved, "gru_seed42")


def test_reproduction_check_rejects_a_one_ulp_difference():
    saved = _saved()
    p = saved["gru_seed42"].to_numpy().copy()
    p[7] = np.nextafter(p[7], np.float32(1))
    with pytest.raises(SystemExit, match="1 of 50 flights differ"):
        m5.check_reproduces(saved["id"].to_numpy(), p, saved, "gru_seed42")


@pytest.mark.parametrize("ids", ["reordered", "missing"])
def test_reproduction_check_rejects_different_flights(ids):
    saved = _saved()
    p = saved["gru_seed42"].to_numpy()
    flight_ids = saved["id"].to_numpy()
    if ids == "reordered":
        flight_ids = flight_ids[::-1]
    else:
        flight_ids, p = flight_ids[:-1], p[:-1]
    with pytest.raises(SystemExit, match="flights differ"):
        m5.check_reproduces(flight_ids, p, saved, "gru_seed42")


def test_operating_point_check_matches_d312_values():
    assert m5.D312 == {"slope": 0.601708, "intercept": -1.106954, "threshold": 0.394665}
    m5.check_frozen_operating_point(
        {"slope": 0.6017081, "intercept": -1.1069538, "threshold": 0.3946649}
    )


@pytest.mark.parametrize(
    "key, value", [("slope", 0.601710), ("intercept", -1.10695), ("threshold", None)]
)
def test_operating_point_check_rejects_any_difference(key, value):
    params = dict(m5.D312)
    params[key] = value
    with pytest.raises(SystemExit, match=key):
        m5.check_frozen_operating_point(params)


def test_rebuilt_calibrator_follows_the_reported_logit_line():
    rng = np.random.default_rng(1)
    n = 6000
    dates = np.where(np.arange(n) < n // 2, "2013-09-15", "2013-10-15")
    p = rng.uniform(0.02, 0.9, n).astype(np.float32)
    logit = np.log(p / (1 - p))
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.6 * logit - 1.1)))).astype(int)
    frame = pd.DataFrame({"date": pd.to_datetime(dates), "is_delayed": y})
    calibrator, params = m5.frozen_operating_point(p, frame)
    assert params["slope"] == pytest.approx(0.6, abs=0.1)
    assert params["intercept"] == pytest.approx(-1.1, abs=0.15)
    expected = 1 / (1 + np.exp(-(params["slope"] * logit + params["intercept"])))
    np.testing.assert_allclose(calibrator(p.astype(float)), expected, rtol=1e-6)
    if params["threshold"] is not None:
        octo = frame["date"].dt.month.to_numpy() == 10
        flagged = calibrator(p[octo].astype(float)) >= params["threshold"]
        assert flagged.sum() == params["october"]["flagged"]
        assert y[octo][flagged].mean() >= m5.MIN_PRECISION


def test_threshold_metrics_count_flights_at_or_above_threshold():
    y = np.array([1, 0, 1, 1, 0, 0])
    q = np.array([0.9, 0.5, 0.5, 0.1, 0.2, 0.49])
    out = m5.threshold_metrics(y, q, 0.5)
    assert out["flagged"] == 3
    assert out["precision"] == pytest.approx(2 / 3)
    assert out["recall"] == pytest.approx(2 / 3)
    none = m5.threshold_metrics(y, q, 0.95)
    assert none["flagged"] == 0 and none["precision"] is None and none["recall"] == 0


# --- Past-only history: adding later days never changes earlier predictions ---------------------


def test_later_days_do_not_change_earlier_predictions(synthetic_config):
    """Building on a longer period (as Jan-Dec vs Jan-Oct) leaves earlier flights' scores equal."""
    raw = make_raw(n_days=40, per_day=30)
    short = raw[raw["date"] <= "2013-02-02"].reset_index(drop=True)
    table_s, tables_s = m5.build_inputs(short, synthetic_config)
    table_l, tables_l = m5.build_inputs(raw, synthetic_config)

    train = table_s[table_s["split"] == "train"].reset_index(drop=True)
    numeric = [c for c in train.columns if c not in {"id", "date", "split", "is_delayed",
                                                     "origin", "dest", "carrier"}
               and pd.api.types.is_numeric_dtype(train[c])]
    prep = MLPPreprocessor(numeric, ["origin", "dest", "carrier"], min_count=1).fit(train)
    scaler = SequenceScaler.fit(tables_s, tables_s.flight_indices(train), 7)
    set_torch_seed(0)
    model = SequenceDelayModel(prep.input_dim_, 10, 7, 8, encoder="gru").eval()

    window = ("2013-01-26", "2013-02-02")
    rows_s, rows_l = m5.rows_between(table_s, window), m5.rows_between(table_l, window)
    assert rows_s["id"].tolist() == rows_l["id"].tolist() and len(rows_s) > 0
    p_s = m5.score(rows_s, tables_s, model, prep, scaler, 7)
    p_l = m5.score(rows_l, tables_l, model, prep, scaler, 7)
    np.testing.assert_array_equal(p_s, p_l)
    # The later rows exist only in the long build and get their own past-only sequences.
    later = m5.rows_between(table_l, ("2013-02-03", "2013-02-09"))
    assert len(later) > 0
    assert np.isfinite(m5.score(later, tables_l, model, prep, scaler, 7)).all()
