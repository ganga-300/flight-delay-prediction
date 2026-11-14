"""Member 5: D-203 evaluation of saved September-October predictions."""

import numpy as np
import pandas as pd
import pytest

from flight_delay.evaluation.evaluate import (
    attach_meta,
    evaluate_run,
    fit_sigmoid,
    group_table,
    load_sep_oct,
    run_columns,
    run_evaluation,
    summarise_runs,
    threshold_text,
)
from flight_delay.models.calibration import calibrate


def _frame(n_per_day=60, seed=0, models=("lr", "nn"), seeds=(42, 43), shift=1.5):
    """Synthetic Sep-Oct frame: lr is one over-confident run, nn has seed columns."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2013-09-01", "2013-10-31", freq="D").repeat(n_per_day)
    n = len(dates)
    y = (rng.random(n) < 0.2).astype(int)
    out = pd.DataFrame({"id": np.arange(n), "date": dates, "is_delayed": y})

    def probs(signal, s):
        r = np.random.default_rng(s)
        z = signal * y + r.normal(size=n)
        return 1 / (1 + np.exp(-(1.5 * z - 1.5 + shift)))

    for m in models:
        if m == "lr":
            out["lr"] = probs(1.2, 1)
        else:
            for s in seeds:
                out[f"{m}_seed{s}"] = probs(1.2, s)
    meta = pd.DataFrame({
        "id": np.arange(n),
        "carrier": rng.choice(["UA", "AA", "DL"], n),
        "origin": rng.choice(["EWR", "JFK", "LGA"], n),
        "dep_hour": rng.integers(5, 23, n),
    })
    return out, meta


def _fake_bootstrap(dates, y, probs, pairs, seed=42):
    rows = [{"comparison": f"{a} - {b}", "difference": 0.01, "ci_low": -0.01, "ci_high": 0.03,
             "excludes_zero": False} for a, b in pairs]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- loading


def test_loader_rejects_november_rows():
    frame, _ = _frame()
    frame.loc[0, "date"] = pd.Timestamp("2013-11-01")
    with pytest.raises(ValueError, match="sealed"):
        load_sep_oct(frame)


def test_loader_rejects_duplicates_and_non_probabilities():
    frame, _ = _frame()
    dup = frame.copy()
    dup.loc[1, "id"] = dup.loc[0, "id"]
    with pytest.raises(ValueError, match="duplicate"):
        load_sep_oct(dup)
    bad = frame.copy()
    bad.loc[0, "lr"] = 2.5  # a logit, not a probability
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        load_sep_oct(bad)


def test_run_columns_finds_seeds_and_single_runs():
    frame = pd.DataFrame(columns=["lr", "gru_seed42", "gru_seed43", "gru_no_history_seed42"])
    assert run_columns(frame, "gru") == ["gru_seed42", "gru_seed43"]
    assert run_columns(frame, "lr") == ["lr"]
    with pytest.raises(KeyError):
        run_columns(frame, "mlp")


def test_attach_meta_requires_every_flight():
    frame, meta = _frame()
    assert len(attach_meta(frame, meta)) == len(frame)
    with pytest.raises(ValueError, match="no carrier"):
        attach_meta(frame, meta.iloc[1:])


# ---------------------------------------------------------------- calibration and threshold


def test_sigmoid_matches_calibrating_a_logistic_regression_directly():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 3))
    y = (X[:, 0] + rng.normal(size=4000) > 1).astype(int)
    model = LogisticRegression(class_weight="balanced").fit(X[:2000], y[:2000])
    direct = calibrate(model, X[2000:3000], y[2000:3000], method="sigmoid")
    p = model.predict_proba(X)[:, 1]
    ours = fit_sigmoid(p[2000:3000], y[2000:3000])
    np.testing.assert_allclose(ours(p[3000:]), direct.predict_proba(X[3000:])[:, 1], atol=1e-6)


def test_calibration_uses_september_only_and_threshold_october_only():
    frame, _ = _frame()
    frame = load_sep_oct(frame)
    base = evaluate_run("lr", "lr", frame).metrics
    sep = frame["date"].dt.month == 9

    oct_changed = frame.copy()
    oct_changed.loc[~sep, "is_delayed"] = 1 - oct_changed.loc[~sep, "is_delayed"]
    r = evaluate_run("lr", "lr", oct_changed).metrics
    assert r["sep_sigmoid_brier"] == base["sep_sigmoid_brier"]  # calibrator untouched

    sep_changed = frame.copy()
    sep_changed.loc[sep, "is_delayed"] = 1 - sep_changed.loc[sep, "is_delayed"]
    r2 = evaluate_run("lr", "lr", sep_changed).metrics
    assert r2["oct_sigmoid_mean_prob"] != base["oct_sigmoid_mean_prob"]  # calibrator changed
    assert r2["oct_raw_brier"] == base["oct_raw_brier"]


def test_calibration_reduces_october_error_of_overconfident_scores():
    frame, _ = _frame(shift=2.5)
    m = evaluate_run("lr", "lr", load_sep_oct(frame)).metrics
    assert m["oct_sigmoid_brier"] < m["oct_raw_brier"]
    assert m["oct_sigmoid_ece_10"] < m["oct_raw_ece_10"]
    assert m["sepoct_pr_auc"] > 0.2


def test_infeasible_threshold_is_recorded_not_lowered():
    frame, _ = _frame()
    october = frame["date"].dt.month == 10
    top = frame.loc[october, "lr"].idxmax()
    frame.loc[top, "is_delayed"] = 0  # the most confident October flight is on time
    m = evaluate_run("lr", "lr", load_sep_oct(frame), min_precision=0.95).metrics
    assert m["oct_threshold_feasible"] is False
    assert np.isnan(m["oct_threshold"]) and m["oct_flagged"] == 0


def test_feasible_threshold_reports_precision_and_flags():
    frame, _ = _frame()
    m = evaluate_run("lr", "lr", load_sep_oct(frame), min_precision=0.25).metrics
    assert m["oct_threshold_feasible"] is True
    assert m["oct_precision"] >= 0.25 and m["oct_flagged"] > 0


# ---------------------------------------------------------------- seeds


def test_seed_summary_mean_sd_and_single_run_sd_zero():
    per_run = pd.DataFrame({
        "model": ["a", "a", "a", "lr"],
        **{c: [0.1, 0.2, 0.3, 0.5] for c in [
            "sepoct_pr_auc", "sepoct_roc_auc", "oct_raw_brier", "oct_sigmoid_brier",
            "oct_raw_ece_10", "oct_sigmoid_ece_10", "oct_raw_mean_prob",
            "oct_sigmoid_mean_prob", "sep_raw_brier", "sep_sigmoid_brier", "sep_raw_ece_10",
            "sep_sigmoid_ece_10"]},
        "oct_threshold_feasible": [True, False, True, False],
        "oct_threshold": [0.4, np.nan, 0.5, np.nan],
        "oct_precision": [0.6, np.nan, 0.7, np.nan],
        "oct_recall": [0.01, np.nan, 0.02, np.nan],
        "oct_flagged": [10, 0, 30, 0],
    })
    s = summarise_runs(per_run, {"a": "A"}).set_index("model")
    assert s.loc["a", "sepoct_pr_auc_mean"] == pytest.approx(0.2)
    assert s.loc["a", "sepoct_pr_auc_sd"] == pytest.approx(0.1)
    assert s.loc["lr", "sepoct_pr_auc_sd"] == 0.0
    assert s.loc["a", "oct_threshold"].startswith("2/3 feasible")
    assert "flags 10-30 flights" in s.loc["a", "oct_threshold"]
    assert s.loc["lr", "oct_threshold"] == "infeasible"


def test_threshold_text_all_runs_infeasible():
    g = pd.DataFrame({"oct_threshold_feasible": [False] * 5, "oct_threshold": [np.nan] * 5,
                      "oct_flagged": [0] * 5, "oct_recall": [np.nan] * 5})
    assert threshold_text(g) == "infeasible (5/5 runs)"


# ---------------------------------------------------------------- error analysis


def test_group_table_is_threshold_free_with_minimum_rows():
    test = pd.DataFrame({
        "is_delayed": [1, 1, 0, 0] * 60 + [1, 0],
        "carrier": ["UA"] * 240 + ["AA"] * 2,
        "origin": ["EWR"] * 242,
        "dep_hour": [6] * 242,
    })
    prob = np.array([0.9, 0.2, 0.8, 0.1] * 60 + [0.9, 0.1])
    g = group_table(test, prob, min_rows=200).set_index(["meta_column", "group"])
    assert {"miss_rate", "false_alarm_rate"}.isdisjoint(g.columns)
    ua = g.loc[("carrier", "UA")]
    assert ua["observed_rate"] == pytest.approx(0.5)
    assert ua["calibration_gap"] == pytest.approx(0.0)
    assert np.isfinite(ua["pr_auc"])
    assert np.isnan(g.loc[("carrier", "AA"), "pr_auc"])


# ---------------------------------------------------------------- pipeline


def test_pipeline_writes_outputs_with_months(tmp_path):
    frame, meta = _frame()
    frame = attach_meta(load_sep_oct(frame), meta)
    calls = {}

    def spy(dates, y, probs, pairs, seed=42):
        calls.update(n=len(y), pairs=pairs, seed=seed, runs={k: len(v) for k, v in probs.items()})
        return _fake_bootstrap(dates, y, probs, pairs, seed)

    result = run_evaluation(frame, {"lr": "LR", "nn": "NN"}, tmp_path, reference="lr",
                            bootstrap_fn=spy, error_analysis_models=["lr", "nn"],
                            notes="## Limitations\n- note")
    for name in ["comparison_table.csv", "per_run_metrics.csv", "bootstrap_differences.csv",
                 "error_analysis_groups.csv", "reliability_october.png", "pr_auc_bar.png",
                 "bootstrap_differences.png", "evaluation_summary.md"]:
        assert (tmp_path / name).exists(), name
    assert calls == {"n": len(frame), "pairs": [("nn", "lr")], "seed": 42,
                     "runs": {"lr": 1, "nn": 2}}
    assert set(result.groups["month"]) == {"October"}
    text = (tmp_path / "evaluation_summary.md").read_text()
    for header in ["PR-AUC (Sep-Oct)", "Brier calibrated (Oct)", "ECE raw (Oct)",
                   "Threshold, precision ≥ 0.60 (Oct)", "(Sep, in-sample)",
                   "95% interval (Sep-Oct)"]:
        assert header in text, header
    assert "no winner on PR-AUC is declared" in text
    assert "isotonic" not in text.lower()
    assert "- note" in text


def test_pipeline_names_only_differences_that_exclude_zero(tmp_path):
    frame, _ = _frame()
    frame = load_sep_oct(frame)

    def real_gap(dates, y, probs, pairs, seed=42):
        return pd.DataFrame([{"comparison": "nn - lr", "difference": -0.03, "ci_low": -0.05,
                              "ci_high": -0.01, "excludes_zero": True}])

    run_evaluation(frame, {"lr": "LR", "nn": "NN"}, tmp_path, reference="lr",
                   bootstrap_fn=real_gap)
    text = (tmp_path / "evaluation_summary.md").read_text()
    assert "lower: nn - lr" in text and "no winner" not in text
