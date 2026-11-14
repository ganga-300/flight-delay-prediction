"""Common evaluation of saved September-October predictions (Member 5, D-203 protocol).

Input: one frame of September-October 2013 flights with ``id``, ``date``, ``is_delayed`` and one
probability column per model run (``<model>_seed<k>``, or ``<model>`` for a single run), e.g.
``reports/results/phase2_member4/sep_oct_predictions.parquet``. Nothing is retrained.

Each month has one job (docs/decision_log.md, D-203):

* September: fit a sigmoid calibrator per run (``models.calibration.calibrate``).
* October:   out-of-sample check of the calibrator, and the operating threshold
             (``models.operating_point.choose_threshold``, precision >= 0.60). If no threshold
             reaches the target the run is recorded as infeasible; the target is never lowered.
* September-October: ranking metrics (PR-AUC, ROC-AUC), which calibration does not change, and
             paired day-level bootstrap differences.

Every reported number names the month it was computed on. November-December is never read; the
loader rejects any row outside September-October.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import BaseEstimator, ClassifierMixin  # noqa: E402

from flight_delay.evaluation.metrics import (  # noqa: E402
    brier,
    expected_calibration_error,
    pr_auc,
    roc_auc,
)
from flight_delay.models.calibration import calibrate  # noqa: E402
from flight_delay.models.operating_point import choose_threshold  # noqa: E402

SEPTEMBER = ("2013-09-01", "2013-09-30")
OCTOBER = ("2013-10-01", "2013-10-31")
PERIOD = (SEPTEMBER[0], OCTOBER[1])
MIN_PRECISION = 0.60  # D-015
N_BINS = 10
MIN_GROUP_ROWS = 200
META_COLUMNS = ("carrier", "origin", "dep_hour")
_EPS = 1e-7

# Reference categorical palette (dataviz skill, light mode), assigned in fixed order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

BootstrapFn = Callable[..., pd.DataFrame]


# --------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------


def load_sep_oct(path_or_frame: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Load and validate saved September-October predictions."""
    frame = (path_or_frame.copy() if isinstance(path_or_frame, pd.DataFrame)
             else pd.read_parquet(path_or_frame))
    for col in ("id", "date", "is_delayed"):
        if col not in frame.columns:
            raise ValueError(f"missing column {col!r}")
    frame["date"] = pd.to_datetime(frame["date"])
    lo, hi = pd.Timestamp(PERIOD[0]), pd.Timestamp(PERIOD[1])
    if frame["date"].min() < lo or frame["date"].max() > hi:
        raise ValueError("rows outside September-October 2013; November-December stays sealed")
    if frame["id"].duplicated().any():
        raise ValueError("duplicate flight ids")
    if not frame["is_delayed"].isin([0, 1]).all():
        raise ValueError("is_delayed must be 0/1")
    for col in frame.columns.difference(["id", "date", "is_delayed"]):
        p = frame[col].to_numpy(float)
        if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError(f"{col}: values must be probabilities in [0, 1]")
    return frame.reset_index(drop=True)


def run_columns(frame: pd.DataFrame, model: str) -> list[str]:
    """Probability columns of one model: ``<model>_seed<k>`` or a single ``<model>`` column."""
    seeds = sorted(c for c in frame.columns if re.fullmatch(rf"{re.escape(model)}_seed\d+", c))
    if seeds:
        return seeds
    if model in frame.columns:
        return [model]
    raise KeyError(f"no prediction columns for model {model!r}")


def attach_meta(frame: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Add carrier, origin and departure hour, matched on flight id (every flight must match)."""
    missing = [c for c in ("id", *META_COLUMNS) if c not in meta.columns]
    if missing:
        raise ValueError(f"meta is missing columns {missing}")
    merged = frame.merge(meta[["id", *META_COLUMNS]], on="id", how="left", validate="1:1")
    if merged[list(META_COLUMNS)].isna().any().any():
        raise ValueError("some flights have no carrier/origin/departure hour")
    return merged


# --------------------------------------------------------------------------------------------
# Calibration (September) and threshold (October)
# --------------------------------------------------------------------------------------------


class FixedProbabilities(ClassifierMixin, BaseEstimator):
    """Wrap saved probabilities so D-203's ``calibrate`` can fit a sigmoid on them.

    The single input column is the probability. ``decision_function`` returns its logit, which
    for Logistic Regression equals the model's own decision function, so the sigmoid matches
    calibrating the fitted model directly (D-204 is reproduced exactly).
    """

    def fit(self, X=None, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        p = np.clip(np.asarray(X, float)[:, 0], _EPS, 1 - _EPS)
        return np.log(p / (1 - p))

    def predict_proba(self, X):
        p = np.asarray(X, float)[:, 0]
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (np.asarray(X, float)[:, 0] >= 0.5).astype(int)


def fit_sigmoid(p_september: np.ndarray, y_september: np.ndarray) -> Callable:
    """Fit a sigmoid calibrator on September; return a function mapping probabilities."""
    calibrated = calibrate(FixedProbabilities().fit(), np.asarray(p_september).reshape(-1, 1),
                           np.asarray(y_september), method="sigmoid")
    return lambda p: calibrated.predict_proba(np.asarray(p).reshape(-1, 1))[:, 1]


def probability_scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "brier": brier(y, p),
        "ece_10": expected_calibration_error(y, p, n_bins=N_BINS),
        "mean_prob": float(np.mean(p)),
    }


@dataclass
class RunResult:
    model: str
    run: str
    metrics: dict
    october_raw: np.ndarray
    october_calibrated: np.ndarray


def evaluate_run(model: str, run: str, frame: pd.DataFrame,
                 min_precision: float = MIN_PRECISION) -> RunResult:
    """All metrics for one probability column, each tagged with its month."""
    y = frame["is_delayed"].to_numpy(int)
    p = frame[run].to_numpy(float)
    sep = frame["date"].between(*SEPTEMBER).to_numpy()
    octo = frame["date"].between(*OCTOBER).to_numpy()
    calibrator = fit_sigmoid(p[sep], y[sep])
    q_sep, q_oct = calibrator(p[sep]), calibrator(p[octo])

    m = {"model": model, "run": run,
         "sepoct_pr_auc": pr_auc(y, p), "sepoct_roc_auc": roc_auc(y, p)}
    for prefix, yy, pp in [("sep_raw", y[sep], p[sep]), ("sep_sigmoid", y[sep], q_sep),
                           ("oct_raw", y[octo], p[octo]), ("oct_sigmoid", y[octo], q_oct)]:
        m.update({f"{prefix}_{k}": v for k, v in probability_scores(yy, pp).items()})
    op = choose_threshold(y[octo], q_oct, min_precision=min_precision)
    m["oct_threshold_feasible"] = bool(op["feasible"])
    if op["feasible"]:
        flagged = q_oct >= op["threshold"]
        m.update({"oct_threshold": op["threshold"], "oct_precision": op["precision"],
                  "oct_recall": op["recall"], "oct_flagged": int(flagged.sum())})
    else:
        m.update({"oct_threshold": np.nan, "oct_precision": np.nan, "oct_recall": np.nan,
                  "oct_flagged": 0})
    return RunResult(model, run, m, p[octo], q_oct)


# --------------------------------------------------------------------------------------------
# Seed aggregation
# --------------------------------------------------------------------------------------------

METRIC_COLUMNS = [
    "sepoct_pr_auc", "sepoct_roc_auc",
    "oct_raw_brier", "oct_sigmoid_brier", "oct_raw_ece_10", "oct_sigmoid_ece_10",
    "oct_raw_mean_prob", "oct_sigmoid_mean_prob",
    "sep_raw_brier", "sep_sigmoid_brier", "sep_raw_ece_10", "sep_sigmoid_ece_10",
]


def _sd(values: pd.Series) -> float:
    """Sample sd over seeds; 0 for a single deterministic run."""
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def summarise_runs(per_run: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    """Mean and sd over seeds per model, plus the October threshold outcome."""
    rows = []
    for model, g in per_run.groupby("model", sort=False):
        row = {"model": model, "label": labels.get(model, model), "n_runs": len(g)}
        for col in METRIC_COLUMNS:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_sd"] = _sd(g[col])
        feasible = g[g["oct_threshold_feasible"]]
        row["oct_threshold_feasible_runs"] = len(feasible)
        row["oct_threshold"] = threshold_text(g)
        for col in ("oct_threshold", "oct_precision", "oct_recall", "oct_flagged"):
            row[f"{col}_feasible_mean"] = float(feasible[col].mean()) if len(feasible) else np.nan
        row["oct_flagged_feasible_min"] = int(feasible["oct_flagged"].min()) if len(feasible) else 0
        row["oct_flagged_feasible_max"] = int(feasible["oct_flagged"].max()) if len(feasible) else 0
        rows.append(row)
    return pd.DataFrame(rows)


def threshold_text(g: pd.DataFrame) -> str:
    """'infeasible' or 'k/n feasible: t = mean +/- sd, flags a-b flights, recall r'."""
    feasible = g[g["oct_threshold_feasible"]]
    n = len(g)
    if feasible.empty:
        return f"infeasible ({n}/{n} runs)" if n > 1 else "infeasible"
    t = feasible["oct_threshold"]
    lo, hi = int(feasible["oct_flagged"].min()), int(feasible["oct_flagged"].max())
    flags = f"{lo}" if lo == hi else f"{lo}-{hi}"
    return (f"{len(feasible)}/{n} feasible: t = {t.mean():.3f} ± {_sd(t):.3f}, "
            f"flags {flags} flights, recall {feasible['oct_recall'].mean():.4f}")


# --------------------------------------------------------------------------------------------
# Error analysis (October, threshold-free)
# --------------------------------------------------------------------------------------------


def group_table(frame: pd.DataFrame, prob: np.ndarray,
                min_rows: int = MIN_GROUP_ROWS) -> pd.DataFrame:
    """Per-group observed rate, mean probability, calibration gap and PR-AUC."""
    work = frame.assign(_p=prob)
    rows = []
    for col in META_COLUMNS:
        for group, g in work.groupby(col, sort=True):
            y, p = g["is_delayed"].to_numpy(int), g["_p"].to_numpy(float)
            ok = len(g) >= min_rows and 0 < y.sum() < len(y)
            rows.append({
                "meta_column": col,
                "group": group,
                "n": len(g),
                "observed_rate": float(y.mean()),
                "mean_predicted": float(p.mean()),
                "calibration_gap": float(p.mean() - y.mean()),
                "pr_auc": pr_auc(y, p) if ok else np.nan,
            })
    return pd.DataFrame(rows)


def error_analysis(october: pd.DataFrame, results: list[RunResult],
                   labels: dict[str, str]) -> pd.DataFrame:
    """Group metrics on calibrated October probabilities, mean and sd over each model's seeds."""
    per_run = []
    for r in results:
        t = group_table(october, r.october_calibrated)
        t.insert(0, "model", r.model)
        per_run.append(t)
    long = pd.concat(per_run, ignore_index=True)
    keys = ["model", "meta_column", "group"]
    agg = long.groupby(keys, sort=False).agg(
        n=("n", "first"),
        observed_rate=("observed_rate", "first"),
        mean_predicted_mean=("mean_predicted", "mean"),
        mean_predicted_sd=("mean_predicted", _sd),
        calibration_gap_mean=("calibration_gap", "mean"),
        calibration_gap_sd=("calibration_gap", _sd),
        pr_auc_mean=("pr_auc", "mean"),
        pr_auc_sd=("pr_auc", lambda s: _sd(s.dropna()) if s.notna().any() else np.nan),
    ).reset_index()
    agg.insert(1, "label", agg["model"].map(lambda m: labels.get(m, m)))
    agg.insert(2, "month", "October")
    return agg


# --------------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------------


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _reliability(y: np.ndarray, p: np.ndarray, min_count: int = 20):
    bins = np.minimum((p * N_BINS).astype(int), N_BINS - 1)
    xs, ys = [], []
    for b in range(N_BINS):
        m = bins == b
        if m.sum() >= min_count:
            xs.append(p[m].mean())
            ys.append(y[m].mean())
    return np.array(xs), np.array(ys)


def plot_reliability(y_oct: np.ndarray, results: list[RunResult], labels: dict[str, str],
                     path: Path) -> None:
    models = list(dict.fromkeys(r.model for r in results))
    cols = min(3, len(models))
    rows = int(np.ceil(len(models) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4 * rows), squeeze=False,
                             facecolor=SURFACE)
    for ax, model in zip(axes.flat, models, strict=False):
        _style(ax)
        ax.plot([0, 1], [0, 1], color=INK_2, linestyle="--", linewidth=1, label="Perfect")
        runs = [r for r in results if r.model == model]
        for i, r in enumerate(runs):
            for p, name, color in [(r.october_raw, "Raw", PALETTE[0]),
                                   (r.october_calibrated, "Sigmoid (fit on September)",
                                    PALETTE[1])]:
                xs, ys = _reliability(y_oct, p)
                ax.plot(xs, ys, color=color, linewidth=1.4, marker="o", markersize=3,
                        alpha=0.85, label=name if i == 0 else None)
        ax.set_title(f"{labels.get(model, model)} ({len(runs)} run{'s' * (len(runs) > 1)})",
                     fontsize=9, color=INK)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability", fontsize=8, color=INK_2)
        ax.set_ylabel("Observed delay rate", fontsize=8, color=INK_2)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    for ax in list(axes.flat)[len(models):]:
        ax.set_visible(False)
    fig.suptitle("Reliability on October (one line per seed), before and after calibration",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_pr_auc(summary: pd.DataFrame, path: Path) -> None:
    t = summary.iloc[::-1]
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(t) + 1.4), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    mean, sd = t["sepoct_pr_auc_mean"].to_numpy(), t["sepoct_pr_auc_sd"].to_numpy()
    ax.barh(y, mean, xerr=sd, height=0.6, color=PALETTE[0], ecolor=INK_2, capsize=3)
    for yi, m, s in zip(y, mean, sd, strict=True):
        ax.text(m + s + 0.003, yi, f"{m:.3f} ± {s:.3f}", va="center", fontsize=7, color=INK_2)
    ax.set_yticks(y, t["label"], fontsize=8, color=INK)
    ax.set_xlabel("PR-AUC, September-October (mean ± sd over seeds)", color=INK_2)
    ax.set_xlim(0, float((mean + sd).max()) * 1.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_bootstrap(boot: pd.DataFrame, path: Path) -> None:
    t = boot.iloc[::-1]
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(t) + 1.4), facecolor=SURFACE)
    _style(ax)
    d = t["difference"].to_numpy()
    err = [d - t["ci_low"].to_numpy(), t["ci_high"].to_numpy() - d]
    ax.errorbar(d, y, xerr=err, fmt="o", color=PALETTE[0], ecolor=PALETTE[0], elinewidth=2,
                capsize=4, markersize=6)
    ax.axvline(0, color=INK_2, linestyle="--", linewidth=1)
    ax.set_yticks(y, t["comparison"], fontsize=8, color=INK)
    ax.set_xlabel("PR-AUC difference, September-October (95% day-bootstrap interval)",
                  color=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------------------------


def markdown_table(frame: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    sep = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join("" if v is None else str(v) for v in row) + " |"
            for row in frame.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def pm(mean: float, sd: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def comparison_markdown(summary: pd.DataFrame) -> str:
    rows = []
    for r in summary.itertuples(index=False):
        rows.append({
            "Model": r.label,
            "Runs": r.n_runs,
            "PR-AUC (Sep-Oct)": pm(r.sepoct_pr_auc_mean, r.sepoct_pr_auc_sd),
            "ROC-AUC (Sep-Oct)": pm(r.sepoct_roc_auc_mean, r.sepoct_roc_auc_sd),
            "Brier raw (Oct)": pm(r.oct_raw_brier_mean, r.oct_raw_brier_sd),
            "Brier calibrated (Oct)": pm(r.oct_sigmoid_brier_mean, r.oct_sigmoid_brier_sd),
            "ECE raw (Oct)": pm(r.oct_raw_ece_10_mean, r.oct_raw_ece_10_sd),
            "ECE calibrated (Oct)": pm(r.oct_sigmoid_ece_10_mean, r.oct_sigmoid_ece_10_sd),
            "Threshold, precision ≥ 0.60 (Oct)": r.oct_threshold,
        })
    return markdown_table(pd.DataFrame(rows))


def september_markdown(summary: pd.DataFrame) -> str:
    rows = [{
        "Model": r.label,
        "Brier raw (Sep)": pm(r.sep_raw_brier_mean, r.sep_raw_brier_sd),
        "Brier calibrated (Sep, in-sample)": pm(r.sep_sigmoid_brier_mean, r.sep_sigmoid_brier_sd),
        "ECE raw (Sep)": pm(r.sep_raw_ece_10_mean, r.sep_raw_ece_10_sd),
        "ECE calibrated (Sep, in-sample)": pm(r.sep_sigmoid_ece_10_mean,
                                              r.sep_sigmoid_ece_10_sd),
    } for r in summary.itertuples(index=False)]
    return markdown_table(pd.DataFrame(rows))


def bootstrap_markdown(boot: pd.DataFrame) -> str:
    rows = [{
        "Comparison": r.comparison,
        "PR-AUC difference (Sep-Oct)": f"{r.difference:+.4f}",
        "95% interval (Sep-Oct)": f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}]",
        "Interval excludes 0": "yes" if r.excludes_zero else "no",
    } for r in boot.itertuples(index=False)]
    return markdown_table(pd.DataFrame(rows))


def weakest_groups_markdown(groups: pd.DataFrame, k: int = 3) -> str:
    """Per meta column, the k groups with lowest October PR-AUC averaged over models."""
    rated = groups.dropna(subset=["pr_auc_mean"])
    pivot = rated.pivot_table(index=["meta_column", "group"], columns="label",
                              values="pr_auc_mean", sort=False)
    pivot["Mean over models"] = pivot.mean(axis=1)
    n = rated.groupby(["meta_column", "group"])["n"].first()
    parts = []
    for col in META_COLUMNS:
        if col in pivot.index.get_level_values(0):
            parts.append(pivot.loc[[col]].nsmallest(k, "Mean over models"))
    if not parts:
        return "No group has enough flights."
    table = pd.concat(parts)
    table.insert(0, "Flights (Oct)", n.loc[table.index].to_numpy())
    table = table.reset_index().rename(columns={"meta_column": "Meta column", "group": "Group"})
    for c in table.columns[3:]:
        table[c] = table[c].map(lambda v: f"{v:.3f}")
    return markdown_table(table)


def largest_gap_markdown(groups: pd.DataFrame, k: int = 3) -> str:
    """Per meta column, the k groups with the largest mean |calibration gap| over models."""
    big = groups[groups["n"] >= MIN_GROUP_ROWS]
    pivot = big.pivot_table(index=["meta_column", "group"], columns="label",
                            values="calibration_gap_mean", sort=False)
    pivot["Mean over models"] = pivot.mean(axis=1)
    parts = []
    for col in META_COLUMNS:
        if col in pivot.index.get_level_values(0):
            sub = pivot.loc[[col]]
            parts.append(sub.loc[sub["Mean over models"].abs().nlargest(k).index])
    if not parts:
        return "No group has enough flights."
    table = pd.concat(parts).reset_index().rename(
        columns={"meta_column": "Meta column", "group": "Group"})
    for c in table.columns[2:]:
        table[c] = table[c].map(lambda v: f"{v:+.3f}")
    return markdown_table(table)


# --------------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    per_run: pd.DataFrame
    summary: pd.DataFrame
    bootstrap: pd.DataFrame | None
    groups: pd.DataFrame | None
    results: list[RunResult] = field(default_factory=list)


def evaluate_models(frame: pd.DataFrame, models: dict[str, str],
                    min_precision: float = MIN_PRECISION) -> tuple[pd.DataFrame, list[RunResult]]:
    results = [evaluate_run(m, run, frame, min_precision)
               for m in models for run in run_columns(frame, m)]
    return pd.DataFrame([r.metrics for r in results]), results


def run_evaluation(
    frame: pd.DataFrame,
    models: dict[str, str],
    output_dir: str | Path,
    *,
    reference: str,
    bootstrap_fn: BootstrapFn | None = None,
    error_analysis_models: list[str] | None = None,
    title: str = "Evaluation summary",
    intro: str = "",
    notes: str = "",
    min_precision: float = MIN_PRECISION,
) -> EvaluationResult:
    """Evaluate ``models`` (key -> label) and write tables, plots and ``evaluation_summary.md``.

    ``bootstrap_fn(dates, y, probs, pairs, seed=42)`` must return columns ``comparison``,
    ``difference``, ``ci_low``, ``ci_high``, ``excludes_zero`` (``scripts/train_gru.day_bootstrap``).
    Each model is compared with ``reference``. Error analysis needs the meta columns.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_run, results = evaluate_models(frame, models, min_precision)
    summary = summarise_runs(per_run, models)
    per_run.to_csv(out / "per_run_metrics.csv", index=False)
    summary.to_csv(out / "comparison_table.csv", index=False)
    plot_pr_auc(summary, out / "pr_auc_bar.png")

    boot = None
    if bootstrap_fn is not None:
        probs = {m: [frame[c].to_numpy(float) for c in run_columns(frame, m)] for m in models}
        pairs = [(m, reference) for m in models if m != reference]
        dates = frame["date"].dt.strftime("%Y-%m-%d").to_numpy()
        boot = bootstrap_fn(dates, frame["is_delayed"].to_numpy(int), probs, pairs, seed=42)
        boot.insert(0, "period", "September-October")
        boot.to_csv(out / "bootstrap_differences.csv", index=False)
        plot_bootstrap(boot, out / "bootstrap_differences.png")

    groups = None
    if error_analysis_models:
        october = frame.loc[frame["date"].between(*OCTOBER)].reset_index(drop=True)
        chosen = [r for r in results if r.model in error_analysis_models]
        groups = error_analysis(october, chosen, models)
        groups.to_csv(out / "error_analysis_groups.csv", index=False)
        plot_reliability(october["is_delayed"].to_numpy(int), chosen, models,
                         out / "reliability_october.png")

    write_summary(out / "evaluation_summary.md", title, intro, notes, summary, boot, groups,
                  reference, models)
    return EvaluationResult(per_run, summary, boot, groups, results)


def write_summary(path: Path, title: str, intro: str, notes: str, summary: pd.DataFrame,
                  boot: pd.DataFrame | None, groups: pd.DataFrame | None, reference: str,
                  labels: dict[str, str]) -> None:
    lines = [f"# {title}", ""]
    if intro:
        lines += [intro.strip(), ""]
    lines += [
        "## Model comparison", "",
        "Mean ± sd over each model's runs (seeds). Month of each number in the column name. "
        "PR-AUC and ROC-AUC are computed on raw probabilities; the September sigmoid does not "
        "change the ranking. Calibrated = sigmoid fitted on September, scored on October.", "",
        comparison_markdown(summary), "",
        "September reference (the calibrator's own fitting month, in-sample):", "",
        september_markdown(summary), "",
    ]
    if boot is not None:
        lines += [
            f"## PR-AUC differences vs {labels.get(reference, reference)}", "",
            "Paired day-level bootstrap (1,000 resamples of whole days, seed 42) on "
            "September-October; each model's PR-AUC is the mean over its seeds. A difference "
            "counts only if its 95% interval excludes 0.", "",
            bootstrap_markdown(boot), "",
        ]
        real = boot[boot["excludes_zero"]]
        if real.empty:
            lines += [f"No model's PR-AUC differs from {labels.get(reference, reference)} "
                      "beyond the bootstrap interval on September-October, so no winner on "
                      "PR-AUC is declared.", ""]
        else:
            better = real[real["difference"] > 0]["comparison"].tolist()
            worse = real[real["difference"] < 0]["comparison"].tolist()
            lines += ["Differences whose interval excludes 0 (September-October): "
                      f"higher: {', '.join(better) or 'none'}; lower: {', '.join(worse) or 'none'}. "
                      "All other differences are within bootstrap noise.", ""]
    if groups is not None:
        lines += [
            "## Error analysis (October, calibrated probabilities)", "",
            f"Groups with at least {MIN_GROUP_ROWS} October flights. Values are means over each "
            "model's seeds; per-seed sd is in `error_analysis_groups.csv`. Threshold-based rates "
            "are not reported because the October thresholds are infeasible or flag very few "
            "flights (D-203).", "",
            "Lowest PR-AUC groups (October):", "", weakest_groups_markdown(groups), "",
            "Largest calibration gaps, mean calibrated probability minus observed rate "
            "(October):", "", largest_gap_markdown(groups), "",
        ]
    if notes:
        lines += [notes.strip(), ""]
    path.write_text("\n".join(lines))
