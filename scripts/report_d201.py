"""D-201 August candidate comparison, XGBoost importance, and D-202."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flight_delay.config import load_config  # noqa: E402  # noqa: E402
from flight_delay.data.load import load_dev_raw  # noqa: E402  # noqa: E402
from flight_delay.features.build import (  # noqa: E402  # noqa: E402
    CATEGORICAL,
    build_feature_table,
    numeric_features,
)
from flight_delay.features.encoding import FeatureEncoder  # noqa: E402  # noqa: E402
from flight_delay.models.logistic import build_logistic_pipeline  # noqa: E402  # noqa: E402
from flight_delay.models.xgboost_model import build_xgboost  # noqa: E402  # noqa: E402

OUT = ROOT / "models/d201"
RESULTS = ROOT / "reports/results"
FIGURES = ROOT / "reports/figures"

def main():
    config = load_config(str(ROOT / "configs/config.yaml"))
    assert config["paths"]["raw"] == "data/interim/flights_dev.csv"
    assert config["seed"] == 42

    selection = json.loads((OUT / "selection.json").read_text())
    final = json.loads((OUT / "final_training.json").read_text())
    assert selection["selected_model"] == final["selected_model"]
    assert selection["selected_model"] == "logreg"
    assert (OUT / "final_model.joblib").exists()
    assert (OUT / "final_encoder.json").exists()

    numeric = list(numeric_features(config))
    categorical = list(CATEGORICAL)
    features = numeric + categorical
    assert len(numeric) == 49 and len(categorical) == 10
    assert len(set(features)) == 59

    table = build_feature_table(load_dev_raw(config, ROOT), config)
    dates = pd.to_datetime(table["date"])
    assert dates.min() >= pd.Timestamp("2013-01-01")
    assert dates.max() <= pd.Timestamp("2013-10-31")

    fit = table.loc[dates.between("2013-01-01", "2013-07-31")]
    aug = table.loc[dates.between("2013-08-01", "2013-08-31")]
    assert len(fit) and len(aug)

    encoder = FeatureEncoder(
        categorical=categorical,
        numeric=numeric,
        min_count=config["features"]["categorical_min_count"],
    )
    encoder.fit(fit)
    X_fit = encoder.transform(fit)[features]
    X_aug = encoder.transform(aug)[features]
    y_fit = fit["is_delayed"]
    y_aug = aug["is_delayed"]

    reports = {
        r["model"]: r for r in selection["candidate_results"]
    }
    assert set(reports) == {"logreg", "xgboost"}

    predictions = {}
    scores = []
    xgb_model = None

    for name in ("logreg", "xgboost"):
        best = reports[name]["best_august_candidate"]
        params = best["parameters"].copy()

        if name == "logreg":
            model = build_logistic_pipeline(
                numeric, categorical, params
            )
            model.fit(X_fit, y_fit)
        else:
            model = build_xgboost(params)
            model.fit(
                X_fit,
                y_fit,
                eval_set=[(X_aug, y_aug)],
                verbose=False,
            )
            assert model.best_iteration == best["best_iteration"]
            xgb_model = model

        prob = model.predict_proba(X_aug)[:, 1]
        predictions[name] = prob

        pr = float(average_precision_score(y_aug, prob))
        roc = float(roc_auc_score(y_aug, prob))

        assert abs(pr - best["pr_auc"]) < 1e-6
        assert abs(roc - best["roc_auc"]) < 1e-6

        scores.append({
            "model": name,
            "august_pr_auc": pr,
            "august_roc_auc": roc,
            "best_iteration": best.get("best_iteration"),
            "parameters": json.dumps(params, sort_keys=True),
        })

        print(f"{name}: PR-AUC={pr:.6f}, ROC-AUC={roc:.6f}", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(scores).to_csv(
        RESULTS / "d201_model_comparison.csv", index=False
    )

    for kind in ("pr", "roc"):
        fig, ax = plt.subplots(figsize=(8, 6))

        for name, prob in predictions.items():
            if kind == "pr":
                PrecisionRecallDisplay.from_predictions(
                    y_aug, prob, name=name, ax=ax
                )
            else:
                RocCurveDisplay.from_predictions(
                    y_aug, prob, name=name, ax=ax
                )

        ax.set_title(f"August 2013 {kind.upper()} comparison")
        fig.tight_layout()
        fig.savefig(
            FIGURES / f"d201_{kind}_comparison.png", dpi=160
        )
        plt.close(fig)

    booster = xgb_model.get_booster()
    gain = booster.get_score(importance_type="gain")
    gain_frame = pd.DataFrame({
        "feature": features,
        "gain": [float(gain.get(f, 0)) for f in features],
    }).sort_values("gain", ascending=False)

    gain_frame.to_csv(
        RESULTS / "d201_xgboost_gain.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    top = gain_frame.head(20).iloc[::-1]
    ax.barh(top["feature"], top["gain"])
    ax.set_title("XGBoost gain importance")
    fig.tight_layout()
    fig.savefig(FIGURES / "d201_xgboost_gain.png", dpi=160)
    plt.close(fig)

    # Descriptive importance only: fixed-size deterministic August sample.
    sample_size = min(2000, len(X_aug))
    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(len(X_aug), sample_size, replace=False))

    perm = permutation_importance(
        xgb_model,
        X_aug.iloc[idx],
        y_aug.iloc[idx],
        scoring="average_precision",
        n_repeats=2,
        random_state=42,
        n_jobs=1,
    )

    permutation_frame = pd.DataFrame({
        "feature": features,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)

    permutation_frame.to_csv(
        RESULTS / "d201_xgboost_permutation.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    top = permutation_frame.head(20).iloc[::-1]
    ax.barh(top["feature"], top["importance_mean"])
    ax.set_title("XGBoost permutation importance — August sample")
    fig.tight_layout()
    fig.savefig(
        FIGURES / "d201_xgboost_permutation.png", dpi=160
    )
    plt.close(fig)

    lr = reports["logreg"]["best_august_candidate"]
    xb = reports["xgboost"]["best_august_candidate"]

    decision = f"""## D-202 — Final model selection

Status: Completed after D-201 candidate training.

- Selection dataset: August 2013.
- Candidate fitting dataset: January–July 2013.
- Selection metric: highest August average precision (PR-AUC).
- Winning model: Logistic Regression.
- Logistic Regression PR-AUC: {lr["pr_auc"]:.6f}.
- Logistic Regression ROC-AUC: {lr["roc_auc"]:.6f}.
- Best XGBoost PR-AUC: {xb["pr_auc"]:.6f}.
- Best XGBoost ROC-AUC: {xb["roc_auc"]:.6f}.
- Selected hyperparameters: `{json.dumps(lr["parameters"], sort_keys=True)}`.
- Final model retrained on January–August 2013.
- FeatureEncoder and model preprocessing refitted on January–August.
- August metrics are from January–July-fitted candidates, not the final refit.
- Fixed feature set: 49 numeric + 10 categorical.
- XGBoost gain and permutation importance are reporting-only.
- Permutation importance: {sample_size} August examples, seed 42, 2 repeats.
- No calibration or operating-threshold selection in this workflow.
- November–December sealed holdout not accessed.

"""

    log = ROOT / "docs/decision_log.md"
    content = log.read_text()
    if "## D-202 — Final model selection" in content:
        raise RuntimeError("D-202 already exists; refusing duplicate")
    log.write_text(content.rstrip() + "\n\n" + decision)

    print("TASKS 8–10 COMPLETE")
    print("Comparison reports:", RESULTS)
    print("Figures:", FIGURES)
    print("D-202 updated:", log)

if __name__ == "__main__":
    main()
