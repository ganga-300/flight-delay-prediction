"""D-201 model training: Jan-Jul tuning, August selection, Jan-Aug refit."""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flight_delay.config import load_config  # noqa: E402
from flight_delay.data.load import load_dev_raw  # noqa: E402
from flight_delay.features.build import (  # noqa: E402
    CATEGORICAL,
    build_feature_table,
    numeric_features,
)
from flight_delay.features.encoding import FeatureEncoder  # noqa: E402
from flight_delay.models.logistic import build_logistic_pipeline  # noqa: E402
from flight_delay.models.persistence import save_model  # noqa: E402
from flight_delay.models.xgboost_model import build_xgboost  # noqa: E402


def split_data(table):
    dates = pd.to_datetime(table["date"])
    if dates.isna().any():
        raise ValueError("Missing flight dates")
    if not dates.between("2013-01-01", "2013-10-31").all():
        raise ValueError("Data outside permitted development period")

    fit = table.loc[dates.between("2013-01-01", "2013-07-31")].copy()
    tune = table.loc[dates.between("2013-08-01", "2013-08-31")].copy()

    if fit.empty or tune.empty:
        raise ValueError("January-July or August data is empty")
    return fit, tune


def make_encoder(cfg, frame, numeric):
    encoder = FeatureEncoder(
        categorical=list(CATEGORICAL),
        numeric=numeric,
        min_count=cfg["features"]["categorical_min_count"],
    )
    encoder.fit(frame)
    return encoder


def evaluate(model, X, y):
    scores = model.predict_proba(X)[:, 1]
    return {
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
    }


def train(cfg, model_name, output_dir, *, finalize=True):
    raw = load_dev_raw(cfg, ROOT)
    table = build_feature_table(raw, cfg)
    fit, tune = split_data(table)

    numeric = list(numeric_features(cfg))
    categorical = list(CATEGORICAL)
    features = numeric + categorical

    if len(numeric) != 49 or len(categorical) != 10:
        raise ValueError("D-201 requires exactly 49 numeric + 10 categorical features")

    encoder = make_encoder(cfg, fit, numeric)
    X_fit = encoder.transform(fit)[features]
    X_tune = encoder.transform(tune)[features]
    y_fit = fit["is_delayed"]
    y_tune = tune["is_delayed"]

    if y_fit.nunique() != 2 or y_tune.nunique() != 2:
        raise ValueError("Both chronological periods must contain both classes")

    records = []
    best = None
    best_score = -np.inf

    if model_name == "logreg":
        grid = [
            {"C": C, "class_weight": weight, "max_iter": 2000}
            for C, weight in itertools.product(
                [0.01, 0.1, 1.0, 10.0],
                [None, "balanced"],
            )
        ]
    else:
        grid = [
            {
                "max_depth": depth,
                "learning_rate": lr,
                "scale_pos_weight": 1.0,
                "n_estimators": 1000,
                "early_stopping_rounds": 50,
            }
            for depth, lr in itertools.product([3, 5], [0.03, 0.10])
        ]

    for index, params in enumerate(grid, 1):
        if model_name == "logreg":
            model = build_logistic_pipeline(
                numeric, categorical, params
            )
            model.fit(X_fit, y_fit)
        else:
            model = build_xgboost(params)
            model.fit(
                X_fit, y_fit,
                eval_set=[(X_tune, y_tune)],
                verbose=False,
            )

        metrics = evaluate(model, X_tune, y_tune)
        record = {
            "model": model_name,
            "run": index,
            "parameters": params.copy(),
            **metrics,
        }

        if model_name == "xgboost":
            record["best_iteration"] = int(model.best_iteration)
            record["effective_trees"] = int(model.best_iteration) + 1

        records.append(record)
        print(
            f"{model_name} run {index}/{len(grid)}: "
            f"PR-AUC={metrics['pr_auc']:.5f} "
            f"ROC-AUC={metrics['roc_auc']:.5f}",
            flush=True,
        )

        if metrics["pr_auc"] > best_score + 1e-12:
            best_score = metrics["pr_auc"]
            best = record

    if not finalize:
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "protocol": "D-201",
            "model": model_name,
            "all_runs": records,
            "best_august_candidate": best,
        }
        (output_dir / f"{model_name}_experiments.json").write_text(
            json.dumps(report, indent=2)
        )
        return report

    # Final fit includes August. Its August score is NOT a validation score.
    final_data = pd.concat([fit, tune], ignore_index=True)
    final_encoder = make_encoder(cfg, final_data, numeric)
    X_final = final_encoder.transform(final_data)[features]
    y_final = final_data["is_delayed"]

    final_params = best["parameters"].copy()

    if model_name == "xgboost":
        final_params["n_estimators"] = best["effective_trees"]
        final_params["early_stopping_rounds"] = None
        final_model = build_xgboost(final_params)
    else:
        final_model = build_logistic_pipeline(
            numeric, categorical, final_params
        )

    final_model.fit(X_final, y_final)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_model(
        final_model,
        output_dir / f"{model_name}_final.joblib",
    )
    final_encoder.save(output_dir / f"{model_name}_encoder.json")

    report = {
        "protocol": "D-201",
        "model": model_name,
        "tuning_period": "2013-08-01 to 2013-08-31",
        "candidate_training_period": "2013-01-01 to 2013-07-31",
        "final_training_period": "2013-01-01 to 2013-08-31",
        "august_scores_are_tuning_only": True,
        "holdout_accessed": False,
        "all_runs": records,
        "best_august_candidate": best,
        "final_parameters": final_params,
    }
    (output_dir / f"{model_name}_experiments.json").write_text(
        json.dumps(report, indent=2)
    )
    print(f"Saved {model_name} final model and experiment logs")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/config.yaml"))
    parser.add_argument("--model", choices=["logreg", "xgboost", "all"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run actual D-201 training after plan approval",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    dev_path = (ROOT / cfg["paths"]["raw"]).resolve()
    expected = (ROOT / "data/interim/flights_dev.csv").resolve()
    if dev_path != expected:
        parser.error("Only Jan-Oct development dataset is allowed")
    if cfg["seed"] != 42:
        parser.error("D-201 requires seed 42")

    if args.dry_run:
        print("DRY RUN: PASS")
        print("D-201 CONFIGURATION: PASS")
        print("Sealed holdout: not accessed")
        print("Training: not executed")
        print("Model fitting: not executed")
        return

    if not args.execute:
        parser.error(
            "Actual training is disabled without --execute. Use --dry-run for validation."
        )

    output = ROOT / "models" / "d201"
    if output.exists() and any(output.glob(f"{args.model}_*")):
        parser.error("Existing model artifacts found; refusing overwrite")

    if args.model == "all":
        reports = [
            train(cfg, name, output, finalize=False)
            for name in ("logreg", "xgboost")
        ]

        def rank(report):
            best = report["best_august_candidate"]
            return (
                -best["pr_auc"],
                0 if report["model"] == "logreg" else 1,
                best["run"],
            )

        winner = min(reports, key=rank)
        selected = winner["model"]

        (output / "selection.json").write_text(
            json.dumps({
                "selected_model": selected,
                "selection_metric": "August PR-AUC",
                "candidate_results": reports,
            }, indent=2)
        )

        print("WINNER:", selected)
        print("BEST AUGUST PR-AUC:",
              winner["best_august_candidate"]["pr_auc"])

        # Final refit must use the already-selected parameters.
        # Do not rerun tuning here.
        final_params = winner["best_august_candidate"]["parameters"].copy()

        raw = load_dev_raw(cfg, ROOT)
        table = build_feature_table(raw, cfg)
        fit, tune = split_data(table)
        final_data = pd.concat([fit, tune], ignore_index=True)

        numeric = list(numeric_features(cfg))
        categorical = list(CATEGORICAL)
        features = numeric + categorical

        encoder = make_encoder(cfg, final_data, numeric)
        X_final = encoder.transform(final_data)[features]
        y_final = final_data["is_delayed"]

        if selected == "xgboost":
            final_params["n_estimators"] = (
                winner["best_august_candidate"]["effective_trees"]
            )
            final_params["early_stopping_rounds"] = None
            model = build_xgboost(final_params)
        else:
            model = build_logistic_pipeline(
                numeric, categorical, final_params
            )

        model.fit(X_final, y_final)

        save_model(model, output / "final_model.joblib")
        encoder.save(output / "final_encoder.json")

        (output / "final_training.json").write_text(
            json.dumps({
                "selected_model": selected,
                "parameters": final_params,
                "training_period": "2013-01-01 to 2013-08-31",
                "selection_period": "2013-08-01 to 2013-08-31",
                "preprocessing_refitted": True,
                "holdout_accessed": False,
            }, indent=2)
        )

        print("FINAL RETRAINING: COMPLETE")
        print("FINAL MODEL SAVED:", output / "final_model.joblib")
    else:
        train(cfg, args.model, output)


if __name__ == "__main__":
    main()
