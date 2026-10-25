"""D-201 candidate tuning and selection; no calibration or threshold."""

from flight_delay.models.selection import select_best_model
from flight_delay.models.tuning import tune_logistic, tune_xgboost


def run_modeling_workflow(
    splits,
    numeric_cols,
    categorical_cols,
    target_column="is_delayed",
):
    """Fit on January-July, tune and select using August only."""
    if set(splits) != {"fit", "tune"}:
        raise ValueError("D-201 requires fit and tune periods only")

    numeric_cols = list(numeric_cols)
    categorical_cols = list(categorical_cols)
    features = numeric_cols + categorical_cols

    if (
        len(numeric_cols) != 49
        or len(categorical_cols) != 10
        or len(set(features)) != 59
    ):
        raise ValueError("D-201 requires 49 numeric and 10 categorical features")

    prepared = {}
    for period in ("fit", "tune"):
        frame = splits[period]

        if frame.empty:
            raise ValueError(f"Empty development period: {period}")
        if target_column not in frame:
            raise ValueError(f"Missing target in {period}")
        if not set(features).issubset(frame.columns):
            raise ValueError(f"Missing frozen features in {period}")

        prepared[period] = (frame[features], frame[target_column])

    X_fit, y_fit = prepared["fit"]
    X_tune, y_tune = prepared["tune"]

    logistic = tune_logistic(
        X_fit, y_fit, X_tune, y_tune,
        numeric_cols, categorical_cols,
    )
    xgboost = tune_xgboost(
        X_fit, y_fit, X_tune, y_tune,
        numeric_cols, categorical_cols,
    )
    selection = select_best_model(logistic, xgboost)

    return {
        "logistic": logistic,
        "xgboost": xgboost,
        "selection": selection,
    }
