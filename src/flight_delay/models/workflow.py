"""D-201 modeling workflow using four separate development periods."""

from flight_delay.models.operating_point import calibrate_and_select_threshold
from flight_delay.models.selection import select_best_model
from flight_delay.models.tuning import tune_logistic, tune_xgboost


def run_modeling_workflow(
    splits,
    numeric_cols,
    categorical_cols,
    target_column="is_delayed",
    min_precision=0.60,
):
    """Train, tune, select, calibrate and choose an operating threshold."""

    expected_periods = {"fit", "tune", "calibrate", "threshold"}

    if set(splits) != expected_periods:
        raise ValueError("Four D-201 development periods are required")

    features = list(numeric_cols) + list(categorical_cols)

    if len(features) != 59 or len(set(features)) != 59:
        raise ValueError("Exactly 59 unique features are required")

    prepared = {}

    for period in ("fit", "tune", "calibrate", "threshold"):
        frame = splits[period]

        if frame.empty:
            raise ValueError(f"Empty development period: {period}")

        if target_column not in frame:
            raise ValueError(f"Missing target in {period}")

        if not set(features).issubset(frame.columns):
            raise ValueError(f"Missing frozen features in {period}")

        prepared[period] = (
            frame[features],
            frame[target_column],
        )

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

    X_cal, y_cal = prepared["calibrate"]
    X_threshold, y_threshold = prepared["threshold"]

    operating = calibrate_and_select_threshold(
        fitted_model=selection["model"],
        X_calibrate=X_cal,
        y_calibrate=y_cal,
        X_threshold=X_threshold,
        y_threshold=y_threshold,
        feature_columns=features,
        min_precision=min_precision,
    )

    return {
        "logistic": logistic,
        "xgboost": xgboost,
        "selection": selection,
        "calibrated_model": operating["calibrated_model"],
        "operating_point": operating["operating_point"],
    }
