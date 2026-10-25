"""XGBoost binary classifier builder."""

from xgboost import XGBClassifier


def build_xgboost(
    params: dict,
    scale_pos_weight: float | None = None,
) -> XGBClassifier:
    """Create an unfitted XGBoost classifier."""
    config = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "enable_categorical": True,
        "random_state": 42,
        "n_jobs": 4,
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "max_depth": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": 50,
    }

    config.update(params)

    if scale_pos_weight is not None:
        config["scale_pos_weight"] = scale_pos_weight

    return XGBClassifier(**config)
