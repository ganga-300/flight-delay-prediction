"""XGBoost classifier."""

from xgboost import XGBClassifier


def build_xgboost(
    params: dict,
    scale_pos_weight: float | None = None,
) -> XGBClassifier:
    """Build an XGBoost classifier.

    Validation-based early stopping is configured during model fitting.
    """

    model_params = {
        "n_estimators": int(params.get("n_estimators", 500)),
        "learning_rate": float(params.get("learning_rate", 0.05)),
        "max_depth": int(params.get("max_depth", 6)),
        "subsample": float(params.get("subsample", 0.8)),
        "colsample_bytree": float(params.get("colsample_bytree", 0.8)),
        "eval_metric": params.get("eval_metric", "aucpr"),
        "early_stopping_rounds": int(params.get("early_stopping_rounds", 30)),
        "random_state": int(params.get("random_state", 42)),
        "n_jobs": int(params.get("n_jobs", 2)),
        "tree_method": "hist",
    }

    if scale_pos_weight is not None:
        model_params["scale_pos_weight"] = float(scale_pos_weight)

    return XGBClassifier(**model_params)
