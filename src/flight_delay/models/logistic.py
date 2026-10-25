"""Logistic Regression baseline."""

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_logistic_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    params: dict,
) -> Pipeline:
    """Create an unfitted Logistic Regression pipeline."""
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_cols),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    classifier = LogisticRegression(
        penalty="l2",
        C=params.get("C", 1.0),
        class_weight=params.get("class_weight"),
        max_iter=params.get("max_iter", 2000),
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", classifier),
        ]
    )
