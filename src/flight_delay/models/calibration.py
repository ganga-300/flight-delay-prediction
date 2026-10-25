"""Post-hoc calibration of an already-fitted classifier."""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator


def calibrate(model, X_valid, y_valid, method: str = "sigmoid"):
    """Calibrate a fitted model without refitting the underlying estimator."""
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError("method must be 'sigmoid' or 'isotonic'")

    calibrated = CalibratedClassifierCV(
        estimator=FrozenEstimator(model),
        method=method,
    )
    calibrated.fit(X_valid, y_valid)
    return calibrated
