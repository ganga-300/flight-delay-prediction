"""Post-hoc probability calibration using the validation split only."""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class CalibratedModel:
    """Wrap a fitted classifier with a post-hoc probability calibrator."""

    def __init__(self, model, calibrator, method: str):
        self.model = model
        self.calibrator = calibrator
        self.method = method
        self.classes_ = np.array([0, 1])

    def _positive_probability(self, X):
        probabilities = np.asarray(self.model.predict_proba(X))
        classes = np.asarray(self.model.classes_)
        positive_index = np.flatnonzero(classes == 1)

        if len(positive_index) != 1:
            raise ValueError("The fitted model must contain class 1.")

        return probabilities[:, positive_index[0]]

    def predict_proba(self, X):
        raw_probability = self._positive_probability(X)

        if self.method == "sigmoid":
            calibrated = self.calibrator.predict_proba(
                raw_probability.reshape(-1, 1)
            )[:, 1]
        else:
            calibrated = self.calibrator.predict(raw_probability)

        calibrated = np.clip(calibrated, 0.0, 1.0)

        return np.column_stack((1.0 - calibrated, calibrated))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def calibrate(model, X_valid, y_valid, method: str = "isotonic"):
    """Fit a sigmoid or isotonic calibrator on validation predictions."""

    if method not in {"sigmoid", "isotonic"}:
        raise ValueError("method must be 'sigmoid' or 'isotonic'.")

    y_valid = np.asarray(y_valid)
    if len(np.unique(y_valid)) != 2:
        raise ValueError("Calibration requires both target classes.")

    raw_probability = np.asarray(model.predict_proba(X_valid))[:, 1]

    if method == "sigmoid":
        calibrator = LogisticRegression()
        calibrator.fit(raw_probability.reshape(-1, 1), y_valid)
    else:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_probability, y_valid)

    return CalibratedModel(model, calibrator, method)
