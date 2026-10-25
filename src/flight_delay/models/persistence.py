"""Save and load trained model artifacts."""

from pathlib import Path

import joblib


def save_model(model, path: str | Path) -> None:
    """Save a model to disk."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)


def load_model(path: str | Path):
    """Load a previously saved model from disk."""
    return joblib.load(path)
