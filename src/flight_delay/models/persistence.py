"""Save and load trained models."""

from pathlib import Path

import joblib


def save_model(model, path: str) -> None:
    """Save a trained model to disk."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)


def load_model(path: str):
    """Load a previously saved model."""
    source = Path(path)

    if not source.is_file():
        raise FileNotFoundError(f"Saved model not found: {source}")

    return joblib.load(source)
