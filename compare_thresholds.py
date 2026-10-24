import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from flight_delay.config import load_config
from flight_delay.models.persistence import load_model

config = load_config("configs/config.yaml")
data_dir = Path(config["paths"]["processed"])
model_dir = Path(config["paths"]["models"])

valid = pd.read_parquet(data_dir / "valid.parquet")
spec = json.loads((data_dir / "feature_spec.json").read_text())

features = spec["numeric"] + spec["categorical"]
target = spec.get("target", "is_delayed")

X = valid[features]
y = valid[target].astype(int).to_numpy()

for name in ["logreg", "xgboost"]:
    model = load_model(str(model_dir / f"{name}.joblib"))
    probabilities = model.predict_proba(X)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    precision = precision[:-1]
    recall = recall[:-1]

    print(f"\n{'=' * 45}")
    print(name.upper())

    for target_precision in [0.50, 0.55, 0.60, 0.65, 0.70]:
        eligible = np.flatnonzero(precision >= target_precision)

        if eligible.size == 0:
            print(f"Target {target_precision:.0%}: no eligible threshold")
            continue

        best_recall = recall[eligible].max()
        candidates = eligible[recall[eligible] == best_recall]
        idx = candidates[np.argmax(thresholds[candidates])]

        print(
            f"Target={target_precision:.0%} | "
            f"Precision={precision[idx]:.3f} | "
            f"Recall={recall[idx]:.4f} | "
            f"Threshold={thresholds[idx]:.4f} | "
            f"Predicted positives={(probabilities >= thresholds[idx]).sum()}"
        )
