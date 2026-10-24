"""raw -> congestion (full schedule) -> target -> features -> chronological split -> encode.

Reads the development file ``paths.raw`` (Jan-Oct) through ``load_dev_raw``, which refuses any
row from the sealed Nov-Dec holdout. Writes to ``paths.processed`` (git-ignored):
  train.parquet, valid.parquet                encoded feature matrices + id, date, target
                                              (no test split file while the holdout is sealed)
  encoder.json                                FeatureEncoder fitted on train only
  feature_spec.json                           feature groups, target, config snapshot, data hash
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flight_delay.config import load_config  # noqa: E402
from flight_delay.data.load import load_dev_raw  # noqa: E402
from flight_delay.data.schema import TARGET_COLUMN  # noqa: E402
from flight_delay.features.build import (  # noqa: E402
    build_feature_table,
    encode_splits,
    feature_groups,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_path = ROOT / cfg["paths"]["raw"]
    out_dir = ROOT / cfg["paths"]["processed"]
    out_dir.mkdir(parents=True, exist_ok=True)

    table = build_feature_table(load_dev_raw(cfg, ROOT), cfg)
    splits, encoder = encode_splits(table, cfg)
    for name, frame in splits.items():
        path = out_dir / f"{name}.parquet"
        if frame.empty:
            # The test split is empty during development (holdout sealed). Remove any file left by
            # an earlier full-year build so nothing downstream can read stale holdout rows.
            if path.exists():
                path.unlink()
                print(f"{name:5s}: empty (holdout sealed) | removed stale {path.relative_to(ROOT)}")
            else:
                print(f"{name:5s}: empty (holdout sealed)")
            continue
        frame.drop(columns="split").to_parquet(path, index=False)
        print(f"{name:5s}: {len(frame):7,d} rows | positive rate {frame[TARGET_COLUMN].mean():.4f}")
    encoder.save(out_dir / "encoder.json")

    spec = {
        "target": TARGET_COLUMN,
        "target_definition": f"arr_delay >= {cfg['target']['threshold_minutes']} min",
        "missing_label_policy": cfg["target"]["cancelled_policy"],
        "decision_time": cfg["decision_time"]["mode"],
        "feature_groups": feature_groups(cfg),
        "categorical": encoder.categorical,
        "numeric": encoder.numeric,
        "split": cfg["split"],
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    }
    (out_dir / "feature_spec.json").write_text(json.dumps(spec, indent=1))
    print(f"wrote {out_dir.relative_to(ROOT)}/: splits, encoder.json, feature_spec.json")


if __name__ == "__main__":
    main()
