"""Flight-delay training CLI; actual training requires D-201 approval."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flight_delay.config import load_config  # noqa: E402
from flight_delay.models.development import PERIODS  # noqa: E402


def validate_plan(cfg: dict) -> None:
    """Validate the proposed development protocol without reading flight data."""
    expected = {
        "fit": ("2013-01-01", "2013-07-31"),
        "tune": ("2013-08-01", "2013-08-31"),
        "calibrate": ("2013-09-01", "2013-09-30"),
        "threshold": ("2013-10-01", "2013-10-31"),
    }

    if PERIODS != expected:
        raise ValueError("Development periods differ from D-201")

    if cfg["seed"] != 42:
        raise ValueError("Random seed must be 42")

    if cfg["decision_time"]["mode"] != "d_minus_1":
        raise ValueError("Unexpected decision-time policy")

    if cfg["evaluation"]["operational_precision"] != 0.60:
        raise ValueError("Operational precision must be 0.60")

    raw = (ROOT / cfg["paths"]["raw"]).resolve()
    permitted = (ROOT / "data/interim/flights_dev.csv").resolve()

    if raw != permitted:
        raise ValueError("Only the approved development dataset is permitted")

    if not raw.is_file():
        raise FileNotFoundError(f"Development data missing: {raw}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "config.yaml"),
    )
    parser.add_argument(
        "--model",
        choices=["logreg", "xgboost"],
        required=True,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without training",
    )

    args = parser.parse_args()

    if not args.dry_run:
        parser.error(
            "Actual training is disabled until D-201 receives team approval. "
            "Use --dry-run."
        )

    cfg = load_config(args.config)
    validate_plan(cfg)

    print(f"Model: {args.model}")
    print("Protocol: D-201 (proposed)")
    print("Development periods:")

    for name, (start, end) in PERIODS.items():
        print(f"  {name:10s}: {start} to {end}")

    print("Sealed holdout: not accessed")
    print("Model fitting: not executed")
    print("DRY RUN: PASS")


if __name__ == "__main__":
    main()
