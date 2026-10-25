"""Evaluate saved models on the test split.

Write metrics and figures to reports/results/ and reports/figures/.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.parse_args()
    # TODO: implement
    raise NotImplementedError


if __name__ == "__main__":
    main()
