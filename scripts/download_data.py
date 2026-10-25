"""Download the Kaggle Flights dataset into data/raw/."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.parse_args()
    # TODO: implement
    raise NotImplementedError


if __name__ == "__main__":
    main()
