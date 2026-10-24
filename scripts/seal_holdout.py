"""Create the development raw file (Jan-Oct) and leave the Nov-Dec holdout sealed.

The committed ``paths.raw_sealed`` file holds all of 2013. This script copies every row dated on
or before the validation end date (``split.valid[1]``) byte-for-byte into ``paths.raw`` and
prints SHA-256 hashes of both files for the decision log.

Only the ``month`` field of each line is read, to decide where the line goes. No delay, time or
label value is parsed, so no Nov-Dec information is looked at. The holdout is read again only by
the final, one-time test evaluation.
"""

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flight_delay.config import load_config  # noqa: E402

SEALED_SHA256 = "8bdc6abbf07ac0cbb1036bc14a787cab3eea616be46c31899d2d8048963b08aa"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    sealed = ROOT / cfg["paths"]["raw_sealed"]
    dev = ROOT / cfg["paths"]["raw"]
    valid_end = cfg["split"]["valid"][1]
    year, last_month, last_day = (int(x) for x in valid_end.split("-"))
    if (year, last_day) != (2013, 31) or last_month != 10:
        raise ValueError(f"expected validation to end on 2013-10-31, got {valid_end}")

    if sha256(sealed) != SEALED_SHA256:
        raise ValueError(f"{sealed} does not match the documented SHA-256 (data/README.md)")

    lines = sealed.read_bytes().splitlines(keepends=True)
    header, rows = lines[0], lines[1:]
    month_idx = header.decode().rstrip("\r\n").split(",").index("month")
    kept = [r for r in rows if int(r.split(b",")[month_idx]) <= last_month]

    dev.parent.mkdir(parents=True, exist_ok=True)
    dev.write_bytes(header + b"".join(kept))

    print(f"sealed  : {sealed.relative_to(ROOT)} | {len(rows):,} rows | sha256 {sha256(sealed)}")
    print(f"dev     : {dev.relative_to(ROOT)} | {len(kept):,} rows (months 1-{last_month}) "
          f"| sha256 {sha256(dev)}")
    print(f"holdout : {len(rows) - len(kept):,} rows not copied")


if __name__ == "__main__":
    main()
