# Data

| Folder | Contents | Committed? |
|---|---|---|
| `raw/` | Original immutable download (`flights.csv`) | **Yes** — `flights.csv` only |
| `interim/` | Cleaned, typed, target added | No |
| `processed/` | Feature matrices per split | No — rebuild with `python scripts/build_features.py` |
| `external/` | Supplementary data (e.g., BTS, airport metadata) | No |

## Getting the data

`data/raw/flights.csv` is committed, so a fresh clone already has it. It holds all of 2013,
including the **sealed Nov–Dec test holdout** (`docs/decision_log.md`, D-002). Development code
never reads it directly. Rebuild everything derived:

```bash
python scripts/seal_holdout.py     # writes data/interim/flights_dev.csv (Jan–Oct only) and prints both hashes
python scripts/build_features.py   # writes data/processed/{train,valid}.parquet, encoder.json, feature_spec.json
```

The dev file must have SHA-256
`8f206680eda4dc42098dbd4e9a6ec6f89f468afc41cbb913fd528fa4fc0018df` (281,373 rows).

Do **not** regenerate `flights.csv` with `scripts/export_dataset.py`. That script builds a different file
(ids start at 1, different source), which changes the hash and the row ids used throughout the pipeline.

## Provenance

- Source: Kaggle *Flights* dataset (2013 US flights).
- Extension: US Bureau of Transportation Statistics, Airline On-Time Performance.
- File: `data/raw/flights.csv` — 336,776 rows × 21 columns, 42,858,599 bytes.
- SHA-256: `8bdc6abbf07ac0cbb1036bc14a787cab3eea616be46c31899d2d8048963b08aa`
  (printed by notebooks 01–03 and stored in `data/processed/feature_spec.json`).
- TODO: Kaggle dataset URL and download date; confirm the dataset licence permits redistribution.
