# Data

| Folder | Contents | Committed? |
|---|---|---|
| `raw/` | Original immutable download (`flights.csv`) | **Yes** — `flights.csv` only |
| `interim/` | Cleaned, typed, target added | No |
| `processed/` | Feature matrices per split | No — rebuild with `python scripts/build_features.py` |
| `external/` | Supplementary data (e.g., BTS, airport metadata) | No |

## Getting the data

`data/raw/flights.csv` is committed, so a fresh clone already has it. Rebuild everything derived:

```bash
python scripts/build_features.py   # writes data/processed/{train,valid,test}.parquet, encoder.json, feature_spec.json
```

Do **not** regenerate `flights.csv` with `scripts/export_dataset.py`. That script builds a different file
(ids start at 1, different source), which changes the hash and the row ids used throughout the pipeline.

## Provenance

- Source: Kaggle *Flights* dataset (2013 US flights).
- Extension: US Bureau of Transportation Statistics, Airline On-Time Performance.
- File: `data/raw/flights.csv` — 336,776 rows × 21 columns, 42,858,599 bytes.
- SHA-256: `8bdc6abbf07ac0cbb1036bc14a787cab3eea616be46c31899d2d8048963b08aa`
  (printed by notebooks 01–03 and stored in `data/processed/feature_spec.json`).
- TODO: Kaggle dataset URL and download date; confirm the dataset licence permits redistribution.
