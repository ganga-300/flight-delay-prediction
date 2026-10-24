"""Single source of truth for columns forbidden in model feature matrices."""

POST_DEPARTURE_COLUMNS: frozenset[str] = frozenset(
    {"dep_time", "dep_delay", "arr_time", "arr_delay", "air_time"}
)

NON_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        "id",
        "date",
        "split",
        "is_delayed",
        "flight_status",
        "tailnum",
    }
)

FORBIDDEN_FEATURE_COLUMNS = POST_DEPARTURE_COLUMNS | NON_FEATURE_COLUMNS


def assert_no_leakage(columns) -> None:
    """Raise if a forbidden column is included in the model feature matrix."""
    columns = list(columns)

    duplicates = sorted({c for c in columns if columns.count(c) > 1})
    if duplicates:
        raise ValueError(f"Duplicate feature columns: {duplicates}")

    leaked = FORBIDDEN_FEATURE_COLUMNS.intersection(columns)
    if leaked:
        raise ValueError(
            f"Forbidden columns in features: {sorted(leaked)}"
        )
