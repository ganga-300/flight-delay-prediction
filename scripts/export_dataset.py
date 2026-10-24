from pathlib import Path
import pandas as pd
import nycflights13

out = Path("data/raw")
out.mkdir(parents=True, exist_ok=True)

def get_table(name):
    table = getattr(nycflights13, name)
    if isinstance(table, pd.DataFrame):
        return table.copy()
    if isinstance(table, dict):
        return pd.DataFrame(table)
    raise TypeError(f"Unexpected table type for {name}: {type(table)}")

flights = get_table("flights")
airlines = get_table("airlines")

print("Flights shape:", flights.shape)
print("Flights columns:", flights.columns.tolist())
print("Airlines shape:", airlines.shape)

# Add airline names and a stable row ID.
flights = flights.merge(airlines, on="carrier", how="left")
flights.insert(0, "id", range(1, len(flights) + 1))

# Match the column names expected by this project's loader.
flights = flights[
    [
        "id", "year", "month", "day", "dep_time",
        "sched_dep_time", "dep_delay", "arr_time",
        "sched_arr_time", "arr_delay", "carrier", "flight",
        "tailnum", "origin", "dest", "air_time", "distance",
        "hour", "minute", "time_hour", "name",
    ]
]

destination = out / "flights.csv"
flights.to_csv(destination, index=False)

print(f"\nSaved: {destination}")
print("Final shape:", flights.shape)
print("Missing airline names:", flights["name"].isna().sum())
print("File size (MB):", round(destination.stat().st_size / (1024 * 1024), 2))
print(flights.head(3).to_string(index=False))
