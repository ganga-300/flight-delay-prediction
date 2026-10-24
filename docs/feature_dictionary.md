# Feature Dictionary

Generated from `features/build.py::feature_groups` for the current `configs/config.yaml`.
Every feature for a flight on date D uses only the published schedule and outcomes known
before D (see `docs/leakage_policy.md`). Categorical features are collapsed by
`FeatureEncoder`, fitted on train only: levels with fewer than 50 training rows, and unseen levels, become `__OTHER__`.

The set of 59 features is fixed and pinned by `test_feature_set_is_pinned`. Why each group is
included, and why the windows (1/7/28 days), prior strength (20), category minimum (50) and
departure-bucket edges have their values, is recorded in `docs/decision_log.md` (D-101 to D-105),
using training-period evidence only.

## Calendar / time (10)

| Feature | Description |
|---|---|
| `sched_dep_hour_frac` | Scheduled departure time in hours (e.g. 17.5 = 17:30) |
| `dow` | Day of week, 0 = Monday |
| `is_weekend` | Saturday or Sunday |
| `is_holiday` | US federal holiday (rule-based calendar) |
| `days_from_holiday` | Signed days from nearest US federal holiday, clipped to ±14 |
| `abs_days_to_holiday` | Absolute distance to nearest US federal holiday, clipped to 14 |
| `dep_hour_sin` | sin(2π · hour / 24) |
| `dep_hour_cos` | cos(2π · hour / 24) |
| `dow_sin` | sin(2π · dow / 7) |
| `dow_cos` | cos(2π · dow / 7) |

## Schedule / route (numeric) (6)

| Feature | Description |
|---|---|
| `sched_dep_min` | Scheduled departure, minutes after local midnight |
| `sched_arr_min` | Scheduled arrival, minutes after midnight (destination local time) |
| `sched_block_min` | Scheduled gate-to-gate minutes on a common UTC basis (airport IANA time zones, DST-aware, arrival date resolved); schedule + static reference data only |
| `overnight_arrival` | Scheduled arrival clock time is earlier than departure (crosses midnight) |
| `distance` | Route distance, miles |
| `log_distance` | ln(distance) |

## Categorical (flight / route and interactions) (10)

| Feature | Description |
|---|---|
| `carrier` | Airline code |
| `origin` | Origin airport (EWR/JFK/LGA) |
| `dest` | Destination airport |
| `route` | origin-dest |
| `dep_bucket` | Scheduled departure time-of-day bucket |
| `carrier_origin` | carrier × origin |
| `carrier_dest` | carrier × destination |
| `carrier_route` | carrier × route |
| `origin_dep_bucket` | origin × departure bucket |
| `carrier_dep_bucket` | carrier × departure bucket |

## Congestion (4)

| Feature | Description |
|---|---|
| `sched_dep_origin_hour` | Scheduled departures from the same origin, date and scheduled hour (full raw schedule) |
| `sched_dep_origin_day` | Scheduled departures from the same origin and date (full raw schedule) |
| `sched_dep_carrier_origin_day` | Scheduled departures of the same carrier from the same origin and date |
| `sched_dep_origin_hour_share` | sched_dep_origin_hour / sched_dep_origin_day |

## Historical (strictly earlier dates) (29)

| Feature | Description |
|---|---|
| `hist_global_all_rate` | Delay rate of all flights with outcome known before D (smoothing prior p0) |
| `hist_carrier_1d_rate` | Delay rate of the same carrier over dates D−1 … D−1, shrunk toward p0 with m = 20 |
| `hist_carrier_1d_n` | Number of carrier flights with outcome known on dates D−1 … D−1 |
| `hist_carrier_7d_rate` | Delay rate of the same carrier over dates D−7 … D−1, shrunk toward p0 with m = 20 |
| `hist_carrier_7d_n` | Number of carrier flights with outcome known on dates D−7 … D−1 |
| `hist_carrier_28d_rate` | Delay rate of the same carrier over dates D−28 … D−1, shrunk toward p0 with m = 20 |
| `hist_carrier_28d_n` | Number of carrier flights with outcome known on dates D−28 … D−1 |
| `hist_carrier_all_rate` | Delay rate of the same carrier over all dates before D, shrunk toward p0 with m = 20 |
| `hist_route_1d_rate` | Delay rate of the same route over dates D−1 … D−1, shrunk toward p0 with m = 20 |
| `hist_route_1d_n` | Number of route flights with outcome known on dates D−1 … D−1 |
| `hist_route_7d_rate` | Delay rate of the same route over dates D−7 … D−1, shrunk toward p0 with m = 20 |
| `hist_route_7d_n` | Number of route flights with outcome known on dates D−7 … D−1 |
| `hist_route_28d_rate` | Delay rate of the same route over dates D−28 … D−1, shrunk toward p0 with m = 20 |
| `hist_route_28d_n` | Number of route flights with outcome known on dates D−28 … D−1 |
| `hist_route_all_rate` | Delay rate of the same route over all dates before D, shrunk toward p0 with m = 20 |
| `hist_origin_1d_rate` | Delay rate of the same origin over dates D−1 … D−1, shrunk toward p0 with m = 20 |
| `hist_origin_1d_n` | Number of origin flights with outcome known on dates D−1 … D−1 |
| `hist_origin_7d_rate` | Delay rate of the same origin over dates D−7 … D−1, shrunk toward p0 with m = 20 |
| `hist_origin_7d_n` | Number of origin flights with outcome known on dates D−7 … D−1 |
| `hist_origin_28d_rate` | Delay rate of the same origin over dates D−28 … D−1, shrunk toward p0 with m = 20 |
| `hist_origin_28d_n` | Number of origin flights with outcome known on dates D−28 … D−1 |
| `hist_origin_all_rate` | Delay rate of the same origin over all dates before D, shrunk toward p0 with m = 20 |
| `hist_dest_1d_rate` | Delay rate of the same dest over dates D−1 … D−1, shrunk toward p0 with m = 20 |
| `hist_dest_1d_n` | Number of dest flights with outcome known on dates D−1 … D−1 |
| `hist_dest_7d_rate` | Delay rate of the same dest over dates D−7 … D−1, shrunk toward p0 with m = 20 |
| `hist_dest_7d_n` | Number of dest flights with outcome known on dates D−7 … D−1 |
| `hist_dest_28d_rate` | Delay rate of the same dest over dates D−28 … D−1, shrunk toward p0 with m = 20 |
| `hist_dest_28d_n` | Number of dest flights with outcome known on dates D−28 … D−1 |
| `hist_dest_all_rate` | Delay rate of the same dest over all dates before D, shrunk toward p0 with m = 20 |

## Not used as features

| Column | Reason |
|---|---|
| `dep_time`, `dep_delay`, `arr_time`, `arr_delay`, `air_time` | Post-departure (used only for the target and for timing past outcomes) |
| `tailnum` | Aircraft swaps; missing only for cancelled flights |
| `id`, `flight`, `name`, `year`, `time_hour` | Identifier, very high cardinality, redundant, or constant |
| `day_of_year`, `hist_*_all_n` | Act as time indices: they grow steadily with the date, so every later period lies outside the training range by construction |
| `month`, `month_sin/cos`, `doy_sin/cos` | Training covers Jan–Aug only, so every later month is unseen (100% of validation rows out of range for `month`, `month_sin`, `doy_sin`); see notebook 03 audit A3 |
| `sched_clock_duration_min` | Local clock difference mixes block time with the destination time-zone offset; replaced by `sched_block_min` |
