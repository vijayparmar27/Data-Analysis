"""
Module 6: Advanced Polars, Time-Series & High Performance
=========================================================
Topics Covered:
- 6.1 Temporal Types & Parsing (Date, Datetime, Duration, str.to_datetime)
- 6.2 Temporal Operations via .dt Accessor (truncate, year, month, weekday)
- 6.3 Dynamic Time-Series Resampling (group_by_dynamic)
- 6.4 Rolling Temporal Aggregations (group_by_rolling)
- 6.5 Nested List Data Processing (pl.List, list.len, list.get, explode)
- 6.6 Hierarchical Struct Data Types (pl.Struct, unnest, struct.field)
- 6.7 User-Defined Functions (map_elements vs map_batches vs native expressions)
- 6.8 Threading, Parallelism & Memory Optimization (POLARS_MAX_THREADS)
- 6.9 Hands-on Challenge: Financial OHLCV Resampling & Rolling Volatility
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Defensive guard: prevent local folder name from shadowing installed library
_script_dir = Path(__file__).resolve().parent
_parent_dir = _script_dir.parent
for _p in ["", ".", str(_script_dir), str(_parent_dir)]:
    while _p in sys.path:
        sys.path.remove(_p)

import polars as pl

sys.path.append(str(_parent_dir))


def section(title: str):
    print(f"\n{'=' * 75}\n  {title}\n{'=' * 75}")


# ----------------------------------------------------------------------
# 6.1 Temporal Data Types & String Parsing
# ----------------------------------------------------------------------
section("6.1 Temporal Data Types & String Parsing (.str.to_datetime)")

# Simulating raw API logs with ISO-8601 string timestamps
raw_timestamps = pl.DataFrame(
    {
        "event_id": [1, 2, 3, 4, 5],
        "timestamp_str": [
            "2026-03-01 09:15:30",
            "2026-03-01 09:30:15",
            "2026-03-01 10:45:00",
            "2026-03-01 11:05:22",
            "2026-03-02 08:20:10",
        ],
        "latency_ms": [140, 220, 95, 310, 180],
    }
)

parsed_df = raw_timestamps.with_columns(
    pl.col("timestamp_str").str.to_datetime("%Y-%m-%d %H:%M:%S").alias("timestamp")
)
print("Parsed Datetime column:\n", parsed_df)
print(f"Timestamp dtype: {parsed_df.schema['timestamp']}")


# ----------------------------------------------------------------------
# 6.2 Temporal Operations via .dt Accessor
# ----------------------------------------------------------------------
section("6.2 Temporal Accessor (.dt) Manipulations")

dt_features = parsed_df.select(
    pl.col("timestamp"),
    pl.col("timestamp").dt.date().alias("date"),
    pl.col("timestamp").dt.year().alias("year"),
    pl.col("timestamp").dt.month().alias("month"),
    pl.col("timestamp").dt.weekday().alias("day_of_week"),
    pl.col("timestamp").dt.hour().alias("hour"),
    # Truncate to hour (equivalent to PostgreSQL date_trunc('hour', timestamp)):
    pl.col("timestamp").dt.truncate("1h").alias("hour_bucket"),
)
print("Extracted Temporal Components:\n", dt_features)


# ----------------------------------------------------------------------
# 6.3 Dynamic Time-Series Resampling (group_by_dynamic)
# ----------------------------------------------------------------------
section("6.3 Dynamic Resampling with group_by_dynamic()")

# Generate 24 hours of minute-by-minute transaction volume
base_time = datetime(2026, 3, 1, 0, 0, 0)
n_minutes = 60 * 12  # 12 hours
minute_series = [base_time + timedelta(minutes=i) for i in range(n_minutes)]

ts_data = pl.DataFrame(
    {
        "time": minute_series,
        "volume": [(i % 10) * 15 + 5 for i in range(n_minutes)],
        "revenue": [50.0 + (i % 25) * 10.0 for i in range(n_minutes)],
    }
).sort("time")

# group_by_dynamic resamples time series into tumbling or sliding windows (e.g. "1h", "15m", "1d"):
hourly_summary = (
    ts_data
    .group_by_dynamic("time", every="1h")
    .agg(
        pl.col("volume").sum().alias("total_volume"),
        pl.col("revenue").sum().alias("hourly_revenue"),
        pl.col("revenue").mean().round(2).alias("avg_order_value"),
    )
)
print("Resampled into 1-Hour Windows (first 5 hours):\n", hourly_summary.head(5))


# ----------------------------------------------------------------------
# 6.4 Rolling Temporal Windows (group_by_rolling)
# ----------------------------------------------------------------------
section("6.4 Rolling Temporal Aggregations (group_by_rolling)")

# Rolling 2-hour window computed for every single data point:
rolling_summary = (
    ts_data
    .rolling("time", period="2h")
    .agg(
        pl.col("revenue").mean().round(2).alias("rolling_2h_avg_revenue"),
        pl.col("volume").sum().alias("rolling_2h_volume"),
    )
)
print("Rolling 2-Hour Window Aggregation preview:\n", rolling_summary.slice(115, 5))


# ----------------------------------------------------------------------
# 6.5 Nested List Data Processing (pl.List)
# ----------------------------------------------------------------------
section("6.5 Nested List Data Types (pl.List & explode)")

# Modern JSON APIs frequently return arrays of tags, SKU line-items, or IDs
users_with_tags = pl.DataFrame(
    {
        "user_id": [101, 102, 103],
        "tags": [["tech", "developer", "rust"], ["management", "hr"], ["developer", "python", "data"]],
        "basket_amounts": [[45.0, 120.0, 30.0], [500.0], [12.5, 95.0]],
    }
)
print("DataFrame with List columns:\n", users_with_tags)

# List operations: length, get index, list contains, and exploding to rows
list_ops = users_with_tags.select(
    pl.col("user_id"),
    pl.col("tags").list.len().alias("tag_count"),
    pl.col("tags").list.get(0).alias("primary_tag"),
    pl.col("tags").list.contains("developer").alias("is_developer"),
    pl.col("basket_amounts").list.sum().alias("total_basket_value"),
)
print("\nList Manipulations:\n", list_ops)

# Explode: Unrolls array elements into separate individual rows
exploded_tags = users_with_tags.select("user_id", "tags").explode("tags")
print("\nExploded Tags (one row per list item):\n", exploded_tags)


# ----------------------------------------------------------------------
# 6.6 Hierarchical Struct Data Types (pl.Struct)
# ----------------------------------------------------------------------
section("6.6 Hierarchical Struct Types (pl.Struct & unnest)")

# Struct columns contain named sub-fields (like an embedded JSON object):
struct_df = pl.DataFrame(
    {
        "order_id": ["O-1", "O-2"],
        "customer": [
            {"name": "Alice Smith", "city": "Bengaluru", "zip": 560001},
            {"name": "Bob Jones", "city": "Mumbai", "zip": 400001},
        ],
    }
)
print("DataFrame with Struct column:\n", struct_df)
print(f"Customer Struct Schema: {struct_df.schema['customer']}")

# Extract specific sub-field:
print("\nExtracting single struct field (customer.city):")
print(struct_df.select(pl.col("order_id"), pl.col("customer").struct.field("city")))

# Unnest all struct fields into root columns:
print("\nUnnesting complete Struct into top-level columns:")
print(struct_df.unnest("customer"))


# ----------------------------------------------------------------------
# 6.7 User-Defined Functions (UDFs) & Best Practices
# ----------------------------------------------------------------------
section("6.7 User-Defined Functions: map_elements() vs Native Expressions")

# GOLDEN RULE: Avoid map_elements() whenever possible!
# map_elements() calls Python line-by-line, triggering GIL contention and disabling SIMD.

test_df = pl.DataFrame({"score": [12.0, 45.0, 78.0, 95.0, 33.0]})

# Anti-pattern (Slow Python function):
def custom_classifier(val: float) -> str:
    return "PASS" if val >= 50.0 else "FAIL"

# Fast, idiomatic Polars pattern (Native Rust expression):
idiomatic_classification = test_df.with_columns(
    pl.when(pl.col("score") >= 50.0)
    .then(pl.lit("PASS"))
    .otherwise(pl.lit("FAIL"))
    .alias("result_native")
)
print("Idiomatic Expression Result (Blazing fast in Rust):\n", idiomatic_classification)


# ----------------------------------------------------------------------
# 6.8 Hands-on Financial Resampling Challenge
# ----------------------------------------------------------------------
section("6.8 Hands-on Challenge: Financial OHLCV 5-Minute Candle Resampling")

# Generate mock 1-minute tick trades
trades = pl.DataFrame(
    {
        "timestamp": [base_time + timedelta(minutes=i) for i in range(30)],
        "trade_price": [100.0 + (i % 7) * 2.5 - (i % 3) * 1.2 for i in range(30)],
        "trade_volume": [10 + (i % 5) * 5 for i in range(30)],
    }
).sort("timestamp")

# Resample 1-minute ticks into standard 5-minute OHLCV candles
# (Open, High, Low, Close, Volume)
ohlcv_candles = (
    trades
    .group_by_dynamic("timestamp", every="5m")
    .agg(
        pl.col("trade_price").first().alias("open"),
        pl.col("trade_price").max().alias("high"),
        pl.col("trade_price").min().alias("low"),
        pl.col("trade_price").last().alias("close"),
        pl.col("trade_volume").sum().alias("volume"),
    )
)

print("Generated 5-Minute Financial OHLCV Candles:\n", ohlcv_candles)

print("\n Module 6 Completed Successfully!")
