"""
Module 1: Polars Fundamentals & Architecture
===========================================
Topics Covered:
- 1.1 Why Polars? (Rust core, Apache Arrow columnar memory, parallel multithreading, zero-copy)
- 1.2 Polars vs Pandas vs NumPy (Architecture, GIL bypass, memory layout)
- 1.3 Environment & Namespace Conventions (`import polars as pl`)
- 1.4 Core Data Structures (Series: 1D Arrow chunked array vs DataFrame: 2D columnar table)
- 1.5 DataFrame Creation (from dict of lists, list of dicts, tuples, NumPy arrays)
- 1.6 DataFrame Inspection (.shape, .columns, .schema, .dtypes, .estimated_size())
- 1.7 Polars Data Types (Int, Float, String, Boolean, Date, Datetime, Duration, Categorical, Enum)
- 1.8 Basic Exploration (.head(), .tail(), .sample(), .glimpse(), pl.Config table formatting)
- 1.9 Interoperability & Zero-Copy Exchange (to_pandas, to_numpy, Arrow C Stream)
- 1.10 Practice Exercise: Creating, Inspecting, and Type Casting an Analytics Dataset
"""

import sys
from pathlib import Path

# Defensive guard: prevent local folder name from shadowing installed library
_script_dir = Path(__file__).resolve().parent
_parent_dir = _script_dir.parent
for _p in ["", ".", str(_script_dir), str(_parent_dir)]:
    while _p in sys.path:
        sys.path.remove(_p)

import numpy as np
import polars as pl
import pyarrow as pa

sys.path.append(str(_parent_dir))


def section(title: str):
    print(f"\n{'=' * 75}\n  {title}\n{'=' * 75}")


# ----------------------------------------------------------------------
# 1.1 Why Polars? (Rust Core, Apache Arrow & Multithreading)
# ----------------------------------------------------------------------
section("1.1 Why Polars? (Architecture & Performance Paradigm)")

# Why Polars for Backend and Full-Stack Developers:
# 1. Rust Core: Written from scratch in Rust, completely memory-safe and compiled to native machine code.
# 2. Apache Arrow Memory: Standardized columnar memory representation with contiguous array buffers,
#    enabling vectorized SIMD CPU instructions and zero-copy data exchange.
# 3. Rayon Multithreading: Unlike Pandas (which is mostly single-threaded and bound to Python's GIL),
#    Polars runs native Rust threads via Rayon, saturating all CPU cores automatically.
# 4. Expressions & Optimization: Queries are expressed as trees of operations and executed in parallel
#    without materializing expensive intermediate DataFrame copies.

print(f"Polars Version: {pl.__version__}")
print(f"Polars Thread Pool Size: {pl.thread_pool_size()} threads")


# ----------------------------------------------------------------------
# 1.2 Core Data Structures: Series & DataFrame
# ----------------------------------------------------------------------
section("1.2 Core Data Structures: Series (1D) vs DataFrame (2D)")

# A Series is a 1D sequence of typed values backed by an Apache Arrow ChunkedArray.
# Note: Unlike Pandas, Polars does NOT have an index! Row positions are implicit 0-based offsets.
prices_series = pl.Series("unit_price", [450.0, 720.5, 310.0, 600.0, 890.0], dtype=pl.Float64)
print("Polars Series:\n", prices_series)
print(f"Series Name: {prices_series.name}, Dtype: {prices_series.dtype}, Length: {len(prices_series)}")
print(f"Series Sum: {prices_series.sum():.2f}, Mean: {prices_series.mean():.2f}")

# Missing data in Polars is represented by `null` (not NaN or None):
status_series = pl.Series("order_status", ["completed", "pending", None, "completed", "failed"])
print("\nSeries with Null values (Apache Arrow null bitmap):\n", status_series)
print(f"Null count: {status_series.null_count()}")


# ----------------------------------------------------------------------
# 1.3 Creating DataFrames from Various Sources
# ----------------------------------------------------------------------
section("1.3 Creating DataFrames from Dictionaries, Records, and NumPy")

# Method A: Dict of lists (Most common for mock data, tests & batch jobs)
inventory_data = {
    "sku": ["SKU-001", "SKU-002", "SKU-003", "SKU-004", "SKU-005"],
    "product_name": ["Pomfret", "Tiger Prawns", "Kingfish", "Bombay Duck", "Crab"],
    "category": ["Fish", "Shellfish", "Fish", "Fish", "Shellfish"],
    "unit_price": [750.0, 950.0, 800.0, 350.0, 600.0],
    "stock_qty": [25, 40, 15, 60, 30],
    "is_active": [True, True, True, False, True],
}

df_inventory = pl.DataFrame(inventory_data)
print("DataFrame from Dict of Lists:\n", df_inventory)

# Method B: List of Dictionaries (Matches JSON REST API payloads / MongoDB documents)
api_payload = [
    {"user_id": 101, "email": "alice@example.com", "tier": "gold", "lifetime_value": 4500.50},
    {"user_id": 102, "email": "bob@example.com", "tier": "silver", "lifetime_value": 1250.00},
    {"user_id": 103, "email": "charlie@example.com", "tier": "bronze", "lifetime_value": 300.00},
]
df_users = pl.DataFrame(api_payload)
print("\nDataFrame from List of Dicts (API Payload):\n", df_users)

# Method C: From 2D NumPy Array
np_matrix = np.array([[1.0, 2.5, 3.2], [4.1, 5.0, 6.3], [7.2, 8.4, 9.9]])
df_from_np = pl.DataFrame(np_matrix, schema=["sensor_1", "sensor_2", "sensor_3"])
print("\nDataFrame from 2D NumPy array:\n", df_from_np)


# ----------------------------------------------------------------------
# 1.4 Inspecting DataFrame Attributes & Schema
# ----------------------------------------------------------------------
section("1.4 DataFrame Inspection: Shape, Schema & Memory Size")

print(f"Shape (rows, cols): {df_inventory.shape}")
print(f"Column Names:       {df_inventory.columns}")
print(f"Schema (name -> dtype):\n{df_inventory.schema}")
print(f"Data Types list:    {df_inventory.dtypes}")
print(f"Estimated RAM footprint: {df_inventory.estimated_size()} bytes")


# ----------------------------------------------------------------------
# 1.5 Polars Data Types & Explicit Schema Definition
# ----------------------------------------------------------------------
section("1.5 Polars Data Types (pl.DataType) & Schema Overrides")

# Enforcing strict schemas at creation time avoids ambiguous type inference:
strict_schema = {
    "order_id": pl.Int64,
    "customer_id": pl.Int32,
    "amount": pl.Float32,
    "tax_rate": pl.Float32,
    "is_settled": pl.Boolean,
}

orders_df = pl.DataFrame(
    [
        (1001, 501, 120.50, 0.18, True),
        (1002, 502, 340.00, 0.18, False),
        (1003, 503, 89.99, 0.05, True),
    ],
    schema=strict_schema,
    orient="row",
)
print("DataFrame with explicit strict schema:\n", orders_df)
print("Schema details:\n", orders_df.schema)


# ----------------------------------------------------------------------
# 1.6 Exploration & Pretty Printing (pl.Config)
# ----------------------------------------------------------------------
section("1.6 DataFrame Exploration (.glimpse, .head, pl.Config)")

print("First 2 rows (.head(2)):")
print(df_inventory.head(2))

print("\nLast 2 rows (.tail(2)):")
print(df_inventory.tail(2))

print("\nRandom 2 sample rows (.sample(2)):")
print(df_inventory.sample(2, seed=42))

print("\nGlimpse (Compact horizontal schema and data preview):")
df_inventory.glimpse()

# Polars Config controls table formatting in terminals/logs:
with pl.Config(tbl_rows=4, tbl_cols=4, tbl_formatting="ASCII_FULL_CONDENSED"):
    print("\nFormatted with pl.Config ASCII condensed:")
    print(df_inventory)


# ----------------------------------------------------------------------
# 1.7 Interoperability with NumPy, Pandas & PyArrow (Zero-Copy)
# ----------------------------------------------------------------------
section("1.7 Interoperability: NumPy, Pandas, PyArrow")

# Polars <-> NumPy:
np_prices = df_inventory["unit_price"].to_numpy()
print("Converted to NumPy 1D array:", np_prices, type(np_prices))

# Polars <-> Pandas:
pandas_df = df_inventory.to_pandas()
print("\nConverted to Pandas DataFrame:\n", pandas_df.head(2))

# Polars <-> PyArrow (Zero-copy Arrow Table):
arrow_table = df_inventory.to_arrow()
print("\nConverted to PyArrow Table:\n", arrow_table)

# Re-creating Polars DataFrame from PyArrow (Zero-copy):
restored_df = pl.from_arrow(arrow_table)
print("Restored from Arrow zero-copy:", restored_df.shape)


# ----------------------------------------------------------------------
# 1.8 Module 1 Practice Exercise
# ----------------------------------------------------------------------
section("1.8 Practice Exercise: Build & Audit a Transaction Dataset")

raw_transactions = [
    {"tx_id": "TX-1", "user_id": 901, "amount": "1450.50", "currency": "INR", "status": "SUCCESS"},
    {"tx_id": "TX-2", "user_id": 902, "amount": "3200.00", "currency": "INR", "status": "PENDING"},
    {"tx_id": "TX-3", "user_id": 903, "amount": "780.25",  "currency": "USD", "status": "SUCCESS"},
    {"tx_id": "TX-4", "user_id": 904, "amount": "5600.00", "currency": "INR", "status": "FAILED"},
]

# 1. Create DataFrame
tx_df = pl.DataFrame(raw_transactions)

# 2. Inspect initial types
print("Initial Types before cleaning:\n", tx_df.schema)

# 3. Clean and cast amount from String to Float64
tx_cleaned = tx_df.with_columns(
    pl.col("amount").cast(pl.Float64).alias("amount_clean")
)
print("\nCleaned Transaction DataFrame:\n", tx_cleaned)
print(f"Total Successful Revenue (INR): {tx_cleaned.filter((pl.col('currency') == 'INR') & (pl.col('status') == 'SUCCESS'))['amount_clean'].sum():.2f}")

print("\n Module 1 Completed Successfully!")
