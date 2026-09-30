"""
Module 2: The Polars Expression Engine & Data Transformation
============================================================
Topics Covered:
- 2.1 The Expression Philosophy (Declarative, embarrassingly parallel, composable)
- 2.2 select() vs with_columns() mechanics
- 2.3 Column selection with pl.col() (single, multi, wildcards, regex patterns)
- 2.4 Modern Polars Selectors (polars.selectors as cs)
- 2.5 Row filtering with filter() and compound boolean conditions
- 2.6 Slicing, top_k(), and bottom_k()
- 2.7 Multi-column sorting and null placement
- 2.8 Renaming, dropping, and casting columns
- 2.9 Unique values and value_counts()
- 2.10 Handling nulls and NaNs (is_null, fill_null, drop_nulls)
- 2.11 String manipulations via .str accessor
- 2.12 Conditional branching with pl.when().then().otherwise() (SQL CASE WHEN)
- 2.13 Hands-on Transformation Pipeline Challenge
"""

import sys
from pathlib import Path

# Defensive guard: prevent local folder name from shadowing installed library
_script_dir = Path(__file__).resolve().parent
_parent_dir = _script_dir.parent
for _p in ["", ".", str(_script_dir), str(_parent_dir)]:
    while _p in sys.path:
        sys.path.remove(_p)

import polars as pl
import polars.selectors as cs

sys.path.append(str(_parent_dir))


def section(title: str):
    print(f"\n{'=' * 75}\n  {title}\n{'=' * 75}")


# Sample e-commerce order dataset for demonstration
orders = pl.DataFrame(
    {
        "order_id": ["ORD-101", "ORD-102", "ORD-103", "ORD-104", "ORD-105", "ORD-106", "ORD-107"],
        "customer_id": [501, 502, 503, 501, 504, 502, 505],
        "category": ["Electronics", "Books", "Electronics", "Groceries", "Fashion", "Electronics", "Groceries"],
        "item_name": ["USB-C Hub", "Rust Programming Guide", "Wireless Earbuds", "Organic Coffee", "Cotton Hoodie", "4K Monitor", "Green Tea"],
        "unit_price": [45.0, 32.5, 79.99, 14.5, 55.0, 320.0, None],
        "quantity": [2, 1, 1, 4, 2, 1, 3],
        "discount_pct": [0.10, 0.0, 0.15, 0.05, 0.20, 0.0, 0.0],
        "status": ["delivered", "shipped", "delivered", "cancelled", "delivered", "processing", "delivered"],
    }
)

print("Baseline Orders Dataset:")
print(orders)


# ----------------------------------------------------------------------
# 2.1 select() vs with_columns()
# ----------------------------------------------------------------------
section("2.1 select() vs. with_columns()")

# select(): Evaluates expressions and produces a NEW DataFrame containing ONLY the evaluated columns.
# (Like SQL: SELECT expr1, expr2 FROM table)
selected_df = orders.select(
    pl.col("order_id"),
    pl.col("unit_price"),
    (pl.col("unit_price") * pl.col("quantity")).alias("gross_amount"),
)
print("select() output (only requested columns):\n", selected_df.head(3))

# with_columns(): Appends or replaces columns while keeping ALL existing columns intact.
# (Like SQL: SELECT *, expr1, expr2 FROM table)
updated_df = orders.with_columns(
    (pl.col("unit_price") * pl.col("quantity")).alias("gross_amount"),
    (pl.col("unit_price") * pl.col("quantity") * (1.0 - pl.col("discount_pct"))).round(2).alias("net_amount"),
)
print("\nwith_columns() output (original columns preserved):\n", updated_df.head(3))


# ----------------------------------------------------------------------
# 2.2 Column Selectors with pl.col() & Regex Patterns
# ----------------------------------------------------------------------
section("2.2 Advanced pl.col() Patterns (Wildcards, Types & Regex)")

# Wildcards: apply transformation across ALL columns
print("All columns lowercase column names or type check:")
print(orders.select(pl.col(pl.String).str.to_uppercase()).head(2))

# Regex matching: select columns matching a regex pattern
print("\nRegex matching columns containing 'price' or 'amount':")
print(updated_df.select(pl.col(r"^.*(price|amount).*$")).head(2))


# ----------------------------------------------------------------------
# 2.3 Modern Polars Selectors (polars.selectors as cs)
# ----------------------------------------------------------------------
section("2.3 Modern Selectors with polars.selectors (cs)")

# Polars provides intuitive type selectors:
# cs.numeric(), cs.string(), cs.temporal(), cs.boolean(), cs.all()
numeric_cols = updated_df.select(cs.numeric())
print("Numeric columns selected with cs.numeric():\n", numeric_cols.head(2))

# Combining selectors with set operations: cs.all() - cs.string()
non_string_cols = updated_df.select(cs.all() - cs.string())
print("\nAll columns EXCEPT string columns (cs.all() - cs.string()):\n", non_string_cols.head(2))


# ----------------------------------------------------------------------
# 2.4 Row Filtering with filter()
# ----------------------------------------------------------------------
section("2.4 Row Filtering with filter() & Compound Conditions")

# Filter rows where status is 'delivered' AND unit_price > 50
high_value_delivered = updated_df.filter(
    (pl.col("status") == "delivered") & (pl.col("unit_price") > 50.0)
)
print("High-value delivered orders:\n", high_value_delivered)

# Filter with `is_in()` (equivalent to SQL IN ('Electronics', 'Fashion'))
filtered_categories = orders.filter(
    pl.col("category").is_in(["Electronics", "Fashion"])
)
print("\nOrders in Electronics or Fashion:\n", filtered_categories)


# ----------------------------------------------------------------------
# 2.5 Slicing, top_k(), and bottom_k()
# ----------------------------------------------------------------------
section("2.5 Slicing, top_k(), and bottom_k()")

# Top 3 most expensive orders by unit_price:
top_expensive = orders.top_k(3, by="unit_price")
print("Top 3 most expensive items (.top_k()):\n", top_expensive.select("order_id", "item_name", "unit_price"))

# Row slice: offset 2, length 3
sliced_rows = orders.slice(2, 3)
print("\nSliced 3 rows from offset 2 (.slice(2, 3)):\n", sliced_rows.select("order_id", "item_name"))


# ----------------------------------------------------------------------
# 2.6 Multi-Column Sorting
# ----------------------------------------------------------------------
section("2.6 Sorting with sort() & Null Placement")

sorted_df = orders.sort(
    by=["category", "unit_price"],
    descending=[False, True],
    nulls_last=True,
)
print("Sorted by Category ASC, Price DESC (Nulls last):\n", sorted_df.select("category", "item_name", "unit_price"))


# ----------------------------------------------------------------------
# 2.7 Handling Missing Values (Nulls & NaNs)
# ----------------------------------------------------------------------
section("2.7 Handling Missing Values (is_null, fill_null, drop_nulls)")

# Inspecting nulls
null_count_df = orders.select(pl.all().null_count())
print("Null counts across all columns:\n", null_count_df)

# Impute null unit_price with category mean or default constant
imputed_df = orders.with_columns(
    pl.col("unit_price").fill_null(strategy="backward").alias("price_bfilled"),
    pl.col("unit_price").fill_null(0.0).alias("price_zero_filled"),
)
print("\nDataFrame with Imputed Null Prices:\n", imputed_df.select("order_id", "unit_price", "price_bfilled", "price_zero_filled"))


# ----------------------------------------------------------------------
# 2.8 String Manipulation via .str Accessor
# ----------------------------------------------------------------------
section("2.8 High-Performance String Manipulations (.str)")

string_ops = orders.select(
    pl.col("item_name"),
    pl.col("item_name").str.to_uppercase().alias("name_upper"),
    pl.col("item_name").str.contains("(?i)coffee|tea").alias("is_beverage"),
    pl.col("order_id").str.slice(4).cast(pl.Int32).alias("order_num"),
)
print("String Accessor operations:\n", string_ops)


# ----------------------------------------------------------------------
# 2.9 Conditional Logic with pl.when().then().otherwise()
# ----------------------------------------------------------------------
section("2.9 Conditional Branching (SQL CASE WHEN Equivalent)")

# Categorize order priority based on price and quantity
categorized_df = updated_df.with_columns(
    pl.when(pl.col("net_amount") >= 200.0)
    .then(pl.lit("VIP"))
    .when(pl.col("net_amount") >= 50.0)
    .then(pl.lit("STANDARD"))
    .otherwise(pl.lit("LOW_VALUE"))
    .alias("order_tier")
)
print("Categorized Order Tiers:\n", categorized_df.select("order_id", "net_amount", "order_tier"))


# ----------------------------------------------------------------------
# 2.10 Hands-on Pipeline Exercise
# ----------------------------------------------------------------------
section("2.10 Hands-on Exercise: Clean & Enrich Raw Sensor Data")

raw_sensor_data = pl.DataFrame(
    {
        "device_id": ["DEV_01", "DEV_02", "DEV_01", "DEV_03", "DEV_02"],
        "reading_celsius": [24.5, -999.0, 26.1, None, 29.8],
        "humidity_pct": [55.0, 60.0, None, 80.0, 52.0],
        "battery_volts": [3.7, 3.2, 3.6, 2.9, 3.8],
    }
)

# Pipeline:
# 1. Replace error code -999.0 with null
# 2. Fill nulls in temperature with 25.0 (default room temp)
# 3. Add battery health flag: LOW (< 3.0V), NORMAL (>= 3.0V)
# 4. Filter out devices with humidity null
cleaned_sensors = (
    raw_sensor_data
    .with_columns(
        pl.when(pl.col("reading_celsius") == -999.0)
        .then(None)
        .otherwise(pl.col("reading_celsius"))
        .fill_null(25.0)
        .alias("temp_clean"),
        pl.when(pl.col("battery_volts") < 3.0)
        .then(pl.lit("LOW"))
        .otherwise(pl.lit("NORMAL"))
        .alias("battery_status"),
    )
    .filter(pl.col("humidity_pct").is_not_null())
)

print("Original Sensor Readings:\n", raw_sensor_data)
print("\nCleaned & Enriched Sensors:\n", cleaned_sensors)

print("\n Module 2 Completed Successfully!")
