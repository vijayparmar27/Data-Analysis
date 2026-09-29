"""
Module 1: Pandas Fundamentals
=============================
Topics Covered:
- 1.1 Why Pandas? (Data structures: Series & DataFrame vs Python dicts & NumPy arrays)
- 1.2 Environment & Conventions (import pandas as pd, version verification)
- 1.3 Series Architecture (1D labelled array, indexes, dtype)
- 1.4 DataFrame Creation (from dict of lists, list of dicts, 2D NumPy array, records)
- 1.5 Core Inspection Attributes (.shape, .columns, .index, .dtypes, .ndim, .size)
- 1.6 Column Selection (single bracket Series vs double bracket DataFrame)
- 1.7 Row & Cell Selection (.loc[] label-based vs .iloc[] integer position-based)
- 1.8 Setting, Resetting & Renaming Indexes & Columns
- 1.9 Adding, Modifying & Dropping Columns and Rows
- 1.10 Practice Challenge for you to solve with complete solution!
"""

import sys
# Defensive guard: prevent local directory name from shadowing installed library
if "" in sys.path:
    sys.path.remove("")
if "." in sys.path:
    sys.path.remove(".")

import numpy as np
import pandas as pd


def section(title: str):
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


# ----------------------------------------------------------------------
# 1.1 Why Pandas? (The Tabular Data Workhorse)
# ----------------------------------------------------------------------
section("1.1 Why Pandas? (Series & DataFrame Architecture)")

# Comparison for Backend / Full-Stack Developers:
# - Python dict: { "id": 1, "name": "Fish" } -> Good for single objects, terrible for 1M rows.
# - NumPy ndarray: Homogeneous raw C-buffer -> Blazing fast for math, but NO column names or mixed types.
# - Pandas DataFrame: Tabular relational structure -> 2D table with heterogeneous typed columns,
#   named headers, explicit row index, missing value awareness, and SQL-like expressive queries.

print(f"Pandas Version: {pd.__version__}")
print(f"NumPy Version:  {np.__version__}")


# ----------------------------------------------------------------------
# 1.2 Pandas Series (1D Labelled Array)
# ----------------------------------------------------------------------
section("1.2 Pandas Series (1D Labelled Array)")

# A Series is a 1D labelled array capable of holding any data type (integers, strings, floats, objects).
# Unlike a NumPy 1D array, a Series has an explicit index (labels).

# Creating from a Python list:
prices = pd.Series([450.0, 720.5, 310.0, 600.0], name="unit_price")
print("Series from list:\n", prices)
print(f"Series dtype: {prices.dtype}, shape: {prices.shape}, name: {prices.name}")

# Creating from a dictionary with custom string indexes (like primary keys):
stock_levels = pd.Series(
    {"SKU-101": 50, "SKU-102": 12, "SKU-103": 0, "SKU-104": 85},
    name="stock_qty"
)
print("\nSeries with custom string index:\n", stock_levels)
print("Access by index label 'SKU-102':", stock_levels["SKU-102"])
print("Summary stats of series:\n", stock_levels.describe())


# ----------------------------------------------------------------------
# 1.3 Creating DataFrames (2D Heterogeneous Table)
# ----------------------------------------------------------------------
section("1.3 Creating DataFrames from Various Sources")

# Method A: Dict of lists (Most common for mock data & tests)
data_dict = {
    "product_id": [101, 102, 103, 104, 105],
    "product_name": ["Pomfret", "Tiger Prawns", "Kingfish", "Bombay Duck", "Crab"],
    "category": ["Fish", "Shellfish", "Fish", "Fish", "Shellfish"],
    "unit_price": [750.0, 950.0, 800.0, 350.0, 600.0],
    "stock_qty": [25, 40, 15, 60, 30],
    "is_active": [True, True, True, False, True]
}
df_from_dict = pd.DataFrame(data_dict)
print("DataFrame from Dict of Lists:\n", df_from_dict)

# Method B: List of Dictionaries (Matches JSON API payloads / MongoDB docs)
api_records = [
    {"user_id": "u1", "email": "alice@example.com", "total_orders": 5},
    {"user_id": "u2", "email": "bob@example.com", "total_orders": 12},
    {"user_id": "u3", "email": "carol@example.com", "total_orders": 1},
]
df_from_records = pd.DataFrame(api_records)
print("\nDataFrame from List of Dicts (API response format):\n", df_from_records)

# Method C: From a 2D NumPy array with custom column names
matrix = np.random.default_rng(42).integers(10, 100, size=(3, 3))
df_from_numpy = pd.DataFrame(matrix, columns=["metric_A", "metric_B", "metric_C"])
print("\nDataFrame from NumPy 2D Array:\n", df_from_numpy)


# ----------------------------------------------------------------------
# 1.4 Core DataFrame Attributes (Inspection)
# ----------------------------------------------------------------------
section("1.4 Core DataFrame Attributes")

df = df_from_dict.copy()

print("df.shape (rows, cols):", df.shape)
print("df.columns:           ", list(df.columns))
print("df.index:             ", df.index)
print("df.dtypes:\n", df.dtypes)
print(f"df.ndim: {df.ndim}, df.size (total cells): {df.size}")
print(f"Underlying raw NumPy buffer values:\n{df[['unit_price', 'stock_qty']].values}")


# ----------------------------------------------------------------------
# 1.5 Column Selection (Series vs DataFrame)
# ----------------------------------------------------------------------
section("1.5 Column Selection: Single Bracket vs Double Bracket")

# Single bracket df['col'] returns a 1D Pandas Series
col_series = df["product_name"]
print("Single bracket -> Series:\n", type(col_series), "\n", col_series.head(2))

# Double bracket df[['col']] returns a 2D Pandas DataFrame
col_dataframe = df[["product_name"]]
print("\nDouble bracket single col -> DataFrame:\n", type(col_dataframe), "\n", col_dataframe.head(2))

# Double bracket multiple cols -> DataFrame projection (like SQL SELECT col1, col2)
subset = df[["product_name", "unit_price", "stock_qty"]]
print("\nMultiple columns projection:\n", subset)


# ----------------------------------------------------------------------
# 1.6 Row & Cell Selection (.loc[] vs .iloc[])
# ----------------------------------------------------------------------
section("1.6 Row Selection: .loc[] (Label) vs .iloc[] (Integer Position)")

# Rule of Thumb:
# .loc[row_label, col_label]  -> Access by names / labels (inclusive of stop bound!)
# .iloc[row_idx, col_idx]     -> Access by 0-based integer positions (standard Python slice, stop is exclusive!)

print("First row by integer position (.iloc[0]):\n", df.iloc[0])

print("\nFirst 3 rows, specific columns by integer position (.iloc[0:3, 1:4]):\n",
      df.iloc[0:3, 1:4])

# Filter by index label:
print("\nFirst row by label (.loc[0, 'product_name']):", df.loc[0, "product_name"])
print("Rows 0 through 2 (inclusive!) by label with .loc[0:2, ['product_name', 'unit_price']]:\n",
      df.loc[0:2, ["product_name", "unit_price"]])


# ----------------------------------------------------------------------
# 1.7 Setting, Resetting & Renaming Indexes and Columns
# ----------------------------------------------------------------------
section("1.7 Index Management & Column Renaming")

# Set product_id as the DataFrame index (like a Primary Key in SQL):
df_indexed = df.set_index("product_id")
print("DataFrame with 'product_id' as index:\n", df_indexed.head(3))
print("Querying by product_id index label 102:\n", df_indexed.loc[102])

# Reset index back to default auto-increment integer:
df_reset = df_indexed.reset_index()
print("\nReset back to integer index:\n", df_reset.head(2))

# Renaming columns (dict mapping old_name -> new_name):
df_renamed = df.rename(columns={
    "product_name": "name",
    "unit_price": "price_inr",
    "stock_qty": "inventory_count"
})
print("\nRenamed columns:\n", df_renamed.columns.tolist())


# ----------------------------------------------------------------------
# 1.8 Adding, Modifying & Dropping Columns and Rows
# ----------------------------------------------------------------------
section("1.8 Adding, Modifying & Dropping Columns/Rows")

# Adding a calculated column (Vectorized - no loops!):
df["total_inventory_value"] = df["unit_price"] * df["stock_qty"]
print("Added 'total_inventory_value':\n", df[["product_name", "unit_price", "stock_qty", "total_inventory_value"]])

# Modifying column conditionally using np.where:
# (CASE WHEN unit_price >= 800 THEN 'Premium' ELSE 'Standard' END)
df["tier"] = np.where(df["unit_price"] >= 800.0, "Premium", "Standard")
print("\nConditional column 'tier':\n", df[["product_name", "unit_price", "tier"]])

# Dropping columns:
df_dropped = df.drop(columns=["is_active", "tier"])
print("\nAfter dropping columns:\n", df_dropped.columns.tolist())

# Dropping rows by index label:
df_dropped_row = df.drop(index=[3])  # Drop row at index 3 (Bombay Duck)
print(f"\nOriginal row count: {len(df)}, row count after dropping index 3: {len(df_dropped_row)}")


# ----------------------------------------------------------------------
# 1.9 Hands-on Practice Challenge
# ----------------------------------------------------------------------
section("1.9 Hands-on Practice Challenge")
print("""
PRACTICE EXERCISE:
1. Create a DataFrame representing 4 SaaS subscription plans:
   - plan_id: [1, 2, 3, 4]
   - plan_name: ["Free", "Starter", "Pro", "Enterprise"]
   - monthly_fee: [0, 29, 99, 299]
   - max_users: [1, 5, 25, 100]
   - is_popular: [False, False, True, False]
2. Set 'plan_id' as the index.
3. Add a column 'annual_fee_discounted' representing the annual price with a 20% discount:
   (monthly_fee * 12 * 0.80).
4. Use .loc[] to select all rows where 'monthly_fee' > 0 and display only 'plan_name' and 'annual_fee_discounted'.
5. Drop the 'is_popular' column.
""")

# --- Challenge Solution ---
plans_df = pd.DataFrame({
    "plan_id": [1, 2, 3, 4],
    "plan_name": ["Free", "Starter", "Pro", "Enterprise"],
    "monthly_fee": [0, 29, 99, 299],
    "max_users": [1, 5, 25, 100],
    "is_popular": [False, False, True, False]
})

# Step 2: Set index
plans_df = plans_df.set_index("plan_id")

# Step 3: Add calculated column
plans_df["annual_fee_discounted"] = plans_df["monthly_fee"] * 12 * 0.80

# Step 4: Use .loc with boolean mask
paid_plans = plans_df.loc[plans_df["monthly_fee"] > 0, ["plan_name", "annual_fee_discounted"]]
print("Paid Plans Annual Pricing:\n", paid_plans)

# Step 5: Drop column
plans_df = plans_df.drop(columns=["is_popular"])
print("\nFinal Plans DataFrame:\n", plans_df)
print("\n Module 1 Completed Successfully!")
