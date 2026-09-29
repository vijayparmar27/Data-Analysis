"""
Module 5: Data Combining & Advanced Pandas
==========================================
Topics Covered:
- 5.1 Relational Merges & SQL-Style Joins (inner, left, right, outer, indicator, validate)
- 5.2 Concatenation & Stacking (pd.concat along rows/cols, ignore_index, keys)
- 5.3 Reshaping Data (Wide to Long with pd.melt() & Long to Wide with pivot())
- 5.4 Multi-Level Index Transformations (stack() & unstack())
- 5.5 Time Series & Date Handling (.dt accessor, Timedeltas, date ranges)
- 5.6 Time Series Resampling (.resample('D' / 'W' / 'ME') for financial / sales rollups)
- 5.7 Advanced Windowing (Exponentially Weighted Moving Average - ewm())
- 5.8 Categorical Data Types (Memory footprint benchmark: 80%+ savings)
- 5.9 High-Performance Vectorization & pd.eval() (Speed benchmarking)
- 5.10 Chunk Processing for Large Multi-GB Datasets (read_csv(chunksize=...))
- 5.11 Method Chaining & .pipe() Pipeline Architecture
- 5.12 Hands-on Practice Challenge with complete solution!
"""

import io
import sys
import time

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
# 5.1 Relational Merges & SQL-Style Joins
# ----------------------------------------------------------------------
section("5.1 Relational Merges & SQL Joins (pd.merge)")

users = pd.DataFrame({
    "user_id": [1, 2, 3, 4],
    "name": ["Alice", "Bob", "Charlie", "David"],
    "country": ["IN", "US", "DE", "IN"]
})

orders = pd.DataFrame({
    "order_id": [101, 102, 103, 104, 105],
    "customer_id": [1, 2, 1, 3, 99],  # Note: 99 is an orphaned order with no matching user!
    "amount": [250.0, 180.0, 420.0, 95.0, 60.0]
})

print("Users Table:\n", users)
print("\nOrders Table:\n", orders)

# 1. Inner Join (Keep only matching records on both sides)
inner_join = pd.merge(users, orders, left_on="user_id", right_on="customer_id", how="inner")
print("\nInner Join (users JOIN orders):\n", inner_join[["user_id", "name", "order_id", "amount"]])

# 2. Left Join (Keep all users, even those with zero orders)
left_join = pd.merge(users, orders, left_on="user_id", right_on="customer_id", how="left")
print("\nLeft Join (Keep all users):\n", left_join[["user_id", "name", "order_id", "amount"]])

# 3. Outer Join with indicator=True (Identifies matching vs unmatching records like full outer join audit)
outer_join = pd.merge(users, orders, left_on="user_id", right_on="customer_id", how="outer", indicator=True)
print("\nOuter Join with _merge audit indicator:\n",
      outer_join[["user_id", "name", "order_id", "amount", "_merge"]])

# 4. Join Integrity Validation: validate='1:m' ensures left table has unique primary keys!
validated_join = pd.merge(users, orders, left_on="user_id", right_on="customer_id", how="inner", validate="1:m")
print("\nJoin validated 1-to-Many relationship successfully!")


# ----------------------------------------------------------------------
# 5.2 Concatenation (pd.concat)
# ----------------------------------------------------------------------
section("5.2 Concatenation: Stacking Rows & Appending Columns")

# Simulating monthly sales tables (SQL UNION ALL equivalent)
q1_sales = pd.DataFrame({"month": ["Jan", "Feb", "Mar"], "sales": [10000, 12000, 15000]})
q2_sales = pd.DataFrame({"month": ["Apr", "May", "Jun"], "sales": [14000, 16000, 19000]})

# Row-wise concatenation (axis=0)
h1_sales = pd.concat([q1_sales, q2_sales], ignore_index=True)
print("Stacked Rows (SQL UNION ALL equivalent):\n", h1_sales)

# Column-wise concatenation (axis=1) with hierarchical keys
q1_budget = pd.DataFrame({"budget": [9500, 11000, 14000]})
combined_q1 = pd.concat([q1_sales, q1_budget], axis=1)
print("\nAppended Columns (axis=1):\n", combined_q1)


# ----------------------------------------------------------------------
# 5.3 Reshaping: Wide to Long (melt) and Long to Wide (pivot)
# ----------------------------------------------------------------------
section("5.3 Reshaping Data: pd.melt() & pivot()")

# Wide format (Common in spreadsheets / financial models):
wide_financials = pd.DataFrame({
    "department": ["Engineering", "Marketing", "Sales"],
    "Q1_spend": [150000, 80000, 95000],
    "Q2_spend": [165000, 92000, 110000],
    "Q3_spend": [170000, 85000, 125000]
})
print("Wide Format Table:\n", wide_financials)

# Unpivot / Melt: Convert wide quarterly columns into a tidy normalized long format
# (Tidy data is the golden standard for databases, BI charts, and regression models!)
tidy_long = pd.melt(
    wide_financials,
    id_vars=["department"],
    value_vars=["Q1_spend", "Q2_spend", "Q3_spend"],
    var_name="quarter",
    value_name="spend_usd"
)
print("\nNormalized Long Format (via pd.melt):\n", tidy_long.head(6))

# Pivot back from Long to Wide:
wide_again = tidy_long.pivot(index="department", columns="quarter", values="spend_usd")
print("\nPivoted Back to Wide Format:\n", wide_again)


# ----------------------------------------------------------------------
# 5.4 Index Pivoting: stack() and unstack()
# ----------------------------------------------------------------------
section("5.4 Index Pivoting: stack() and unstack()")

# Grouped MultiIndex table
grouped_summary = tidy_long.groupby(["department", "quarter"])["spend_usd"].sum()
print("MultiIndex Series:\n", grouped_summary)

# unstack(): Moves innermost index level (quarter) to column headers
unstacked = grouped_summary.unstack(level="quarter")
print("\nAfter .unstack('quarter') -> Columns pivoted:\n", unstacked)

# stack(): Moves column headers back into row index levels
stacked_again = unstacked.stack()
print("\nAfter .stack() -> Back to MultiIndex:\n", stacked_again.head(4))


# ----------------------------------------------------------------------
# 5.5 Time Series Handling & .dt Accessor
# ----------------------------------------------------------------------
section("5.5 Time Series Fundamentals & the .dt Accessor")

timestamps = pd.date_range(start="2026-01-01 08:00:00", periods=5, freq="14h")
ts_df = pd.DataFrame({"event_time": timestamps, "sensor_reading": [22.4, 23.1, 19.8, 25.6, 21.0]})

# Extract datetime attributes using the .dt accessor:
ts_df["year"] = ts_df["event_time"].dt.year
ts_df["month"] = ts_df["event_time"].dt.month_name()
ts_df["day_name"] = ts_df["event_time"].dt.day_name()
ts_df["hour"] = ts_df["event_time"].dt.hour
ts_df["is_weekend"] = ts_df["event_time"].dt.dayofweek >= 5

print("Extracted Datetime Properties:\n", ts_df)


# ----------------------------------------------------------------------
# 5.6 Time Series Resampling (.resample())
# ----------------------------------------------------------------------
section("5.6 Time Series Resampling (Downsampling & Aggregation)")

# High-frequency hourly transactional stream
hourly_dates = pd.date_range(start="2026-03-01", periods=72, freq="h")
rng = np.random.default_rng(42)
txn_stream = pd.DataFrame({
    "timestamp": hourly_dates,
    "transactions": rng.integers(5, 50, size=72),
    "revenue": rng.uniform(200, 3000, size=72)
}).set_index("timestamp")

# Resample down from Hourly ('h') to Daily ('D')
daily_rollup = txn_stream.resample("D").agg(
    total_txns=("transactions", "sum"),
    total_revenue=("revenue", "sum"),
    avg_txn_value=("revenue", "mean")
)
print("Resampled 72 Hourly Records into 3 Daily Financial Aggregates:\n", daily_rollup.round(2))


# ----------------------------------------------------------------------
# 5.7 Advanced Windowing: Exponentially Weighted Moving Average (ewm)
# ----------------------------------------------------------------------
section("5.7 Exponentially Weighted Moving Average (.ewm())")

# EWM gives more weight to recent data points compared to older ones (vital for financial volatility & algorithmic trading)
daily_revenue = daily_rollup["total_revenue"]
ewm_series = daily_revenue.ewm(span=3, adjust=False).mean()

comparison_window = pd.DataFrame({
    "actual_revenue": daily_revenue,
    "simple_rolling_2d": daily_revenue.rolling(2).mean(),
    "exponential_weighted_ewm": ewm_series
})
print("Simple Rolling vs Exponential Weighted Window:\n", comparison_window.round(2))


# ----------------------------------------------------------------------
# 5.8 Memory Optimization with Categorical Data Types
# ----------------------------------------------------------------------
section("5.8 Memory Optimization: Categorical Dtypes Benchmark")

# Create a dataset with 500,000 repeated string records (e.g. status, region, plan)
n_rows = 500_000
statuses = ["PENDING", "COMPLETED", "FAILED", "REFUNDED", "PROCESSING"]

raw_statuses = np.random.choice(statuses, size=n_rows)
df_object = pd.DataFrame({"status": raw_statuses})

# Memory usage of standard 'object' dtype (Python heap pointers):
mem_object = df_object.memory_usage(deep=True)["status"]

# Convert to optimized 'category' dtype (stores integers referencing a unique dictionary pool):
df_category = df_object.copy()
df_category["status"] = df_category["status"].astype("category")
mem_category = df_category.memory_usage(deep=True)["status"]

print(f"Memory with standard 'object' dtype:   {mem_object / (1024 * 1024):.2f} MB")
print(f"Memory with optimized 'category' dtype: {mem_category / (1024 * 1024):.2f} MB")
reduction_pct = ((mem_object - mem_category) / mem_object) * 100
print(f"⚡ Memory Reduction: {reduction_pct:.1f}% less RAM consumed!")


# ----------------------------------------------------------------------
# 5.9 High-Performance Vectorization & pd.eval()
# ----------------------------------------------------------------------
section("5.9 Performance Benchmark: Vectorization vs Iteration")

benchmark_df = pd.DataFrame({
    "a": np.random.rand(1_000_000),
    "b": np.random.rand(1_000_000),
    "c": np.random.rand(1_000_000)
})

# Approach 1: Vectorized arithmetic
t0 = time.perf_counter()
res_vectorized = benchmark_df["a"] * 2 + benchmark_df["b"] - benchmark_df["c"]
t_vec = time.perf_counter() - t0

# Approach 2: pd.eval() (compiles expression to C-level vectorized code without intermediate allocations)
t0 = time.perf_counter()
res_eval = pd.eval("benchmark_df.a * 2 + benchmark_df.b - benchmark_df.c")
t_eval = time.perf_counter() - t0

print(f"Vectorized NumPy-backed arithmetic (1M rows): {t_vec:.4f}s")
print(f"High-throughput pd.eval() expression:        {t_eval:.4f}s")


# ----------------------------------------------------------------------
# 5.10 Chunk Processing for Large Multi-GB Datasets
# ----------------------------------------------------------------------
section("5.10 Chunk Processing for Large Datasets (Out-of-Core Processing)")

# When a CSV is 50GB and cannot fit in RAM, read it in chunks:
mock_big_csv = """order_id,category,amount
1,Books,12.50
2,Electronics,450.00
3,Books,8.99
4,Clothing,75.00
5,Electronics,120.00
6,Books,19.99
"""

category_totals = {}
# Process in micro-batches of 2 rows per chunk:
for chunk_idx, chunk in enumerate(pd.read_csv(io.StringIO(mock_big_csv), chunksize=2)):
    print(f"Processing Chunk {chunk_idx + 1} (Shape: {chunk.shape})...")
    chunk_summary = chunk.groupby("category")["amount"].sum()
    for cat, amt in chunk_summary.items():
        category_totals[cat] = category_totals.get(cat, 0.0) + amt

aggregated_df = pd.DataFrame(list(category_totals.items()), columns=["category", "total_amount"])
print("\nStream-Aggregated Results across all chunks without loading entire file into RAM:\n", aggregated_df)


# ----------------------------------------------------------------------
# 5.11 Method Chaining & the .pipe() Pipeline Architecture
# ----------------------------------------------------------------------
section("5.11 Method Chaining & .pipe() Pipeline Architecture")

# In production applications (FastAPI backend / data microservices),
# clean idempotent pipelines are composed of pure transformation functions:

def filter_active_customers(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["is_active"] == True]

def compute_net_spend(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["net_spend"] = df["gross_spend"] - df["discount"]
    return df

def assign_spend_tier(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tier"] = np.where(df["net_spend"] > 500, "High Value", "Standard")
    return df

# Raw customer input
raw_customers = pd.DataFrame({
    "customer": ["Cust_A", "Cust_B", "Cust_C", "Cust_D"],
    "gross_spend": [750.0, 320.0, 1200.0, 150.0],
    "discount": [50.0, 20.0, 100.0, 0.0],
    "is_active": [True, False, True, True]
})

# Production Pipeline using .pipe()
clean_pipeline_result = (
    raw_customers
    .pipe(filter_active_customers)
    .pipe(compute_net_spend)
    .pipe(assign_spend_tier)
)

print("Pipeline Processed Data (Clean, Testable, Composable):\n", clean_pipeline_result)


# ----------------------------------------------------------------------
# 5.12 Hands-on Practice Challenge
# ----------------------------------------------------------------------
section("5.12 Hands-on Practice Challenge")
print("""
PRACTICE EXERCISE:
Given two relational tables:
customers = pd.DataFrame({
    "id": [1, 2, 3],
    "name": ["Aarav", "Priya", "Rohan"],
    "city": ["Mumbai", "Delhi", "Bengaluru"]
})

orders = pd.DataFrame({
    "order_id": [101, 102, 103, 104],
    "cust_id": [1, 1, 2, 4],  # Note: customer 4 is not in customers table
    "amount": [500, 750, 1200, 300]
})

Your Task:
1. Perform a LEFT JOIN of customers with orders on id == cust_id.
2. Fill NaN order amounts with 0.
3. Compute total lifetime spend for each customer.
4. Convert 'city' to categorical dtype and print its memory savings.
""")

# --- Challenge Solution ---
customers = pd.DataFrame({
    "id": [1, 2, 3],
    "name": ["Aarav", "Priya", "Rohan"],
    "city": ["Mumbai", "Delhi", "Bengaluru"]
})

orders = pd.DataFrame({
    "order_id": [101, 102, 103, 104],
    "cust_id": [1, 1, 2, 4],
    "amount": [500, 750, 1200, 300]
})

# 1. Left Join
merged = pd.merge(customers, orders, left_on="id", right_on="cust_id", how="left")

# 2. Fill NaN amounts
merged["amount"] = merged["amount"].fillna(0.0)

# 3. Lifetime spend per customer
clv_summary = merged.groupby(["id", "name"])["amount"].sum().reset_index(name="total_spend")
print("Customer Lifetime Spend:\n", clv_summary)

# 4. Convert city to category
customers["city"] = customers["city"].astype("category")
print("\nCustomers with categorical 'city':\n", customers.dtypes)
print("\n Module 5 Completed Successfully!")
