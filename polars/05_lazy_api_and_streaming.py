"""
Module 5: The Lazy API, Query Optimizer & Streaming Execution
=============================================================
Topics Covered:
- 5.1 Eager vs. Lazy Execution Architecture (The Query Graph)
- 5.2 Lazy Scanning (scan_csv, scan_parquet, scan_ndjson)
- 5.3 Schema Validation without Computation (.collect_schema())
- 5.4 Query Plan Inspection with .explain() (AST vs. Optimized Logical Plan)
- 5.5 Predicate Pushdown in Action (Filter pushed to disk scan)
- 5.6 Projection Pushdown in Action (Only read required columns)
- 5.7 Common Subexpression Elimination (CSE)
- 5.8 Slice Pushdown (Pushing head/limit to scanner)
- 5.9 Out-of-Core Streaming Execution (collect(streaming=True))
- 5.10 Query Profiling & Bottleneck Inspection (.profile())
- 5.11 Lazy Pipeline Design Pattern for Production
"""

import sys
import tempfile
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


tmp_dir = Path(tempfile.mkdtemp(prefix="polars_module5_"))

# Generate a synthetic dataset and save as Parquet
n_records = 200_000
raw_dataset = pl.DataFrame(
    {
        "order_id": [f"ORD_{i:06d}" for i in range(n_records)],
        "customer_id": [i % 5000 for i in range(n_records)],
        "region": [["North", "South", "East", "West"][i % 4] for i in range(n_records)],
        "product_category": [["Electronics", "Apparel", "Home", "Sports", "Books"][i % 5] for i in range(n_records)],
        "order_amount": [15.0 + (i % 500) * 1.5 for i in range(n_records)],
        "shipping_cost": [5.0 + (i % 20) * 0.5 for i in range(n_records)],
        "is_returned": [(i % 17 == 0) for i in range(n_records)],
    }
)

parquet_file = tmp_dir / "large_orders.parquet"
raw_dataset.write_parquet(parquet_file, compression="zstd")
print(f"Created benchmark Parquet: {parquet_file.name} ({parquet_file.stat().st_size / 1024:.1f} KB, {n_records:,} rows)")


# ----------------------------------------------------------------------
# 5.1 Eager vs. Lazy Execution Architecture
# ----------------------------------------------------------------------
section("5.1 Eager vs. Lazy Execution Architecture")

# Eager: Executes immediately in RAM (returns DataFrame)
eager_df = pl.read_parquet(parquet_file).filter(pl.col("order_amount") > 500)
print(f"Eager evaluation materialized: {type(eager_df)} with shape {eager_df.shape}")

# Lazy: Builds a computation graph without executing any work! (returns LazyFrame)
lazy_q = pl.scan_parquet(parquet_file).filter(pl.col("order_amount") > 500)
print(f"Lazy plan created: {type(lazy_q)} (Zero disk I/O performed yet)")


# ----------------------------------------------------------------------
# 5.2 Schema Validation without Materialization (.collect_schema())
# ----------------------------------------------------------------------
section("5.2 Schema Validation without Execution (.collect_schema())")

# In production pipelines, you want to validate column names and types BEFORE
# loading gigabytes of data. .collect_schema() checks types in microsecond time!
lazy_pipeline = (
    pl.scan_parquet(parquet_file)
    .filter(pl.col("is_returned") == False)
    .with_columns(
        (pl.col("order_amount") + pl.col("shipping_cost")).alias("total_bill")
    )
)

schema = lazy_pipeline.collect_schema()
print("Validated Output Schema before execution:")
for col_name, dtype in schema.items():
    print(f" - {col_name:<20}: {dtype}")


# ----------------------------------------------------------------------
# 5.3 Inspecting Query Plans with .explain()
# ----------------------------------------------------------------------
section("5.3 Inspecting Query Plans with .explain()")

query = (
    pl.scan_parquet(parquet_file)
    .filter(pl.col("region") == "North")
    .filter(pl.col("order_amount") > 200)
    .select("customer_id", "product_category", "order_amount")
    .group_by("product_category")
    .agg(
        pl.col("order_amount").sum().alias("total_revenue"),
        pl.len().alias("order_count"),
    )
    .sort("total_revenue", descending=True)
)

# Unoptimized Abstract Syntax Tree (AST):
print("Unoptimized Query Plan (Raw AST):")
print(query.explain(optimized=False))

# Optimized Physical / Logical Plan:
print("\nOptimized Query Plan (After Predicate & Projection Pushdown):")
print(query.explain(optimized=True))


# ----------------------------------------------------------------------
# 5.4 Optimization Mechanics: Predicate & Projection Pushdown
# ----------------------------------------------------------------------
section("5.4 Optimization Mechanics: Predicate & Projection Pushdown")

# 1. PREDICATE PUSHDOWN:
# In the optimized plan above, notice how:
#   SELECTION: [([(col("region")) == ("North")]) & ([(col("order_amount")) > (200.0)])]
# was pushed directly into the PARQUET SCAN step!
# Polars uses Parquet file metadata (row-group statistics: min/max) to skip entire
# chunks of files from disk without reading them into memory!

# 2. PROJECTION PUSHDOWN:
# Downstream, we only needed ['customer_id', 'product_category', 'order_amount', 'region'].
# Polars does NOT read columns like 'order_id', 'shipping_cost', or 'is_returned' from disk!

# Executing the optimized query:
final_result = query.collect()
print("Query Execution Output:\n", final_result)


# ----------------------------------------------------------------------
# 5.5 Common Subexpression Elimination (CSE)
# ----------------------------------------------------------------------
section("5.5 Common Subexpression Elimination (CSE)")

# When expressions compute identical math multiple times:
complex_math_query = (
    pl.scan_parquet(parquet_file)
    .select(
        (pl.col("order_amount") * 1.18).alias("tax_included"),
        ((pl.col("order_amount") * 1.18) + pl.col("shipping_cost")).alias("final_charge"),
    )
)

print("Optimized Plan with Common Subexpression Elimination:")
# Notice Polars evaluates (order_amount * 1.18) only ONCE instead of recalculating twice!
print(complex_math_query.explain(optimized=True))


# ----------------------------------------------------------------------
# 5.6 Out-of-Core Streaming Execution
# ----------------------------------------------------------------------
section("5.6 Out-of-Core Streaming Execution (Datasets larger than RAM)")

# Polars' streaming engine processes data in streaming batches.
# This allows processing 100GB datasets on a machine with only 16GB of RAM!
streaming_query = (
    pl.scan_parquet(parquet_file)
    .filter(pl.col("order_amount") > 100.0)
    .group_by("region", "product_category")
    .agg(
        pl.len().alias("count"),
        pl.col("order_amount").sum().alias("sum_amount"),
    )
)

# In Polars >= 1.25, use engine="streaming" (or streaming=True in older versions)
streaming_df = streaming_query.collect(engine="streaming")
print("Streaming execution completed successfully:\n", streaming_df.head(4))


# ----------------------------------------------------------------------
# 5.7 Query Profiling (.profile())
# ----------------------------------------------------------------------
section("5.7 Query Profiling (.profile()) for Bottleneck Detection")

# .profile() executes the query and returns (result_df, profile_df)
# showing the exact time spent in each physical operator!
profiled_result, timing_df = query.profile()
print("Execution Result:\n", profiled_result)
print("\nPhysical Operator Profiling Breakdown:\n", timing_df)

print("\n Module 5 Completed Successfully!")
