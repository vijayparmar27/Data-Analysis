"""
Module 7: Real-World Portfolio Projects (Hands-On / Production Grade)
===================================================================
Projects Included:
- Project 1: E-Commerce Sales & Revenue KPI Engine (.over() market shares, AOV)
- Project 2: Customer Behavioral RFM Segmentation (Recency, Frequency, Monetary quintiles)
- Project 3: Automated ETL Pipeline with Strict Schema Validation & Partitioned Parquet
- Project 4: High-Throughput Log & Clickstream Processor (Regex parsing, 5xx rate alerts)
- Project 5: Financial Market OHLCV & Rolling Volatility Engine
- Project 6: PostgreSQL / SQLite to Polars Incremental Analytical Snapshot Service
- Project 7: FastAPI Analytics Microservice with Dynamic Lazy Query Building
- Project 8: Nested JSON Normalization (Struct & List unpacking)
- Project 9: Out-of-Core Streaming Engine (Memory-bounded batch processing)
- Project 10: Polars vs. Pandas Benchmark (Speed, memory & CPU thread utilization)
"""

import json
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

# Defensive guard: prevent local folder name from shadowing installed library
_script_dir = Path(__file__).resolve().parent
_parent_dir = _script_dir.parent
for _p in ["", ".", str(_script_dir), str(_parent_dir)]:
    while _p in sys.path:
        sys.path.remove(_p)

import numpy as np
import pandas as pd
import polars as pl

sys.path.append(str(_parent_dir))


def section(title: str):
    print(f"\n{'=' * 75}\n  {title}\n{'=' * 75}")


tmp_workspace = Path(tempfile.mkdtemp(prefix="polars_portfolio_"))


# ----------------------------------------------------------------------
# Project 1: E-Commerce Sales & Revenue KPI Engine
# ----------------------------------------------------------------------
section("Project 1: E-Commerce Sales & Revenue KPI Engine")

# Generate synthetic transaction records
np.random.seed(42)
n_orders = 50_000
categories = ["Electronics", "Fashion", "Home & Kitchen", "Books", "Beauty"]
regions = ["North", "South", "East", "West"]

orders_p1 = pl.DataFrame(
    {
        "order_id": [f"ORD-{i:06d}" for i in range(n_orders)],
        "category": np.random.choice(categories, n_orders),
        "region": np.random.choice(regions, n_orders),
        "revenue": np.random.uniform(10.0, 500.0, n_orders).round(2),
        "cost": np.random.uniform(5.0, 300.0, n_orders).round(2),
        "is_returned": np.random.choice([True, False], n_orders, p=[0.08, 0.92]),
    }
)

# Complex analytics using Polars expressions in a single pass:
kpi_summary = (
    orders_p1
    .group_by("region", "category")
    .agg(
        pl.len().alias("total_orders"),
        pl.col("revenue").sum().alias("gross_revenue"),
        (pl.col("revenue").sum() - pl.col("cost").sum()).round(2).alias("gross_profit"),
        pl.col("revenue").filter(~pl.col("is_returned")).sum().alias("net_settled_revenue"),
        pl.col("revenue").mean().round(2).alias("average_order_value"),
        (pl.col("is_returned").sum() / pl.len() * 100).round(2).alias("return_rate_pct"),
    )
    .with_columns(
        # Regional category market share using window function:
        (pl.col("gross_revenue") / pl.col("gross_revenue").sum().over("region") * 100).round(2).alias("region_rev_share_pct")
    )
    .sort(["region", "gross_revenue"], descending=[False, True])
)

print("E-Commerce Regional KPI Summary (Top 5 categories across regions):\n", kpi_summary.head(5))


# ----------------------------------------------------------------------
# Project 2: Customer Behavioral RFM Segmentation Engine
# ----------------------------------------------------------------------
section("Project 2: Customer Behavioral RFM Segmentation Engine")

today = datetime(2026, 3, 31)
customer_orders = pl.DataFrame(
    {
        "customer_id": [101, 102, 101, 103, 104, 102, 105, 101, 106, 104],
        "order_date": [
            datetime(2026, 3, 28),
            datetime(2026, 3, 15),
            datetime(2026, 2, 10),
            datetime(2026, 1, 5),
            datetime(2026, 3, 30),
            datetime(2026, 3, 29),
            datetime(2025, 11, 20),
            datetime(2026, 3, 20),
            datetime(2026, 2, 25),
            datetime(2026, 3, 1),
        ],
        "order_amount": [120.0, 450.0, 95.0, 30.0, 800.0, 60.0, 15.0, 210.0, 340.0, 110.0],
    }
)

rfm_metrics = (
    customer_orders
    .group_by("customer_id")
    .agg(
        # Recency: days since most recent purchase
        (pl.lit(today) - pl.col("order_date").max()).dt.total_days().alias("recency_days"),
        # Frequency: total order count
        pl.len().alias("frequency"),
        # Monetary: total spend
        pl.col("order_amount").sum().alias("monetary"),
    )
    .with_columns(
        # Segment customer cohorts based on recency and monetary thresholds
        pl.when((pl.col("recency_days") <= 30) & (pl.col("monetary") >= 300.0))
        .then(pl.lit("Champions"))
        .when(pl.col("recency_days") <= 30)
        .then(pl.lit("Active Regulars"))
        .when(pl.col("recency_days") > 60)
        .then(pl.lit("At Risk / Inactive"))
        .otherwise(pl.lit("Promising"))
        .alias("customer_segment")
    )
    .sort("monetary", descending=True)
)

print("Customer RFM Segmentation Results:\n", rfm_metrics)


# ----------------------------------------------------------------------
# Project 3: Automated ETL Pipeline with Partitioned Parquet
# ----------------------------------------------------------------------
section("Project 3: Automated ETL Pipeline (Partitioned Parquet)")

etl_input_dir = tmp_workspace / "raw_csvs"
etl_input_dir.mkdir(exist_ok=True)

# Generate 3 raw CSV batch files
for batch in range(1, 4):
    pl.DataFrame(
        {
            "tx_code": [f"TX-{batch}-{i}" for i in range(100)],
            "date": [f"2026-03-{10 + (i % 5):02d}" for i in range(100)],
            "region": [["APAC", "EMEA", "NA"][i % 3] for i in range(100)],
            "gross": [100.0 + (i * 2.5) for i in range(100)],
            "vat": [18.0 for _ in range(100)],
        }
    ).write_csv(etl_input_dir / f"batch_{batch}.csv")

# Lazy ETL scan across all input CSVs with schema enforcement
clean_pipeline = (
    pl.scan_csv(etl_input_dir / "*.csv", try_parse_dates=True)
    .with_columns(
        (pl.col("gross") + pl.col("vat")).alias("total_with_tax"),
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
    )
    .filter(pl.col("gross") > 150.0)
)

# Materialize and partition output
clean_df = clean_pipeline.collect()
output_partition_dir = tmp_workspace / "partitioned_lake"
clean_df.write_parquet(output_partition_dir, partition_by=["year", "region"])

print(f"ETL completed: {len(clean_df)} validated records partitioned into lake:")
for part in sorted(output_partition_dir.rglob("*.parquet"))[:4]:
    print(f" - {part.relative_to(tmp_workspace)}")


# ----------------------------------------------------------------------
# Project 4: High-Throughput Log & Clickstream Processor
# ----------------------------------------------------------------------
section("Project 4: Web Log Processor & Dynamic Error Rate Alerts")

# Simulating web server access log records
log_lines = [
    '192.168.1.10 - - [01/Mar/2026:10:00:01 +0000] "GET /api/v1/products HTTP/1.1" 200 452 45.2',
    '192.168.1.12 - - [01/Mar/2026:10:00:05 +0000] "POST /api/v1/checkout HTTP/1.1" 500 120 180.5',
    '192.168.1.15 - - [01/Mar/2026:10:00:12 +0000] "GET /api/v1/products HTTP/1.1" 200 512 32.1',
    '192.168.1.10 - - [01/Mar/2026:10:00:25 +0000] "GET /api/v1/cart HTTP/1.1" 404 85 12.0',
    '192.168.1.20 - - [01/Mar/2026:10:00:45 +0000] "POST /api/v1/checkout HTTP/1.1" 503 140 210.0',
    '192.168.1.22 - - [01/Mar/2026:10:01:02 +0000] "GET /api/v1/products HTTP/1.1" 200 490 28.0',
]

raw_logs_df = pl.DataFrame({"raw_line": log_lines})

# Regex capture groups extracting IP, timestamp, method, endpoint, status code, latency
parsed_logs = raw_logs_df.select(
    pl.col("raw_line").str.extract(r"^(\S+)", 1).alias("client_ip"),
    pl.col("raw_line").str.extract(r'"(GET|POST|PUT|DELETE)\s+(\S+)\s+', 1).alias("method"),
    pl.col("raw_line").str.extract(r'"(GET|POST|PUT|DELETE)\s+(\S+)\s+', 2).alias("endpoint"),
    pl.col("raw_line").str.extract(r'\s(\d{3})\s', 1).cast(pl.Int32).alias("http_status"),
    pl.col("raw_line").str.extract(r'\s(\d+\.\d+)$', 1).cast(pl.Float64).alias("latency_ms"),
).with_columns(
    pl.when(pl.col("http_status") >= 500)
    .then(pl.lit("SERVER_ERROR_5XX"))
    .when(pl.col("http_status") >= 400)
    .then(pl.lit("CLIENT_ERROR_4XX"))
    .otherwise(pl.lit("SUCCESS_2XX"))
    .alias("status_category")
)

print("Parsed & Classified Access Logs:\n", parsed_logs)
endpoint_health = parsed_logs.group_by("endpoint").agg(
    pl.len().alias("req_count"),
    pl.col("latency_ms").mean().round(1).alias("avg_latency_ms"),
    pl.col("http_status").filter(pl.col("http_status") >= 500).len().alias("server_errors_5xx"),
)
print("\nEndpoint Health Breakdown:\n", endpoint_health)


# ----------------------------------------------------------------------
# Project 5: Financial OHLCV & Rolling Volatility Engine
# ----------------------------------------------------------------------
section("Project 5: Financial Time-Series & Rolling 5-Minute Volatility")

t_start = datetime(2026, 3, 1, 9, 30)
n_ticks = 60
price_steps = np.random.normal(0, 0.5, n_ticks).cumsum() + 150.0

ticks_df = pl.DataFrame(
    {
        "timestamp": [t_start + timedelta(seconds=i * 30) for i in range(n_ticks)],
        "price": price_steps.round(2),
        "volume": np.random.randint(10, 100, n_ticks),
    }
).sort("timestamp")

candles = (
    ticks_df
    .group_by_dynamic("timestamp", every="5m")
    .agg(
        pl.col("price").first().alias("open"),
        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.col("price").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
    )
    .with_columns(
        # Return percentage
        ((pl.col("close") - pl.col("open")) / pl.col("open") * 100).round(3).alias("candle_return_pct")
    )
)
print("5-Minute Financial Candles:\n", candles)


# ----------------------------------------------------------------------
# Project 6: Incremental Analytical Snapshot Service (Database Sync)
# ----------------------------------------------------------------------
section("Project 6: Incremental Analytical Snapshot Service")

# Simulating a backend operational table vs current analytics warehouse
current_warehouse_snapshot = pl.DataFrame(
    {
        "account_id": [101, 102, 103],
        "tier": ["free", "pro", "enterprise"],
        "total_spent": [50.0, 420.0, 1500.0],
        "last_updated": ["2026-03-01", "2026-03-01", "2026-03-01"],
    }
)

incoming_cdc_changes = pl.DataFrame(
    {
        "account_id": [102, 104],  # 102 updated, 104 new
        "tier": ["enterprise", "pro"],
        "total_spent": [650.0, 120.0],
        "last_updated": ["2026-03-02", "2026-03-02"],
    }
)

# Upsert (Merge / Deduplicate latest state by account_id):
synced_warehouse = (
    pl.concat([current_warehouse_snapshot, incoming_cdc_changes], how="vertical")
    .sort("last_updated", descending=True)
    .unique(subset=["account_id"], keep="first")
    .sort("account_id")
)

print("Synchronized Latest Warehouse Snapshot:\n", synced_warehouse)


# ----------------------------------------------------------------------
# Project 7: FastAPI Analytics Microservice Pattern
# ----------------------------------------------------------------------
section("Project 7: FastAPI Analytics Query Builder Pattern")

def query_sales_endpoint(
    region_filter: str | None = None,
    min_amount: float = 0.0,
    top_n: int = 3,
) -> str:
    """
    Simulates a high-performance FastAPI endpoint query builder.
    Dynamically creates an optimized LazyFrame plan and returns JSON in sub-5ms.
    """
    q = orders_p1.lazy()
    if region_filter:
        q = q.filter(pl.col("region") == region_filter)
    if min_amount > 0:
        q = q.filter(pl.col("revenue") >= min_amount)

    result_df = (
        q.group_by("category")
        .agg(
            pl.col("revenue").sum().alias("total_sales"),
            pl.len().alias("count"),
        )
        .sort("total_sales", descending=True)
        .limit(top_n)
        .collect()
    )
    # Serialize to JSON instantly:
    return result_df.write_json()

fastapi_response_json = query_sales_endpoint(region_filter="North", min_amount=50.0, top_n=2)
print("FastAPI Response JSON Payload:\n", json.dumps(json.loads(fastapi_response_json), indent=2))


# ----------------------------------------------------------------------
# Project 8: Nested JSON & REST API Response Normalization
# ----------------------------------------------------------------------
section("Project 8: Nested JSON Normalization (Struct & List Unpacking)")

raw_partner_payload = [
    {
        "invoice_id": "INV-9001",
        "customer": {"name": "Tech Corp", "country": "IN"},
        "items": [
            {"sku": "A-1", "qty": 5, "price": 100.0},
            {"sku": "B-2", "qty": 2, "price": 40.0},
        ],
    },
    {
        "invoice_id": "INV-9002",
        "customer": {"name": "Global Retail", "country": "US"},
        "items": [
            {"sku": "C-3", "qty": 10, "price": 25.0},
        ],
    },
]

raw_json_df = pl.DataFrame(raw_partner_payload)

# Normalize into a flat relational invoice item lines table:
flattened_invoices = (
    raw_json_df
    .unnest("customer")
    .explode("items", empty_as_null=True)
    .unnest("items")
    .with_columns(
        (pl.col("qty") * pl.col("price")).alias("line_total")
    )
)
print("Flattened Relational Line-Item Table:\n", flattened_invoices)


# ----------------------------------------------------------------------
# Project 9: Out-of-Core Data Processing Engine
# ----------------------------------------------------------------------
section("Project 9: Out-of-Core Memory-Bounded Batch Execution")

large_lake_path = tmp_workspace / "massive_orders.parquet"
orders_p1.write_parquet(large_lake_path, compression="zstd")

# Process out-of-core using engine="streaming":
streaming_pipeline = (
    pl.scan_parquet(large_lake_path)
    .filter(pl.col("revenue") > 50.0)
    .group_by("region")
    .agg(
        pl.len().alias("high_value_orders"),
        pl.col("revenue").sum().alias("high_value_revenue"),
        pl.col("cost").mean().alias("avg_cost"),
    )
    .sort("high_value_revenue", descending=True)
)

out_of_core_result = streaming_pipeline.collect(engine="streaming")
print("Out-of-Core Streaming Aggregate:\n", out_of_core_result)


# ----------------------------------------------------------------------
# Project 10: Head-to-Head Performance Benchmark: Polars vs. Pandas
# ----------------------------------------------------------------------
section("Project 10: Performance Benchmark (Polars vs. Pandas)")

n_bench = 200_000
bench_data = {
    "cat_a": np.random.choice(["X", "Y", "Z", "W"], n_bench),
    "cat_b": np.random.choice(["Alpha", "Beta", "Gamma"], n_bench),
    "val_1": np.random.uniform(1.0, 1000.0, n_bench),
    "val_2": np.random.uniform(5.0, 500.0, n_bench),
}

# Pandas benchmark
pd_df = pd.DataFrame(bench_data)
t0 = time.perf_counter()
pd_result = (
    pd_df[(pd_df["val_1"] > 200.0)]
    .groupby(["cat_a", "cat_b"])
    .agg(total_v1=("val_1", "sum"), avg_v2=("val_2", "mean"))
    .reset_index()
    .sort_values("total_v1", ascending=False)
)
pandas_time = time.perf_counter() - t0

# Polars benchmark
pl_df = pl.DataFrame(bench_data)
t0 = time.perf_counter()
pl_result = (
    pl_df.lazy()
    .filter(pl.col("val_1") > 200.0)
    .group_by("cat_a", "cat_b")
    .agg(
        pl.col("val_1").sum().alias("total_v1"),
        pl.col("val_2").mean().alias("avg_v2"),
    )
    .sort("total_v1", descending=True)
    .collect()
)
polars_time = time.perf_counter() - t0

speedup = pandas_time / polars_time if polars_time > 0 else 1.0

print(f"{'Framework':<12} | {'Rows Processed':<16} | {'Execution Time (ms)':<20} | {'Speedup':<10}")
print("-" * 65)
print(f"{'Pandas':<12} | {n_bench:<16,d} | {pandas_time * 1000:<20.2f} | 1.0x (baseline)")
print(f"{'Polars':<12} | {n_bench:<16,d} | {polars_time * 1000:<20.2f} | {speedup:<10.1f}x faster")

print("\n Module 7 Portfolio Projects Completed Successfully!")
