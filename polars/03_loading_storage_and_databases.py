"""
Module 3: Data Ingestion, Storage & Database Integration
========================================================
Topics Covered:
- 3.1 Reading & Writing CSV Files (separator, schema overrides, null handling)
- 3.2 Reading & Writing Apache Parquet (Snappy & ZSTD compression, columnar efficiency)
- 3.3 Semi-Structured JSON and Newline-Delimited JSON (NDJSON)
- 3.4 Scanning Multiple Files using File Globs
- 3.5 Schema Inference Limits & Explicit Schema Overrides
- 3.6 Partitioned Parquet Datasets (partition_by)
- 3.7 Arrow IPC / Feather Serialization (Instant zero-copy disk caching)
- 3.8 Database Integration Patterns (PostgreSQL & SQLite via read_database_uri)
- 3.9 Storage Format Benchmark: CSV vs Parquet vs IPC (Disk size & load speed)
"""

import sys
import tempfile
import time
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


# Create a persistent temporary directory for demo files
tmp_dir = Path(tempfile.mkdtemp(prefix="polars_module3_"))


# ----------------------------------------------------------------------
# 3.1 Reading & Writing CSV Files
# ----------------------------------------------------------------------
section("3.1 Reading & Writing CSV Files")

sample_sales = pl.DataFrame(
    {
        "transaction_id": [f"TX-{i:04d}" for i in range(1, 6)],
        "store_id": ["STORE-A", "STORE-B", "STORE-A", "STORE-C", "STORE-B"],
        "amount": [120.50, 450.00, 35.75, 890.20, 210.00],
        "customer_rating": [5, 4, 3, 5, 4],
        "date": ["2026-03-01", "2026-03-01", "2026-03-02", "2026-03-02", "2026-03-03"],
    }
)

csv_path = tmp_dir / "sales.csv"
sample_sales.write_csv(csv_path)
print(f"Exported CSV to: {csv_path.name} ({csv_path.stat().st_size} bytes)")

# Reading CSV with schema override & date parsing
read_sales_csv = pl.read_csv(
    csv_path,
    schema_overrides={"customer_rating": pl.Int8, "amount": pl.Float64},
    try_parse_dates=True,
)
print("Read CSV schema:\n", read_sales_csv.schema)
print("Read CSV preview:\n", read_sales_csv.head(3))


# ----------------------------------------------------------------------
# 3.2 Reading & Writing Apache Parquet
# ----------------------------------------------------------------------
section("3.2 Reading & Writing Apache Parquet (ZSTD & Snappy)")

# Parquet is the gold standard for data engineering:
# - Columnar storage (reads only requested columns)
# - Embedded type metadata and dictionary encoding
# - Heavy compression (often 80-90% smaller than CSV)
parquet_path = tmp_dir / "sales.parquet"
sample_sales.write_parquet(parquet_path, compression="zstd", compression_level=3)
print(f"Exported Parquet to: {parquet_path.name} ({parquet_path.stat().st_size} bytes)")

read_parquet_df = pl.read_parquet(parquet_path, columns=["transaction_id", "amount"])
print("Read Parquet (selective columns projection):\n", read_parquet_df.head(3))


# ----------------------------------------------------------------------
# 3.3 Semi-Structured JSON & Newline-Delimited JSON (NDJSON)
# ----------------------------------------------------------------------
section("3.3 Semi-Structured JSON & Newline-Delimited JSON (NDJSON)")

# NDJSON (1 JSON object per line) is ideal for log streaming and microservices:
ndjson_path = tmp_dir / "events.ndjson"
sample_sales.write_ndjson(ndjson_path)
print(f"Exported NDJSON to: {ndjson_path.name}")

read_ndjson_df = pl.read_ndjson(ndjson_path)
print("Read NDJSON preview:\n", read_ndjson_df.head(2))

# Standard JSON:
json_path = tmp_dir / "events.json"
sample_sales.write_json(json_path)
read_json_df = pl.read_json(json_path)
print(f"Read standard JSON shape: {read_json_df.shape}")


# ----------------------------------------------------------------------
# 3.4 Multi-File Ingestion using File Globs
# ----------------------------------------------------------------------
section("3.4 Multi-File Ingestion using File Globs (*.parquet)")

# Create partitioned mock log files:
logs_dir = tmp_dir / "partitioned_logs"
logs_dir.mkdir(exist_ok=True)

for day in [1, 2, 3]:
    day_df = pl.DataFrame(
        {
            "log_id": [f"L_{day}_{i}" for i in range(100)],
            "day": [day] * 100,
            "response_time_ms": [20.0 + (i % 15) for i in range(100)],
        }
    )
    day_df.write_parquet(logs_dir / f"day_{day}.parquet")

# Ingest all days simultaneously using glob:
all_logs = pl.read_parquet(logs_dir / "*.parquet")
print(f"Loaded {len(all_logs)} rows across all globbed day_*.parquet files!")
print(all_logs.group_by("day").agg(pl.len().alias("record_count"), pl.col("response_time_ms").mean().round(2).alias("avg_latency")))


# ----------------------------------------------------------------------
# 3.5 Arrow IPC / Feather Serialization (Instant Disk Cache)
# ----------------------------------------------------------------------
section("3.5 Arrow IPC / Feather Serialization (Zero-Copy Cache)")

# Arrow IPC is the exact in-memory format serialized to disk.
# Reading an IPC file requires virtually zero deserialization CPU overhead.
ipc_path = tmp_dir / "sales_cache.arrow"
sample_sales.write_ipc(ipc_path)

cached_df = pl.read_ipc(ipc_path)
print(f"Loaded Arrow IPC file in instantaneous zero-copy: {cached_df.shape}")


# ----------------------------------------------------------------------
# 3.6 Partitioned Parquet Datasets
# ----------------------------------------------------------------------
section("3.6 Partitioned Parquet Datasets (Hive Partitioning)")

# High-volume tables are partitioned by columns (e.g. region, year, category)
partition_dir = tmp_dir / "hive_sales"
sample_sales.write_parquet(partition_dir, partition_by=["store_id"])

print("Hive-partitioned directory structure:")
for p in sorted(partition_dir.rglob("*.parquet")):
    print(f" - {p.relative_to(tmp_dir)}")

# Read all partitions seamlessly:
reloaded_partitioned = pl.read_parquet(partition_dir / "**/*.parquet")
print(f"Reloaded from partitioned directory: {reloaded_partitioned.shape}")


# ----------------------------------------------------------------------
# 3.7 Database Integration Pattern (PostgreSQL / SQLite)
# ----------------------------------------------------------------------
section("3.7 Database Integration Pattern (ConnectorX / ADBC)")

# Polars provides `read_database` for DBAPI connections (sqlite3, psycopg2, asyncpg)
# and `read_database_uri` for ultra-fast C/Rust database extraction with ConnectorX / ADBC:
#
# Production PostgreSQL example:
# >>> df = pl.read_database_uri(
# ...     query="SELECT order_id, amount, status FROM orders WHERE created_at >= '2026-01-01'",
# ...     uri="postgresql://user:pass@localhost:5432/analytics_db",
# ...     engine="connectorx"  # 10x-20x faster than standard psycopg2 / pandas
# ... )

# Demonstrating with Python's built-in sqlite3 DBAPI connection:
import sqlite3

sqlite_db_path = tmp_dir / "app.db"
conn = sqlite3.connect(sqlite_db_path)
conn.execute("CREATE TABLE metrics (service_name TEXT, cpu_pct REAL, mem_pct REAL)")
conn.executemany(
    "INSERT INTO metrics VALUES (?, ?, ?)",
    [("auth-service", 12.5, 45.0), ("payment-service", 34.0, 68.2), ("search-api", 55.4, 82.1)],
)
conn.commit()

# read_database works with standard Python sqlite3 connection:
db_df = pl.read_database(
    query="SELECT * FROM metrics WHERE cpu_pct > 20.0",
    connection=conn,
)
conn.close()
print("Polars extracted directly from Database via DBAPI connection:\n", db_df)


# ----------------------------------------------------------------------
# 3.8 Storage Benchmark: CSV vs Parquet vs IPC
# ----------------------------------------------------------------------
section("3.8 Benchmark: CSV vs Parquet vs IPC (100k Rows)")

# Generate 100,000 synthetic records
n_rows = 100_000
bench_df = pl.DataFrame(
    {
        "id": list(range(n_rows)),
        "sensor_id": [f"SENS_{i % 50}" for i in range(n_rows)],
        "value_a": [1.5 * (i % 100) for i in range(n_rows)],
        "value_b": [2.5 * (i % 200) for i in range(n_rows)],
        "flag": [(i % 2 == 0) for i in range(n_rows)],
    }
)

bench_csv = tmp_dir / "bench.csv"
bench_parquet = tmp_dir / "bench.parquet"
bench_ipc = tmp_dir / "bench.arrow"

# Write & Time CSV
t0 = time.perf_counter()
bench_df.write_csv(bench_csv)
csv_write_time = time.perf_counter() - t0

# Write & Time Parquet
t0 = time.perf_counter()
bench_df.write_parquet(bench_parquet, compression="zstd")
parquet_write_time = time.perf_counter() - t0

# Write & Time IPC
t0 = time.perf_counter()
bench_df.write_ipc(bench_ipc)
ipc_write_time = time.perf_counter() - t0

# Read Timings
t0 = time.perf_counter()
_ = pl.read_csv(bench_csv)
csv_read_time = time.perf_counter() - t0

t0 = time.perf_counter()
_ = pl.read_parquet(bench_parquet)
parquet_read_time = time.perf_counter() - t0

t0 = time.perf_counter()
_ = pl.read_ipc(bench_ipc)
ipc_read_time = time.perf_counter() - t0

print(f"{'Format':<10} | {'Disk Size (KB)':<15} | {'Write Time (ms)':<16} | {'Read Time (ms)':<16}")
print("-" * 65)
print(f"{'CSV':<10} | {bench_csv.stat().st_size / 1024:<15.1f} | {csv_write_time * 1000:<16.1f} | {csv_read_time * 1000:<16.1f}")
print(f"{'Parquet':<10} | {bench_parquet.stat().st_size / 1024:<15.1f} | {parquet_write_time * 1000:<16.1f} | {parquet_read_time * 1000:<16.1f}")
print(f"{'Arrow IPC':<10} | {bench_ipc.stat().st_size / 1024:<15.1f} | {ipc_write_time * 1000:<16.1f} | {ipc_read_time * 1000:<16.1f}")

print("\n Module 3 Completed Successfully!")
