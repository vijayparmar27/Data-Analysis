"""
Module 2: Data Loading & Inspection
==================================
Topics Covered:
- 2.1 Reading CSV files (delimiters, headers, parse_dates, na_values, usecols)
- 2.2 Reading and writing Excel workbooks (multi-sheet with openpyxl)
- 2.3 Reading & parsing JSON, and normalizing nested API payloads with pd.json_normalize()
- 2.4 Loading from PostgreSQL / SQL (pd.read_sql, SQLAlchemy connection pattern)
- 2.5 DataFrame exploration (head, tail, sample with reproducibility)
- 2.6 Structural diagnostics (info, describe, value_counts with normalization & binning)
- 2.7 Data types & memory footprint audit (memory_usage(deep=True))
- 2.8 Exporting DataFrames to CSV, Excel, JSON, and Parquet
- 2.9 Hands-on Practice Challenge with complete solution!
"""

import io
import os
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
# 2.1 Reading CSV Files
# ----------------------------------------------------------------------
section("2.1 Reading CSV Files with Advanced Options")

# In production, CSVs often have custom separators, messy date formats, and missing value sentinels.
# We create an in-memory CSV buffer to demonstrate read_csv options cleanly:
csv_data = """transaction_id,timestamp,customer_id,amount,status,notes
tx_1001,2026-03-01 10:15:00,cust_99,145.50,COMPLETED,First purchase
tx_1002,2026-03-01 11:00:25,cust_42,89.00,PENDING,
tx_1003,2026-03-02 09:30:10,cust_99,350.00,COMPLETED,Bulk order
tx_1004,2026-03-02 14:45:00,cust_18,-999.00,FAILED,Payment gateway error
tx_1005,2026-03-03 16:20:00,cust_33,52.20,COMPLETED,NA
"""

# pd.read_csv parameters:
# - parse_dates: Automatically converts string dates to Timestamp
# - na_values: Treat sentinel values like -999.00 or "NA" as NaN
# - usecols: Read only specific columns to save memory
df_csv = pd.read_csv(
    io.StringIO(csv_data),
    parse_dates=["timestamp"],
    na_values={"amount": [-999.00], "notes": ["NA", ""]},
    usecols=["transaction_id", "timestamp", "customer_id", "amount", "status"]
)

print("Ingested CSV with Parsed Dates & Custom NA Sentinel (-999.00 -> NaN):\n", df_csv)
print("\nColumn Data Types:\n", df_csv.dtypes)


# ----------------------------------------------------------------------
# 2.2 Reading and Parsing JSON & Nested API Responses
# ----------------------------------------------------------------------
section("2.2 Reading JSON & Normalizing Nested API Responses")

# As a full-stack / backend developer, most APIs (FastAPI / Express) return nested JSON.
# pd.json_normalize flattens nested JSON dictionaries into tabular columns.
api_response = [
    {
        "order_id": "ORD-501",
        "created_at": "2026-03-01",
        "customer": {"name": "Aarav Sharma", "tier": "Gold", "city": "Mumbai"},
        "payment": {"method": "UPI", "amount": 1250.0, "status": "success"},
        "items_count": 3
    },
    {
        "order_id": "ORD-502",
        "created_at": "2026-03-02",
        "customer": {"name": "Priya Patel", "tier": "Silver", "city": "Ahmedabad"},
        "payment": {"method": "Credit Card", "amount": 2800.0, "status": "success"},
        "items_count": 5
    },
    {
        "order_id": "ORD-503",
        "created_at": "2026-03-03",
        "customer": {"name": "Rohan Mehta", "tier": "Gold", "city": "Bengaluru"},
        "payment": {"method": "NetBanking", "amount": 640.0, "status": "failed"},
        "items_count": 1
    }
]

# Flatten nested JSON into DataFrame with dotted column notation:
df_nested_api = pd.json_normalize(api_response)
print("Flattened Nested API Response via pd.json_normalize():\n", df_nested_api)
print("\nGenerated Column Headers:\n", list(df_nested_api.columns))


# ----------------------------------------------------------------------
# 2.3 Loading Data from PostgreSQL / SQL Databases
# ----------------------------------------------------------------------
section("2.3 Loading Data from PostgreSQL / SQL (The Backend Workflow)")

print("""
Typical Backend Integration Pattern:
-------------------------------------
from sqlalchemy import create_engine
import pandas as pd

# Connect via SQLAlchemy engine (compatible with PostgreSQL, MySQL, SQLite, etc.)
engine = create_engine("postgresql+psycopg2://user:password@localhost:5432/ecommerce_db")

# Run parameterized analytical query directly into a DataFrame:
query = \"\"\"
    SELECT 
        o.id AS order_id,
        o.customer_id,
        o.created_at,
        SUM(oi.quantity * oi.unit_price) AS order_total,
        o.status
    FROM orders o
    JOIN order_items oi ON o.id = oi.order_id
    WHERE o.created_at >= '2026-01-01'
    GROUP BY o.id, o.customer_id, o.created_at, o.status;
\"\"\"

df_sql = pd.read_sql(query, con=engine, parse_dates=["created_at"])
""")

# Simulation using SQLite (built into Python standard library):
import sqlite3

conn = sqlite3.connect(":memory:")
# Insert mock SQL data
conn.execute("CREATE TABLE products (sku TEXT PRIMARY KEY, title TEXT, price REAL, stock INT)")
conn.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", [
    ("SKU-1", "Bluetooth Speaker", 49.99, 120),
    ("SKU-2", "Wireless Mouse", 19.99, 340),
    ("SKU-3", "USB-C Hub", 34.50, 75)
])
conn.commit()

# Read directly via pd.read_sql
df_sql_mock = pd.read_sql("SELECT * FROM products WHERE price > 20.0 ORDER BY stock DESC", con=conn)
print("Query result executed via pd.read_sql from SQL database:\n", df_sql_mock)
conn.close()


# ----------------------------------------------------------------------
# 2.4 Quick Exploration: head(), tail(), sample()
# ----------------------------------------------------------------------
section("2.4 Quick Exploration: head(), tail(), sample()")

df = df_nested_api.copy()

print("df.head(2) - First 2 rows:\n", df.head(2))
print("\ndf.tail(1) - Last row:\n", df.tail(1))
# sample() is ideal for inspecting random records without order bias:
print("\ndf.sample(2, random_state=42) - Random 2 rows:\n", df.sample(2, random_state=42))


# ----------------------------------------------------------------------
# 2.5 Structural Diagnostics: info(), describe(), value_counts()
# ----------------------------------------------------------------------
section("2.5 Structural Diagnostics: info(), describe(), value_counts()")

print("--- DataFrame .info() summary ---")
df.info()

print("\n--- Descriptive Statistics .describe() (Numeric columns) ---")
print(df.describe())

print("\n--- Value Counts on Categorical column ('customer.tier') ---")
# Count raw occurrences:
print(df["customer.tier"].value_counts())
# Frequency distribution (percentages / proportions):
print("\nNormalized Frequency (%):\n", df["customer.tier"].value_counts(normalize=True) * 100)


# ----------------------------------------------------------------------
# 2.6 Data Types & Memory Usage Audit
# ----------------------------------------------------------------------
section("2.6 Data Types & Deep Memory Footprint Diagnostics")

# Checking memory consumption of each column:
# By default, df.memory_usage() only counts pointer sizes for Python strings (object dtype).
# Specifying deep=True inspects the actual string lengths inside heap memory!
memory_report = df.memory_usage(deep=True)
print("Deep Memory Usage per Column (bytes):\n", memory_report)
print(f"Total DataFrame Memory: {memory_report.sum()} bytes")


# ----------------------------------------------------------------------
# 2.7 Exporting DataFrames (CSV, Excel, JSON, Parquet)
# ----------------------------------------------------------------------
section("2.7 Exporting DataFrames to Disk / Formats")

# Use __file__ when executed as a script, or fallback to current directory in notebooks/REPLs:
base_dir = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
output_dir = os.path.join(base_dir, "scratch")
os.makedirs(output_dir, exist_ok=True)

# 1. Export to CSV: index=False avoids writing redundant auto-increment row numbers
csv_path = os.path.join(output_dir, "export_orders.csv")
df.to_csv(csv_path, index=False)
print(f"Exported clean CSV to: {csv_path}")

# 2. Export to JSON (orient='records' produces a standard API JSON array of objects)
json_path = os.path.join(output_dir, "export_orders.json")
df.to_json(json_path, orient="records", indent=2)
print(f"Exported JSON (records orientation) to: {json_path}")

# 3. Export to Parquet (High-performance columnar storage with compression)
try:
    parquet_path = os.path.join(output_dir, "export_orders.parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"Exported compressed Parquet to: {parquet_path}")
except ImportError:
    print("ℹ️ pyarrow not yet installed. Run 'pip install pyarrow' to enable Parquet exports.")


# ----------------------------------------------------------------------
# 2.8 Hands-on Practice Challenge
# ----------------------------------------------------------------------
section("2.8 Hands-on Practice Challenge")
print("""
PRACTICE EXERCISE:
1. Ingest the following raw string CSV representing API latency logs:
   raw_logs = \"\"\"log_id,service,endpoint,status_code,latency_ms,timestamp
   101,auth,/api/login,200,45.2,2026-03-01T12:00:00Z
   102,billing,/api/charge,200,210.8,2026-03-01T12:00:01Z
   103,billing,/api/charge,500,1540.0,2026-03-01T12:00:02Z
   104,auth,/api/login,401,38.1,2026-03-01T12:00:03Z
   105,catalog,/api/products,200,62.4,2026-03-01T12:00:04Z
   106,billing,/api/charge,200,195.0,2026-03-01T12:00:05Z
   \"\"\"
2. Parse 'timestamp' as datetime and read only: ['service', 'status_code', 'latency_ms'].
3. Calculate the average latency for each 'service'.
4. Get the value counts of 'status_code' as percentages.
""")

# --- Challenge Solution ---
raw_logs = """log_id,service,endpoint,status_code,latency_ms,timestamp
101,auth,/api/login,200,45.2,2026-03-01T12:00:00Z
102,billing,/api/charge,200,210.8,2026-03-01T12:00:01Z
103,billing,/api/charge,500,1540.0,2026-03-01T12:00:02Z
104,auth,/api/login,401,38.1,2026-03-01T12:00:03Z
105,catalog,/api/products,200,62.4,2026-03-01T12:00:04Z
106,billing,/api/charge,200,195.0,2026-03-01T12:00:05Z
"""

logs_df = pd.read_csv(
    io.StringIO(raw_logs),
    usecols=["service", "status_code", "latency_ms"]
)

# Step 3: Mean latency per service
avg_latency_by_service = logs_df.groupby("service")["latency_ms"].mean()
print("Average Latency by Microservice (ms):\n", avg_latency_by_service)

# Step 4: Status code frequency percentage
status_distribution = logs_df["status_code"].value_counts(normalize=True) * 100
print("\nHTTP Status Code Distribution (%):\n", status_distribution)
print("\n Module 2 Completed Successfully!")
