# Polars Learning Roadmap (Beginner to Advanced)

> **Context**: Tailored for a Full-Stack & Backend Developer (Python, NumPy, Pandas, PostgreSQL, FastAPI, TypeScript/Node.js). We focus on high-throughput data processing, Apache Arrow columnar memory, the Polars Expression Engine, Lazy evaluation, query optimization (predicate & projection pushdown), out-of-core streaming, and building high-performance backend analytics services.

---

## 📊 Learning Progress: `0 / 77 Topics Completed`

---

## Module 1: Polars Fundamentals & Architecture (Beginner)

_Goal: Understand the Apache Arrow columnar memory model, Rust-backed multithreaded core, Series and DataFrame structures, and eager execution basics._

- [ ] **1.1** What is Polars and why use it? (Rust core, Apache Arrow columnar memory, parallel multithreading, zero-copy architecture)
- [ ] **1.2** Polars vs. Pandas vs. NumPy: Architectural differences, memory overhead, GIL bypass, and performance paradigms
- [ ] **1.3** Environment setup & package management with `uv` (`uv add polars pyarrow connectorx adbc-driver-postgresql`)
- [ ] **1.4** Importing conventions and namespace standards (`import polars as pl`)
- [ ] **1.5** Core data structures: `pl.Series` (1D homogenous Arrow chunked array) vs. `pl.DataFrame` (2D columnar table)
- [ ] **1.6** DataFrame creation from dictionaries, lists of tuples, Python dataclasses/dicts, and NumPy 2D arrays
- [ ] **1.7** DataFrame inspection attributes: `.shape`, `.columns`, `.schema`, `.dtypes`, `.estimated_size()`, and `.flags`
- [ ] **1.8** Polars Data Types (`pl.DataType`): Integer (`Int8`..`Int64`), Float (`Float32`, `Float64`), `String`, `Boolean`, `Date`, `Datetime`, `Duration`, `Categorical`, and `Enum`
- [ ] **1.9** Basic DataFrame inspection & printing: `.head()`, `.tail()`, `.sample()`, `.glimpse()`, and Polars table formatting configuration (`pl.Config`)
- [ ] **1.10** Eager execution fundamentals: In-memory evaluation, `.to_pandas()`, `.to_numpy()`, and Arrow zero-copy exchange (`.__arrow_c_stream__()`)

---

## Module 2: The Expression Engine & Data Transformation (Beginner → Intermediate)

_Goal: Master Polars' core superpower—the columnar Expression Engine (`pl.col()`), parallel column evaluation, filtering, and conditional logic._

- [ ] **2.1** The Polars Expression philosophy: Composable, embarrassingly parallel, and side-effect free expressions
- [ ] **2.2** Selecting columns with `select()` vs generating/modifying columns with `with_columns()`
- [ ] **2.3** Column references with `pl.col()`: Single column, multi-column (`pl.col("a", "b")`), wildcards (`pl.col("*")`), and regex selectors (`pl.col("^sales_.*$")`)
- [ ] **2.4** Selector expressions with `polars.selectors` (`cs.numeric()`, `cs.string()`, `cs.temporal()`, `cs.matches()`)
- [ ] **2.5** Row filtering with `filter()`: Boolean expressions, logical operators (`&`, `|`, `~`), and null-safe comparisons
- [ ] **2.6** Row slicing and indexing: `.slice(offset, length)`, `.top_k()`, `.bottom_k()`, and why integer indexing (`df[row, col]`) is an anti-pattern in Polars
- [ ] **2.7** Sorting operations: `sort()`, multi-column sorting, null positioning (`nulls_last=True`), and descending orders
- [ ] **2.8** Column management: Renaming (`rename()`), dropping (`drop()`), and casting (`cast()`, `strict=False`)
- [ ] **2.9** Unique values and frequencies: `.unique()`, `.n_unique()`, and `.value_counts(sort=True, parallel=True)`
- [ ] **2.10** Handling missing values: Detection (`is_null()`, `is_not_null()`, `is_nan()`), fill strategies (`fill_null()`, `fill_nan()`, forward/backward fill), and `drop_nulls()`
- [ ] **2.11** High-performance string manipulation with `.str` accessor: `.to_lowercase()`, `.slice()`, `.contains()`, `.replace()`, `.extract()`, and regex capture groups
- [ ] **2.12** Branching & conditional expressions: `pl.when(...).then(...).when(...).then(...).otherwise(...)` (SQL `CASE WHEN` equivalent)

---

## Module 3: Data Ingestion, Storage & Database Integration (Intermediate)

_Goal: Ingest and persist data across CSV, Parquet, JSON, and PostgreSQL with optimal memory usage, schema control, and partitioning._

- [ ] **3.1** Ingesting CSV files: `pl.read_csv()` (separator, schema overrides, null values, chunked parsing, and date parsing)
- [ ] **3.2** Writing CSV files: `pl.DataFrame.write_csv()` (delimiter, date formats, float precision)
- [ ] **3.3** Parquet format mastery: `pl.read_parquet()`, `pl.DataFrame.write_parquet()` (compression codecs: Snappy, ZSTD, dictionary encoding)
- [ ] **3.4** Semi-structured data: Reading & writing standard JSON (`read_json()`) and newline-delimited JSON / NDJSON (`read_ndjson()`, `scan_ndjson()`)
- [ ] **3.5** Excel integration: Reading Excel workbooks (`read_excel()`) and multi-sheet export with `xlsx2csv` / `calamine` engines
- [ ] **3.6** High-speed PostgreSQL integration: Loading via `pl.read_database_uri()` powered by ConnectorX and Apache Arrow ADBC
- [ ] **3.7** Writing DataFrames back to PostgreSQL: `write_database()` with SQLAlchemy and ADBC engines
- [ ] **3.8** Multi-file ingestion: Scanning directory globs (`pl.scan_csv("data/*.csv")`, `pl.scan_parquet("data/**/*.parquet")`)
- [ ] **3.9** Schema management: Inference limits (`infer_schema_length`), explicit schema definition (`schema={...}`), and resolving schema drift
- [ ] **3.10** High-volume file partitioning: Exporting partitioned datasets (`write_parquet(..., partition_by=["year", "region"])`)
- [ ] **3.11** IPC / Arrow Feather serialization: Blazing-fast disk caching using `pl.read_ipc()` and `pl.DataFrame.write_ipc()`

---

## Module 4: Aggregation, Window Functions & Relational Joins (Intermediate)

_Goal: Perform high-throughput business analytics, group-level aggregations, rolling calculations, and relational operations._

- [ ] **4.1** Core aggregation expressions: `sum()`, `mean()`, `median()`, `min()`, `max()`, `std()`, `quantile()`, and `pl.len()`
- [ ] **4.2** Split-Apply-Combine with `group_by()`: Single and multi-column grouping, eager vs. parallel execution
- [ ] **4.3** Advanced `.agg()` expressions: Running multiple independent expressions simultaneously inside one aggregation pass
- [ ] **4.4** Filtered aggregations: Combining expressions with `.filter()` inside aggregation (e.g. `pl.col("rev").filter(pl.col("active")).sum()`)
- [ ] **4.5** Relational joins: `join()` types (inner, left, full, cross) with multiple join keys and duplicate suffix handling
- [ ] **4.6** Filtering joins: Semi-joins (`how="semi"`) and Anti-joins (`how="anti"`) for set-membership operations
- [ ] **4.7** Table concatenation: `pl.concat()` along vertical axes (rows), horizontal axes (columns), and diagonal schema alignment
- [ ] **4.8** Reshaping tabular layouts: Long to Wide (`pivot()`) and Wide to Long (`unpivot()` / `melt()`)
- [ ] **4.9** Window functions with `.over()`: Computing group metrics without collapsing rows (SQL `OVER (PARTITION BY ...)` equivalent)
- [ ] **4.10** Ranking & cumulative operations: `rank()`, `dense_rank()`, `cum_sum()`, `cum_count()`, `shift()`, and percentage change
- [ ] **4.11** Statistical summaries & relationships: Covariance (`cov()`), Pearson / Spearman correlation (`corr()`), and descriptive statistics (`describe()`)

---

## Module 5: The Lazy API, Query Optimizer & Streaming (Advanced)

_Goal: Unlock Polars' true performance via LazyFrames, query plan inspection, predicate/projection pushdown, and out-of-core streaming execution._

- [ ] **5.1** LazyFrame architecture: `pl.LazyFrame`, transitioning eager to lazy (`df.lazy()`), and deferred execution
- [ ] **5.2** Lazy scanning vs. eager loading: `pl.scan_csv()`, `pl.scan_parquet()`, `pl.scan_ndjson()`, and `pl.scan_ipc()`
- [ ] **5.3** Materializing queries: `.collect()` and schema validation without execution via `.collect_schema()`
- [ ] **5.4** Inspecting query plans: Unoptimized AST (`df.explain(optimized=False)`) vs. Optimized plan (`df.explain(optimized=True)`)
- [ ] **5.5** Predicate Pushdown: How Polars moves filters (`filter()`) down to the storage scan layer to avoid reading unnecessary rows
- [ ] **5.6** Projection Pushdown: How Polars inspects downstream needs and only reads required columns from disk
- [ ] **5.7** Common Subexpression Elimination (CSE): Identifying duplicate sub-trees and calculating intermediate expressions once
- [ ] **5.8** Type coercion & slice pushdown: Downcasting types at parse time and pushing `.limit()` / `.head()` into file scanners
- [ ] **5.9** Out-of-Core Streaming execution: Processing datasets larger than RAM via `collect(streaming=True)` / the new streaming engine
- [ ] **5.10** Query profiling & bottleneck detection: Visualizing execution timelines with `df.profile()`
- [ ] **5.11** Memory optimization techniques: Chunk sizes, Arrow memory pools, and preventing out-of-memory (OOM) crashes
- [ ] **5.12** Lazy pipeline construction: Building modular, reusable, testable analytics pipelines without computing intermediate states

---

## Module 6: Advanced Polars, Time-Series & High Performance (Advanced)

_Goal: Master complex temporal analysis, list/struct nested data types, Rust-backed UDFs, Arrow interoperability, and parallel tuning._

- [ ] **6.1** Temporal data types & parsing: `Date`, `Datetime`, `Duration`, timezone awareness, and string-to-date conversion (`str.to_datetime()`)
- [ ] **6.2** Date & time operations via `.dt` accessor: Year, month, day, weekday, quarter, epoch time, and date offsets
- [ ] **6.3** Time-series resampling & dynamic grouping: `group_by_dynamic()` (tumbling, hopping, overlapping time windows)
- [ ] **6.4** Rolling time windows: `group_by_rolling()` (backward, forward, and centered rolling business metrics)
- [ ] **6.5** Advanced list expressions: `pl.List` data type, list lengths (`list.len()`), element slicing (`list.get()`), and exploding (`explode()`)
- [ ] **6.6** Hierarchical struct types: `pl.Struct` data type, packing columns into structs, unpack operations, and nested JSON ingestion
- [ ] **6.7** User-Defined Functions (UDFs): When and how to use `map_elements()` vs. `map_batches()`, and performance caveats
- [ ] **6.8** Writing high-performance custom Rust plugins: Introduction to `polars-plugin` for zero-overhead extensions
- [ ] **6.9** Interoperability with NumPy, PyArrow, and Pandas: Zero-copy conversions, PyCapsule protocol, and Arrow Table sharing
- [ ] **6.10** Threading & parallel configuration: `POLARS_MAX_THREADS`, thread pools, rayon work-stealing scheduler
- [ ] **6.11** Incremental & batch processing pipelines: Micro-batch ingestion and appending to existing Parquet partitions

---

## Module 7: Real-World Portfolio Projects (Hands-On / Production Grade)

_Goal: Synthesize all Polars concepts into production-grade data pipelines, FastAPI analytics microservices, and large-scale data lake processing._

- [ ] **Project 1: Multi-Gigabyte E-Commerce Sales Analytics (Beginner → Intermediate)**
  - Ingest 10M+ transaction records lazily from compressed Parquet files.
  - Compute revenue, gross margin, Average Order Value (AOV), and customer return rates.
  - Implement window functions (`.over()`) to rank top products per region and calculate category market shares.
  - Export executive KPI summary reports to partitioned Parquet files.

- [ ] **Project 2: Customer Behavioral RFM Segmentation Engine (Intermediate)**
  - Ingest user interaction and purchase logs from PostgreSQL via ConnectorX / ADBC.
  - Calculate Recency, Frequency, and Monetary (RFM) metrics using optimized Polars expressions.
  - Assign percentile quintiles using `.qcut()` / `rank()` expressions without Python loops.
  - Segment users into Champions, Loyalists, At-Risk, and Inactive cohorts, exporting results to analytical database tables.

- [ ] **Project 3: Production High-Performance ETL Pipeline (Intermediate)**
  - Extract dirty transaction logs across multiple directories using file globbing (`pl.scan_csv()`).
  - Implement idempotent data cleansing: handle corrupted IDs, impute nulls, normalize date formats, and filter outliers.
  - Enforce strict schema validation using `.collect_schema()`.
  - Write clean, validated outputs into partitioned Parquet data lakes organized by `year/month/region`.

- [ ] **Project 4: Large-Scale Application Log & Clickstream Processor (Advanced)**
  - Process 50GB+ web server access logs (NDJSON / CSV) using streaming execution (`collect(streaming=True)`).
  - Extract HTTP status codes, request paths, response times, and IP addresses via regex string expressions.
  - Calculate rolling error rates (5xx / 4xx) per minute and identify DDoS or bot traffic spikes using `group_by_dynamic()`.
  - Generate automated alerting datasets for infrastructure monitoring.

- [ ] **Project 5: Financial Time-Series & Multi-Asset Rolling Volatility (Advanced)**
  - Ingest tick-level and minute-level OHLCV financial market data.
  - Resample into multiple timeframes (5m, 1h, 1d) using `group_by_dynamic()`.
  - Calculate rolling 30-day volatility, Exponentially Weighted Moving Averages (EWMA), Relative Strength Index (RSI), and drawdowns.
  - Handle market closing gaps and missing timestamps cleanly with `.upsample()` and `fill_null()`.

- [ ] **Project 6: PostgreSQL to Polars Data Lake Ingestion & Sync Service (Backend Integration)**
  - Connect to a production PostgreSQL database instance using Arrow ADBC / ConnectorX.
  - Read millions of transactional records in parallel chunks without taxing database memory.
  - Perform incremental ETL: identify changed records via timestamps, compute aggregations, and upsert analytical snapshots.
  - Benchmark query throughput against standard SQLAlchemy / Pandas approaches.

- [ ] **Project 7: FastAPI Analytics Microservice Powered by Polars (Full-Stack Backend Integration)**
  - Build a high-throughput REST API using FastAPI that delegates heavy analytics queries to Polars.
  - Expose parameterized endpoints for date-range filtering, customer cohorts, and sales breakdowns.
  - Use Polars LazyFrames to dynamically build query plans based on request query parameters.
  - Return aggregated JSON responses in sub-10ms latency using Polars' fast `.write_json()` serializer.

- [ ] **Project 8: Nested JSON & REST API Response Normalization (Advanced)**
  - Ingest complex, multi-level nested JSON responses from external third-party partner APIs.
  - Unpack `pl.Struct` and `pl.List` columns into flat relational tables using `.struct.field()` and `.explode()`.
  - Deduplicate and normalize customer profiles, billing addresses, and order item lines into normalized tables.

- [ ] **Project 9: Out-of-Core Data Processing Engine (100GB+ on 16GB RAM) (Performance Engineering)**
  - Generate a synthetic 100GB transaction dataset.
  - Build an end-to-end data pipeline using `scan_parquet()` and `collect(streaming=True)`.
  - Execute multi-table joins, grouped aggregations, and window calculations without triggering OS memory thrashing or OOM errors.
  - Monitor memory footprints and thread utilization across execution phases.

- [ ] **Project 10: Head-to-Head Performance Benchmark: Polars vs. Pandas vs. DuckDB (Benchmarking)**
  - Construct standardized benchmark scenarios: CSV parsing speed, Parquet read/write speed, multi-column group-by, large joins, and memory usage.
  - Benchmark on identical datasets ranging from 1M to 50M rows.
  - Measure execution time, peak memory footprint (RSS), and CPU thread saturation using Python's `time` and `tracemalloc`.
  - Compile the results into an analytical benchmark report with comparative visualizations.

---

## 🗓️ Recommended Learning Schedule

| Phase | Module | Focus Topics | Suggested Time |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Module 1: Fundamentals & Architecture** | Apache Arrow, Series, DataFrame, dtypes, setup with `uv` | 2 Days |
| **Phase 2** | **Module 2: Expression Engine & Transformation** | `select()`, `with_columns()`, `pl.col()`, filtering, conditionals | 3 Days |
| **Phase 3** | **Module 3: Ingestion, Storage & PostgreSQL** | CSV, Parquet, JSON, ConnectorX, ADBC, partitioned writes | 2–3 Days |
| **Phase 4** | **Module 4: Aggregation & Window Functions** | `group_by()`, `.agg()`, `.over()`, joins, pivot/unpivot, ranking | 3 Days |
| **Phase 5** | **Module 5: Lazy API, Optimization & Streaming** | LazyFrame, `explain()`, predicate pushdown, streaming engine | 3–4 Days |
| **Phase 6** | **Module 6: Advanced Polars & Time-Series** | `group_by_dynamic()`, rolling windows, lists, structs, UDFs | 3–4 Days |
| **Phase 7** | **Module 7: Real-World Portfolio Projects** | Build Projects 1 through 10, connecting PostgreSQL & FastAPI | 6–8 Days |
| **Total** | | **Complete Beginner to Production-Grade Mastery** | **22–27 Days** |

---

## 💡 Mental Model & Translation: SQL vs. Pandas vs. Polars

For developers coming from SQL and Pandas, Polars combines the **declarative query optimization of SQL** with the **expressive syntax of Python**.

| Operation | SQL Equivalent | Pandas Equivalent | Polars Equivalent |
| :--- | :--- | :--- | :--- |
| **Select Columns** | `SELECT a, b FROM t` | `df[['a', 'b']]` | `df.select("a", "b")` |
| **Add / Transform Column** | `SELECT *, (a * b) AS c FROM t` | `df['c'] = df['a'] * df['b']` | `df.with_columns((pl.col("a") * pl.col("b")).alias("c"))` |
| **Filter Rows** | `WHERE a > 10 AND b = 'X'` | `df[(df['a'] > 10) & (df['b'] == 'X')]` | `df.filter((pl.col("a") > 10) & (pl.col("b") == "X"))` |
| **Sort Records** | `ORDER BY a DESC, b ASC` | `df.sort_values(['a', 'b'], ascending=[False, True])` | `df.sort(["a", "b"], descending=[True, False])` |
| **Group-by Aggregation** | `GROUP BY cat SELECT sum(rev), avg(qty)` | `df.groupby('cat').agg({'rev': 'sum', 'qty': 'mean'})` | `df.group_by("cat").agg(pl.col("rev").sum(), pl.col("qty").mean())` |
| **Filtered Aggregation** | `SUM(rev) FILTER (WHERE status = 'paid')` | _(Complex custom lambda / pre-filter)_ | `pl.col("rev").filter(pl.col("status") == "paid").sum()` |
| **Window Calculation** | `AVG(val) OVER (PARTITION BY cat)` | `df.groupby('cat')['val'].transform('mean')` | `pl.col("val").mean().over("cat")` |
| **Conditional Expression** | `CASE WHEN a > 0 THEN 'pos' ELSE 'neg' END` | `np.where(df['a'] > 0, 'pos', 'neg')` | `pl.when(pl.col("a") > 0).then(pl.lit("pos")).otherwise(pl.lit("neg"))` |
| **Inner Join** | `JOIN orders o ON u.id = o.user_id` | `pd.merge(u, o, left_on='id', right_on='user_id', how='inner')` | `u.join(o, left_on="id", right_on="user_id", how="inner")` |
| **Semi Join** | `WHERE id IN (SELECT user_id FROM orders)` | _(isin filtering or merge with indicator)_ | `u.join(o, left_on="id", right_on="user_id", how="semi")` |
| **Missing Value Imputation** | `COALESCE(col, 0)` | `df['col'].fillna(0)` | `pl.col("col").fill_null(0)` |
| **Unique Row Count** | `COUNT(DISTINCT user_id)` | `df['user_id'].nunique()` | `pl.col("user_id").n_unique()` |
| **Lazy Execution Plan** | Query Optimizer (`EXPLAIN ANALYZE`) | _(None, executes eagerly)_ | `df.lazy().filter(...).explain()` |

---

## ⚡ The 5 Golden Rules of High-Performance Polars

1. **Never iterate with Python loops**: Never use `for row in df.iter_rows()` for calculations. Always write vectorized expressions using `pl.col()`.
2. **Default to Lazy execution (`pl.scan_*`)**: Start pipelines with `pl.scan_parquet()` or `pl.scan_csv()`. Polars will optimize predicates and projections to read minimal data from disk.
3. **Avoid `.apply()` and `.map_elements()` whenever possible**: Python function calls cross the FFI boundary, acquire the GIL, and disable Rust's multithreaded SIMD vectorization. Use native Polars expressions.
4. **Use `.with_columns()` for all column additions**: Rather than mutating state or reassigning iteratively, declare multiple column modifications in one batch: Polars executes them in parallel.
5. **Differentiate Null vs. NaN**: In Polars, `null` indicates missing data (compatible with SQL `NULL` and Arrow null bitmaps), while `NaN` represents invalid floating-point arithmetic. Treat them intentionally.
