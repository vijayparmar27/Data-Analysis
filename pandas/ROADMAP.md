# Pandas Learning Roadmap (Beginner to Advanced)

> **Context**: Tailored for a Full-Stack & Backend Developer (Python, NumPy, PostgreSQL, FastAPI, TypeScript/Node.js). We emphasize real-world data pipelines, bridging SQL query patterns to Pandas operations, memory optimization, and building automated reporting services.

---

## 📊 Learning Progress: `0 / 59 Topics Completed`

---

## Module 1: Pandas Fundamentals (Beginner)

_Goal: Understand the core 1D (Series) and 2D (DataFrame) tabular data structures, index mechanics, and fundamental manipulation patterns._

- [ ] **1.1** What is Pandas and why use it? (Python dicts/objects vs. NumPy C-buffers vs. Pandas labelled DataFrames)
- [ ] **1.2** Environment setup and package management with `uv` (`uv add pandas numpy pyarrow openpyxl sqlalchemy`)
- [ ] **1.3** Importing conventions & namespace standards (`import pandas as pd`, `import numpy as np`)
- [ ] **1.4** Series vs. DataFrame architecture (1D labelled array vs. 2D heterogeneous column store)
- [ ] **1.5** Creating DataFrames from lists, dictionaries, records, and NumPy 2D arrays
- [ ] **1.6** DataFrame inspection attributes: `.shape`, `.columns`, `.index`, `.dtypes`, `.ndim`, `.size`
- [ ] **1.7** Selecting columns (single bracket `df['col']` vs double bracket `df[['col1', 'col2']]`)
- [ ] **1.8** Row & cell selection: Label-based `.loc[]` vs. Integer position-based `.iloc[]`
- [ ] **1.9** Setting, resetting, and renaming indexes & columns (`set_index()`, `reset_index()`, `rename()`)
- [ ] **1.10** Adding, updating, and dropping columns & rows (`df['new'] = ...`, `assign()`, `drop(..., axis=1)`)

---

## Module 2: Data Loading & Inspection (Beginner)

_Goal: Ingest structured data from diverse sources (CSV, Excel, JSON, PostgreSQL), inspect schemas, and export reports._

- [ ] **2.1** Reading CSV files (`pd.read_csv()` with separators, headers, parse_dates, na_values)
- [ ] **2.2** Reading and writing Excel files (`read_excel()`, `to_excel()`, multi-sheet workbooks with `openpyxl`)
- [ ] **2.3** Reading and parsing JSON data (`read_json()`, nested JSON normalization with `pd.json_normalize()`)
- [ ] **2.4** Loading data directly from PostgreSQL / SQL (`pd.read_sql()`, SQLAlchemy engine, connection pooling)
- [ ] **2.5** DataFrame exploration: `.head()`, `.tail()`, and random inspection with `.sample()`
- [ ] **2.6** Structural diagnostics: `.info()`, descriptive statistics `.describe()`, and frequency counts `.value_counts()`
- [ ] **2.7** Data types audit and memory footprint diagnostics (`df.dtypes`, `df.memory_usage(deep=True)`)
- [ ] **2.8** Exporting clean DataFrames to CSV (`to_csv()`), Excel (`to_excel()`), JSON (`to_json()`), and Parquet (`to_parquet()`)

---

## Module 3: Data Cleaning & Transformation (Intermediate)

_Goal: Clean dirty real-world datasets, impute missing values, standardize strings, handle outliers, and transform columns._

- [ ] **3.1** Handling missing values: Detection (`isna()`, `notna()`), imputation (`fillna()`, forward/backward fill), and removal (`dropna()`)
- [ ] **3.2** Detecting, inspecting, and removing duplicate records (`duplicated()`, `drop_duplicates(subset=[...])`)
- [ ] **3.3** Type conversions: `.astype()`, safe numeric parsing (`pd.to_numeric()`), and datetime parsing (`pd.to_datetime()`)
- [ ] **3.4** Vectorized string operations with `.str` accessor (`lower()`, `strip()`, `contains()`, `replace()`, regex extraction)
- [ ] **3.5** Applying functions: Row/column-wise `.apply()`, element-wise `.map()`, and group-aware `.transform()`
- [ ] **3.6** Filtering rows with boolean masks, compound logical operators (`&`, `|`, `~`), and SQL-like `df.query()`
- [ ] **3.7** Sorting and ranking: `sort_values()`, multi-column sort, `sort_index()`, and ranking methods (`rank()`)
- [ ] **3.8** Replacing values: `.replace()`, conditional masking with `.where()`, `.mask()`, and `np.select()`
- [ ] **3.9** Renaming and standardizing data (cleaning column casing, trimming whitespace, standardizing categorical labels)
- [ ] **3.10** Outlier detection and treatment: Interquartile Range (IQR), Z-score thresholds, and percentile clipping (`clip()`)

---

## Module 4: Data Analysis & Aggregation (Intermediate)

_Goal: Perform business analytics, group-level aggregations, pivot tables, cross-tabulations, and KPI computations._

- [ ] **4.1** Descriptive statistical summaries: `mean()`, `median()`, `mode()`, `std()`, `quantile()`, `skew()`
- [ ] **4.2** GroupBy fundamentals: The Split-Apply-Combine pattern (`df.groupby('category')`)
- [ ] **4.3** Advanced aggregation with `.agg()`: Multiple metrics per column, custom functions, and named aggregations
- [ ] **4.4** Multi-dimensional summary with `pivot_table()` (index, columns, values, aggfunc, margins) and `pd.crosstab()`
- [ ] **4.5** MultiIndex and hierarchical data: Indexing, slicing, levels manipulation, and cross-sections (`.xs()`)
- [ ] **4.6** Advanced filtering with boolean masks and group-level filtering (`groupby().filter()`)
- [ ] **4.7** Correlation and covariance matrices: `df.corr()`, `df.cov()`, and relationship identification
- [ ] **4.8** Rolling, cumulative, and window calculations: `rolling().mean()`, `cumsum()`, `cummax()`, and percentage change (`pct_change()`)
- [ ] **4.9** Frequency distributions and binning continuous data: `pd.cut()` (equal width) vs. `pd.qcut()` (quantiles)
- [ ] **4.10** Business KPI calculations: MoM / YoY growth rates, Customer Churn, Average Order Value (AOV), and Conversion Rates

---

## Module 5: Data Combining & Advanced Pandas (Advanced)

_Goal: Master complex relational merges, reshaping, time-series resampling, memory optimization, and streaming chunk processing._

- [ ] **5.1** Relational merges and SQL-style joins: `pd.merge()` (inner, left, right, outer joins, join keys, and validation `validate='1:m'`)
- [ ] **5.2** Concatenation and stacking: `pd.concat()` along rows (`axis=0`) and columns (`axis=1`), hierarchical keys
- [ ] **5.3** Reshaping data: Wide to Long format using `pd.melt()` and Long to Wide using `pivot()`
- [ ] **5.4** Index pivoting with `stack()` and `unstack()` for multi-level tables
- [ ] **5.5** Comprehensive Date and Time handling: Timestamps, DatetimeIndex, `.dt` accessor, date arithmetic with `pd.Timedelta`
- [ ] **5.6** Time series frequency resampling: Downsampling/Upsampling with `.resample('D' / 'W' / 'M')` and gap filling
- [ ] **5.7** Advanced window operations: Exponentially weighted moving averages (`ewm()`) and expanding windows (`expanding()`)
- [ ] **5.8** Categorical data types (`pd.CategoricalDtype`): Memory footprint reduction (up to 80-90%) and ordered categories
- [ ] **5.9** Vectorized operations vs. Python loops: Using `eval()` and `query()` for high-throughput computations
- [ ] **5.10** Performance profiling & memory optimization: Downcasting integer/float types, PyArrow backend engine (`dtype_backend='pyarrow'`)
- [ ] **5.11** Large dataset chunk processing: Reading multi-gigabyte files via `pd.read_csv(chunksize=...)` and batch streaming

---

## Module 6: Real-World Portfolio Projects (Practical)

_Goal: Build production-grade data pipelines, business intelligence reports, and analytics workflows connecting to SQL databases._

- [ ] **Project 1: E-Commerce Sales & Revenue Analysis (Beginner)**
  - Load orders, order items, and products from CSV/PostgreSQL.
  - Clean missing customer emails and invalid order states.
  - Compute total revenue, gross margin, AOV, and top 10 bestselling products.
- [ ] **Project 2: Customer Behavior & RFM Segmentation (Intermediate)**
  - Calculate Recency, Frequency, and Monetary scores per customer.
  - Segment users into Champions, Loyalists, At-Risk, and Inactive cohorts.
  - Export structured segmentation lists for marketing automation.
- [ ] **Project 3: Automated Data Cleaning & Validation Pipeline (Intermediate)**
  - Ingest raw messy transaction logs with mixed date formats, corrupted IDs, and outliers.
  - Build an idempotent cleaning pipeline using method chaining (`pipe()`).
  - Validate schema types, non-null constraints, and value boundaries.
- [ ] **Project 4: Monthly Revenue & Business KPI Reporting Engine (Intermediate)**
  - Compute Monthly Recurring Revenue (MRR), MoM growth rates, customer retention, and cohort churn.
  - Summarize key financial metrics with formatted pivot tables.
- [ ] **Project 5: Inventory Turnover & Stock Depletion Analysis (Advanced)**
  - Track stock replenishment and daily depletion velocity.
  - Compute days of inventory remaining (DOI), identify dead stock, and generate replenishment alerts.
- [ ] **Project 6: Marketing Campaign Performance & Funnel Conversion (Advanced)**
  - Analyze multi-channel campaign clicks, signups, and purchases.
  - Compute drop-off rates at each stage of the user conversion funnel.
  - Calculate Return on Ad Spend (ROAS) and Cost Per Acquisition (CPA).
- [ ] **Project 7: A/B Testing & Conversion Rate Significance (Advanced)**
  - Analyze control vs. treatment groups for variant conversion performance.
  - Compute conversion rates, confidence intervals, and hypothesis significance tests (p-values).
- [ ] **Project 8: Financial Time-Series & Portfolio Return Analysis (Advanced)**
  - Ingest multi-asset daily stock price history.
  - Resample to weekly/monthly returns, compute 30-day rolling volatility, Sharpe ratio, and drawdowns.
- [ ] **Project 9: PostgreSQL to Pandas ETL & Reporting Workflow (Backend Integration)**
  - Connect to PostgreSQL database using SQLAlchemy.
  - Execute parameterized analytical SQL queries, transform data in Pandas, and write aggregate analytical tables back to database.
- [ ] **Project 10: Automated Multi-Tab Excel Business Reports (BI Automation)**
  - Generate an executive-ready `.xlsx` workbook using Pandas and `openpyxl`/`xlsxwriter`.
  - Add summary KPI cards, category breakdown tabs, conditional formatting, and embedded charts.

---

## 🗓️ Recommended Learning Schedule

| Phase | Module | Focus Topics | Est. Time |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Module 1: Fundamentals** | Series, DataFrame, `.loc`/`.iloc`, column indexing, adding/updating columns | 2 Days |
| **Phase 2** | **Module 2: Data Loading & Inspection** | CSV, Excel, JSON, PostgreSQL ingestion, `.info()`, `.describe()`, exports | 2 Days |
| **Phase 3** | **Module 3: Cleaning & Transformation** | Missing values, duplicates, `.str` operations, filtering, outlier clipping | 3–4 Days |
| **Phase 4** | **Module 4: Analysis & Aggregation** | Split-apply-combine, `groupby()`, `.agg()`, `pivot_table()`, KPIs, rolling | 3–4 Days |
| **Phase 5** | **Module 5: Advanced & Performance** | Joins/merges, time series resampling, categorical dtypes, chunks, PyArrow | 4–5 Days |
| **Phase 6** | **Module 6: Real-World Projects** | Build Projects 1 through 10, connecting PostgreSQL and automated reports | 5–7 Days |
| **Total** | | **Complete Beginner to Advanced Mastery** | **19–24 Days** |

---

## 💡 SQL to Pandas Mental Translation (For Backend Developers)

| SQL Operation | Pandas Equivalent |
| :--- | :--- |
| `SELECT col1, col2 FROM table` | `df[['col1', 'col2']]` |
| `WHERE col > 100 AND status = 'active'` | `df[(df['col'] > 100) & (df['status'] == 'active')]` or `df.query('col > 100 and status == "active"')` |
| `ORDER BY price DESC LIMIT 10` | `df.sort_values('price', ascending=False).head(10)` |
| `SELECT DISTINCT category FROM table` | `df['category'].unique()` or `df[['category']].drop_duplicates()` |
| `GROUP BY category SELECT sum(rev), avg(qty)` | `df.groupby('category').agg(total_rev=('rev', 'sum'), avg_qty=('qty', 'mean'))` |
| `HAVING count(*) > 5` | `df.groupby('category').filter(lambda g: len(g) > 5)` |
| `INNER JOIN orders ON users.id = orders.user_id` | `pd.merge(users, orders, left_on='id', right_on='user_id', how='inner')` |
| `UNION ALL` | `pd.concat([df1, df2], ignore_index=True)` |
| `COALESCE(col, 0)` | `df['col'].fillna(0)` |
| `CASE WHEN col > 50 THEN 'High' ELSE 'Low' END` | `np.where(df['col'] > 50, 'High', 'Low')` |
| `ROW_NUMBER() OVER (PARTITION BY cat ORDER BY val)` | `df.groupby('cat')['val'].rank(method='first', ascending=True)` |
