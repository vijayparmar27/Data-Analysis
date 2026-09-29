# Data Analysis Roadmap for a Full-Stack Developer

I've compressed the 12 areas in your brief into one path. Ask for any section in full depth (code, datasets, exercises) and I'll expand it.

## 0. Where you fit: DA vs DS vs ML vs DE

| Role               | Core question                            | Main tools                        | Overlap with you          |
| ------------------ | ---------------------------------------- | --------------------------------- | ------------------------- |
| **Data Analyst**   | "What happened, and why?"                | SQL, Pandas, BI tools, statistics | High: SQL, Python, APIs   |
| **Data Engineer**  | "How do I move and store data reliably?" | Airflow, dbt, Spark, warehouses   | Very high: backend skills |
| **Data Scientist** | "What will happen? What causes it?"      | Statistics, ML, experimentation   | Medium                    |
| **ML Engineer**    | "How do I ship models to production?"    | scikit-learn, PyTorch, MLOps      | Medium                    |

Your backend background makes you a strong **analytics engineer / full-stack analyst**, so you should skip basic Python and focus on **SQL depth, statistics, and business thinking**. Those are usually where developers are weakest.

## 1. Tool priorities

**Essential:** SQL/PostgreSQL, Pandas, NumPy, Matplotlib/Seaborn/Plotly, Jupyter, Git, one BI tool (Power BI _or_ Tableau, not both), and basic statistics.

**Valuable soon:** Polars, DuckDB, dbt basics, Airflow or cron, and FastAPI for analytics endpoints.

**Optional or later:** Spark (only for data too big for one machine), data lakes, cloud warehouses (BigQuery, Snowflake, Redshift), and Redis for caching heavy results.

**Data Engineering / Data Science territory (don't go deep yet):** Spark clusters, Kafka, feature stores, deep learning, MLOps.

## 2. The learning path

### Phase 1: Python for analysis (Weeks 1–3)

- **Why:** it's the workhorse for cleaning, reshaping, and exploring data.
- **Learn:** NumPy vectorization, Pandas (`groupby`, `merge`, `pivot_table`, `resample`, `.str`/`.dt` accessors), then Polars (lazy evaluation, expressions), reading CSV/JSON/Excel/Parquet, missing data strategies, and validation with `pandera` or `pydantic`.
- **Performance:** use `category` dtypes, read in chunks, prefer Parquet over CSV, and swap Pandas for Polars or DuckDB when data gets large.

```python
import polars as pl

df = (
    pl.scan_csv("orders.csv")
      .with_columns(pl.col("created_at").str.to_datetime())
      .filter(pl.col("status") == "completed")
      .group_by(pl.col("created_at").dt.truncate("1mo"))
      .agg(pl.col("total").sum().alias("revenue"))
      .sort("created_at")
      .collect()
)
```

- **Exercise:** clean a messy dataset (mixed date formats, duplicates, nulls) and write a validation schema for it.
- **Leads to:** EDA and SQL, since Pandas and SQL express the same ideas.

### Phase 2: SQL and advanced PostgreSQL (Weeks 2–6, overlaps with Phase 1)

This is the most valuable skill for analyst jobs, so go deep.

- **Learn:** CTEs, window functions (`ROW_NUMBER`, `LAG/LEAD`, `SUM() OVER`, `NTILE`), conditional aggregation with `FILTER`, `generate_series` for filling date gaps, JSONB operators, materialized views, indexes (B-tree, GIN, partial), and `EXPLAIN (ANALYZE, BUFFERS)`.

```sql
-- Monthly cohort retention
WITH first_order AS (
  SELECT user_id, date_trunc('month', MIN(created_at)) AS cohort
  FROM orders GROUP BY user_id
),
activity AS (
  SELECT DISTINCT o.user_id, f.cohort,
         date_trunc('month', o.created_at) AS active_month
  FROM orders o JOIN first_order f USING (user_id)
)
SELECT cohort,
       (EXTRACT(year FROM age(active_month, cohort)) * 12
        + EXTRACT(month FROM age(active_month, cohort)))::int AS month_n,
       COUNT(*) AS users
FROM activity GROUP BY 1, 2 ORDER BY 1, 2;
```

- **Exercises:** 50+ problems on StrataScratch, DataLemur, or LeetCode SQL. Then optimize a slow query 10x using an index and a rewritten plan.
- **Leads to:** cohort, funnel, and retention analysis.

### Phase 3: EDA (Weeks 5–7)

- **Workflow:** shape and dtypes → missingness → distributions → outliers (IQR, z-score, robust methods) → relationships (correlation, grouped comparisons) → hypotheses → written findings.
- **Tools:** `ydata-profiling` for a quick first pass, then manual analysis with Seaborn.
- **Habit to build:** every EDA ends with three to five _business-relevant statements_, not just charts.

### Phase 4: Statistics (Weeks 6–10)

Developers tend to skip this, and it separates analysts from people who make charts.

- **Learn:** descriptive statistics, distributions (normal, binomial, Poisson, exponential), the Central Limit Theorem, confidence intervals, hypothesis testing (t-test, chi-square, Mann-Whitney), p-values and their misuse, effect sizes, power and sample size, multiple-testing correction, and linear/logistic regression.
- **A/B testing:** sample size calculation, novelty effects, sample ratio mismatch, and practical vs statistical significance.
- **Libraries:** `scipy.stats`, `statsmodels`.
- **Resources:** _Think Stats_ (free), _Statistics for Data Science_-type courses, Khan Academy for gaps.
- **Exercise:** simulate an A/B test in Python 1,000 times and see how often a null effect looks "significant".

### Phase 5: Visualization and BI (Weeks 8–12)

- **Python:** Matplotlib basics, Seaborn for statistics, Plotly for interactivity.
- **Chart choice rules:** trend → line, comparison → bar, distribution → histogram/box, relationship → scatter, composition → stacked bar (avoid pie charts beyond 3–4 slices), funnel → horizontal bar.
- **BI tool:** pick **Power BI** (more job postings in many markets, including India) unless your target companies use Tableau. Learn the data model, DAX basics, and drill-through.
- **Storytelling:** lead with the conclusion, one message per chart, annotate the insight.
- **Deliverable:** a KPI dashboard with date filters, comparisons to the prior period, and a written summary.

### Phase 6: ETL and pipelines (Weeks 10–14)

- **Learn:** REST extraction with `httpx`, scraping fundamentals (`BeautifulSoup`, Playwright, and robots.txt/terms respect), idempotent loads, incremental loading via watermark columns, upserts (`INSERT ... ON CONFLICT`), data quality checks (row counts, nulls, freshness), and scheduling with cron first, then Airflow.
- **Modern pattern:** extract → load raw → transform inside the database with **dbt** (ELT).
- **Leads to:** your analytics platform capstone.

### Phase 7: Advanced analysis (Weeks 12–20)

- **Cohort and retention:** the SQL above, plus retention heatmaps.
- **RFM segmentation:** score recency, frequency, and monetary value into quintiles, then name segments (Champions, At Risk, Hibernating).
- **CLV:** start with the simple form (average order value × frequency × lifespan), then BG/NBD via the `lifetimes` library.
- **Churn:** define churn carefully (a business decision), do survival analysis basics, then a logistic regression baseline.
- **Funnel:** step-to-step conversion, drop-off by segment.
- **Time series:** decomposition (trend, seasonality), moving averages, then Prophet or `statsmodels` (SARIMA/ETS). Always benchmark against a naive forecast.
- **Anomalies:** rolling z-scores, IQR, isolation forest for fundamentals.
- **Leads to:** the industry projects below.

## 3. Industry use cases

Every case follows the same template: problem → data → method → output → insight.

| Industry              | Business problem                                   | Key data                                     | Method                                                                                 | Insight to produce                                              |
| --------------------- | -------------------------------------------------- | -------------------------------------------- | -------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| **E-commerce**        | Which products and customers drive revenue?        | orders, items, carts, sessions               | RFM, funnel, cart abandonment, ABC inventory analysis, forecasting                     | "20% of SKUs drive 70% of revenue; stock-outs cost X"           |
| **SaaS**              | Why do users leave?                                | events, subscriptions, plans                 | MRR movements (new/expansion/churn), cohorts, activation analysis                      | "Users who do X in week 1 retain 2x"                            |
| **Fintech**           | Where do payments fail, and what looks fraudulent? | transactions, gateway responses              | Failure rate by method/bank/hour, velocity rules, anomaly detection                    | "UPI failures spike on bank X at peak hours"                    |
| **Healthcare**        | Are resources used efficiently?                    | appointments, wait times, no-shows           | No-show rate, utilization, peak-hour analysis                                          | "Tuesday slots have 30% no-shows; overbook safely"              |
| **Marketing**         | Which channels earn their spend?                   | spend, clicks, conversions                   | CAC, ROAS, conversion funnels, attribution (first/last touch)                          | "Channel A has cheap clicks but poor LTV"                       |
| **Logistics**         | Why are deliveries late or costly?                 | shipments, carriers, routes                  | On-time rate, delay root causes, cost per shipment by zone                             | "Carrier B delays cluster in region Y"                          |
| **BI / executive**    | Is the business on track?                          | all of the above                             | KPI trees, period-over-period, targets vs actuals                                      | A one-page dashboard leadership actually reads                  |
| **Booking platforms** | Are slots used, and who comes back?                | bookings, providers, cancellations, payments | Slot utilization, cancellation analysis, provider ranking, retention, revenue per slot | "Off-peak slots sit 60% empty; targeted offers could fill them" |

Booking platforms are the best fit for you, because you can generate realistic data from your own multi-tenant booking app instead of relying only on public datasets.

## 4. Building the full analytics system with your stack

```
Next.js / RN app ──events──> Express API ──> PostgreSQL (OLTP)
                                                │
                              Python ETL (cron/Airflow) ──> analytics schema
                                                │              (dbt models, materialized views)
                                        FastAPI analytics API ──(Redis cache)──>
                                                │
                               Next.js dashboard  +  React Native mobile view
                                     (Recharts / Plotly / ECharts)
```

Key design decisions:

1. **Event tracking:** log structured events (`booking_created`, `payment_failed`) into an append-only `events` table with JSONB properties.
2. **Separate analytics from OLTP:** use a dedicated `analytics` schema or a read replica so heavy queries never slow the app.
3. **Transform layer:** raw → staging → marts (`fct_bookings`, `dim_customers`), built with dbt or plain SQL.
4. **Serving:** FastAPI endpoints return pre-aggregated JSON, cached in Redis with sensible TTLs.
5. **Docker Compose:** Postgres, FastAPI, Redis, Airflow, and the dashboard in one reproducible stack.
6. **Quality:** tests for row counts, uniqueness, and freshness that fail the pipeline loudly.

## 5. Projects

**Beginner (Weeks 1–8)**

1. _Data cleaning and EDA_ on a messy public dataset. Deliver a notebook with a README.
2. _SQL reporting pack:_ 15 business queries on a sample e-commerce DB (Olist), with written insights.
3. _Simple visualization report_ using Plotly.

**Intermediate (Weeks 9–16)** 4. _E-commerce analytics:_ sales trends, RFM segmentation, and a Power BI dashboard. 5. _A/B test analysis:_ power calculation plus a decision memo. 6. _Automated weekly report:_ a script that queries Postgres and emails or generates a PDF.

**Advanced (Weeks 17–24)** 7. _SaaS product analytics:_ cohorts, activation, churn model. 8. _Forecasting:_ revenue or demand with backtesting against naive baselines. 9. _Large-scale analysis:_ 10M+ rows with DuckDB/Polars, showing memory and runtime comparisons with Pandas.

**Industry-ready capstone (Weeks 25–32)** 10. _Booking analytics platform:_ event tracking → ETL → Postgres marts → FastAPI → Next.js dashboard, with data validation, scheduling, and Docker. Include slot utilization, cancellation analysis, provider performance, retention, and a revenue forecast. Ship it with an architecture diagram, a live demo or video, and a case-study README.

Each project's GitHub deliverables should be: problem statement, data source, reproducible setup (`docker compose up`), clean code, a findings section written for non-technical readers, and screenshots.

**Datasets:** Kaggle (Olist Brazilian e-commerce, Telco churn, Online Retail II), NYC Taxi (large-scale), data.gov.in, Google BigQuery public datasets, Instacart, and Kaggle healthcare appointment no-show data.

## 6. Schedule (about 2 hrs/day on top of work, ~8 months)

| Months | Focus                              | Monthly deliverable                     |
| ------ | ---------------------------------- | --------------------------------------- |
| 1      | Pandas/Polars + SQL foundations    | Cleaning project + 15 SQL reports       |
| 2      | Advanced SQL + EDA + visualization | EDA notebook + first dashboard          |
| 3      | Statistics + A/B testing           | A/B test memo                           |
| 4      | BI tool + ETL basics               | E-commerce dashboard + automated report |
| 5      | Cohorts, RFM, CLV, funnels         | Segmentation and retention project      |
| 6      | Time series + churn + anomalies    | Forecasting and churn project           |
| 7–8    | Full-stack analytics platform      | Capstone + case study                   |

**Weekly rhythm:** Mon–Thu learn and code (about 1 hr theory, 1 hr practice), Fri weekly challenge (a SQL set or mini-analysis), Sat build project work, Sun review and write up one insight publicly (blog or LinkedIn).

## 7. Common mistakes to avoid

- Over-focusing on tools and ignoring the business question.
- Charts with no conclusion.
- Trusting p-values without effect sizes or sample sizes.
- Skipping data validation, so one bad join silently corrupts results.
- Doing in Pandas what SQL does better (and vice versa).
- Assuming correlation is causation.
- Building models before understanding the data.
- Learning Spark or deep learning too early.

## 8. Interview preparation

- **SQL rounds:** window functions, cohort/retention queries, de-duplication, running totals, gaps-and-islands.
- **Case questions:** "Revenue dropped 15%. What do you check?" Structure it as data quality → segmentation (channel, region, product, cohort) → funnel step → external factors.
- **Stats questions:** explain p-values simply, design an A/B test, and choose metrics.
- **Product-sense questions:** define success metrics for a feature.
- **Portfolio:** 3–4 strong projects beat 15 weak ones. Lead with the capstone.

## Suggested first step

Start this week with two parallel tracks: 30 minutes of SQL window-function practice daily, and cleaning one messy dataset in Pandas, then redoing the same cleaning in Polars to feel the difference.

If you'd like, I can turn this into a downloadable doc, or expand any one phase into a full lesson plan with code and exercises. The SQL phase or the capstone architecture would be my suggestion.
