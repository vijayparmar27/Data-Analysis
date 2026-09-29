"""
Module 4: Data Analysis & Aggregation
=====================================
Topics Covered:
- 4.1 Descriptive Statistical Analysis (mean, median, std, quantiles, skew)
- 4.2 GroupBy Mechanics: Split-Apply-Combine
- 4.3 Advanced Aggregations with .agg() & Named Aggregations
- 4.4 Multi-Dimensional Summaries: pivot_table() & pd.crosstab()
- 4.5 MultiIndex (Hierarchical Indexes) & Cross-Sections (.xs)
- 4.6 Group Filtering (groupby().filter()) & Group Transformations (.transform())
- 4.7 Correlation & Covariance Matrices (corr, cov)
- 4.8 Rolling, Cumulative & Growth Calculations (rolling, cumsum, pct_change)
- 4.9 Frequency Distributions & Continuous Binning (pd.cut vs pd.qcut)
- 4.10 Business KPI Calculations (AOV, MoM Growth, Churn Rate)
- 4.11 Hands-on Practice Challenge with complete solution!
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
# 4.1 Descriptive Statistical Analysis
# ----------------------------------------------------------------------
section("4.1 Descriptive Statistics & Distribution Properties")

# Sample order value distribution
orders = pd.DataFrame({
    "order_id": range(1, 11),
    "category": ["Electronics", "Clothing", "Electronics", "Groceries", "Clothing",
                 "Electronics", "Groceries", "Clothing", "Electronics", "Groceries"],
    "revenue": [1200, 85, 450, 60, 110, 890, 45, 95, 310, 75],
    "items_count": [2, 3, 1, 6, 2, 4, 5, 2, 1, 7],
    "margin_pct": [0.25, 0.40, 0.20, 0.15, 0.35, 0.28, 0.12, 0.38, 0.22, 0.14]
})
print("Sample Dataset:\n", orders)

print(f"Mean Revenue:   ${orders['revenue'].mean():.2f}")
print(f"Median Revenue: ${orders['revenue'].median():.2f}")
print(f"Std Deviation:  ${orders['revenue'].std():.2f}")
print(f"25th Percentile (Q1): ${orders['revenue'].quantile(0.25):.2f}")
print(f"75th Percentile (Q3): ${orders['revenue'].quantile(0.75):.2f}")
print(f"Distribution Skewness: {orders['revenue'].skew():.2f} (Positive = Right-skewed by high electronics orders)")


# ----------------------------------------------------------------------
# 4.2 GroupBy Fundamentals: Split-Apply-Combine
# ----------------------------------------------------------------------
section("4.2 GroupBy: The Split-Apply-Combine Pattern")

# Equivalent SQL: SELECT category, SUM(revenue) FROM orders GROUP BY category;
grouped_sum = orders.groupby("category")["revenue"].sum()
print("Revenue by Category (SQL GROUP BY equivalent):\n", grouped_sum)

# Multiple grouping keys:
orders["channel"] = ["Online", "Store", "Online", "Store", "Online", "Online", "Store", "Store", "Online", "Store"]
multi_group = orders.groupby(["category", "channel"])["revenue"].sum()
print("\nMulti-Column GroupBy (category x channel):\n", multi_group)


# ----------------------------------------------------------------------
# 4.3 Advanced Aggregation with .agg() & Named Aggregations
# ----------------------------------------------------------------------
section("4.3 Advanced Aggregations with .agg() & Named Aggregations")

# Named Aggregations (Pandas modern recommended syntax for clean column names):
# Generates clear, unambiguous headers without multi-level column indexes!
summary_kpis = orders.groupby("category").agg(
    total_revenue=("revenue", "sum"),
    average_order_value=("revenue", "mean"),
    order_volume=("order_id", "count"),
    avg_items_per_order=("items_count", "mean"),
    max_single_sale=("revenue", "max"),
    weighted_margin=("margin_pct", lambda m: round(m.mean() * 100, 1))
)
print("Business Executive Summary using Named Aggregations:\n", summary_kpis)


# ----------------------------------------------------------------------
# 4.4 Pivot Tables & Cross-Tabulations (crosstab)
# ----------------------------------------------------------------------
section("4.4 Multi-Dimensional Summaries: pivot_table() & crosstab()")

# pivot_table(): Summarize revenue across category (rows) and channel (columns)
pivot = orders.pivot_table(
    index="category",
    columns="channel",
    values="revenue",
    aggfunc="sum",
    fill_value=0,
    margins=True,       # Adds 'All' row and column (Total summaries)
    margins_name="Total"
)
print("Revenue Pivot Table (Category vs Channel with Totals):\n", pivot)

# pd.crosstab(): Frequency counts or percentage matrix
cross_tab = pd.crosstab(
    orders["category"],
    orders["channel"],
    normalize="all"     # Converts cell counts to proportions summing to 1.0 (100%)
) * 100
print("\nOrder Frequency Cross-Tabulation (% of Total Orders):\n", cross_tab.round(1))


# ----------------------------------------------------------------------
# 4.5 MultiIndex (Hierarchical Indexing) & Cross-Sections (.xs)
# ----------------------------------------------------------------------
section("4.5 MultiIndex & Cross-Sections (.xs)")

# Creating a MultiIndex DataFrame
multi_df = orders.groupby(["category", "channel"]).agg(
    total_rev=("revenue", "sum"),
    avg_rev=("revenue", "mean")
)
print("MultiIndex DataFrame:\n", multi_df)
print("\nIndex Levels:", multi_df.index.levels)

# Selecting a cross-section using .xs()
# e.g., Extract all rows where channel == 'Online' across all categories:
online_cross_section = multi_df.xs("Online", level="channel")
print("\nCross-section across all categories where channel == 'Online':\n", online_cross_section)


# ----------------------------------------------------------------------
# 4.6 Group Filtering and Group Transformations
# ----------------------------------------------------------------------
section("4.6 Group Filtering (.filter()) & Group Transformations (.transform())")

# 1. Group Filtering: Keep only categories with total revenue exceeding $300
high_rev_categories = orders.groupby("category").filter(lambda grp: grp["revenue"].sum() > 300)
print(f"Categories with total revenue > $300 (Kept {len(high_rev_categories)} rows):\n",
      high_rev_categories[["category", "revenue"]])

# 2. Group Transformation: Compute each order's percentage contribution to its category
category_total = orders.groupby("category")["revenue"].transform("sum")
orders["pct_of_category_revenue"] = (orders["revenue"] / category_total) * 100
print("\nOrder revenue share within its category (%):\n",
      orders[["category", "revenue", "pct_of_category_revenue"]])


# ----------------------------------------------------------------------
# 4.7 Correlation and Covariance Matrices
# ----------------------------------------------------------------------
section("4.7 Correlation & Covariance Matrices")

numeric_cols = orders[["revenue", "items_count", "margin_pct"]]
correlation_matrix = numeric_cols.corr()
print("Pearson Correlation Matrix:\n", correlation_matrix.round(3))
print("Interpretation: revenue vs items_count has correlation of",
      round(correlation_matrix.loc["revenue", "items_count"], 3))


# ----------------------------------------------------------------------
# 4.8 Rolling, Cumulative & Growth Calculations
# ----------------------------------------------------------------------
section("4.8 Rolling Windows, Cumulative Sums & Growth Rates")

# Create chronological daily revenue time-series
dates = pd.date_range(start="2026-03-01", periods=10, freq="D")
daily_sales = pd.DataFrame({
    "date": dates,
    "daily_revenue": [1200, 1450, 980, 1600, 2100, 1850, 1300, 1950, 2400, 2200]
}).set_index("date")

# 1. Cumulative Revenue (Running Total)
daily_sales["running_total"] = daily_sales["daily_revenue"].cumsum()

# 2. 3-Day Rolling Moving Average (Smoothing out daily volatility)
daily_sales["3_day_moving_avg"] = daily_sales["daily_revenue"].rolling(window=3).mean()

# 3. Day-over-Day Growth Rate (pct_change)
daily_sales["dod_growth_pct"] = daily_sales["daily_revenue"].pct_change() * 100

print("Daily Sales Time Series Analytics:\n", daily_sales.round(1))


# ----------------------------------------------------------------------
# 4.9 Frequency Binning (pd.cut vs pd.qcut)
# ----------------------------------------------------------------------
section("4.9 Continuous Data Binning: pd.cut() vs pd.qcut()")

prices = pd.Series([15, 22, 45, 50, 75, 80, 110, 250, 320, 500, 850, 1200], name="price")

# pd.cut(): Equal-width intervals (e.g. Budget, Mid-range, Luxury)
bins = [0, 50, 200, 1500]
labels = ["Budget (<$50)", "Mid-Range ($50-$200)", "Luxury (>$200)"]
price_segments = pd.cut(prices, bins=bins, labels=labels)
print("Equal-Width Segmentation via pd.cut():\n", price_segments.value_counts())

# pd.qcut(): Equal-frequency quantiles (e.g., Quartiles with 25% of items per bin)
price_quartiles = pd.qcut(prices, q=4, labels=["Q1 (Low)", "Q2 (Med-Low)", "Q3 (Med-High)", "Q4 (High)"])
print("\nQuantile Segmentation via pd.qcut() (Equal count per bin):\n", price_quartiles.value_counts())


# ----------------------------------------------------------------------
# 4.10 Hands-on Practice Challenge
# ----------------------------------------------------------------------
section("4.10 Hands-on Practice Challenge")
print("""
PRACTICE EXERCISE:
Given customer order records:
df_orders = pd.DataFrame({
    "customer_id": ["C1", "C2", "C1", "C3", "C2", "C1", "C3", "C4"],
    "region": ["North", "South", "North", "West", "South", "North", "West", "East"],
    "order_amount": [120, 450, 80, 210, 300, 150, 95, 600],
    "is_discounted": [False, True, False, False, True, True, False, False]
})

Your Task:
1. Group by 'region' and compute:
   - 'total_spend' (sum of order_amount)
   - 'avg_order_value' (mean of order_amount)
   - 'orders_count' (count of orders)
2. Create a pivot table showing total 'order_amount' broken down by 'region' (rows) and 'is_discounted' (columns),
   including row/column margins.
3. Compute each customer's lifetime value (total spend) and rank them highest to lowest.
""")

# --- Challenge Solution ---
df_orders = pd.DataFrame({
    "customer_id": ["C1", "C2", "C1", "C3", "C2", "C1", "C3", "C4"],
    "region": ["North", "South", "North", "West", "South", "North", "West", "East"],
    "order_amount": [120, 450, 80, 210, 300, 150, 95, 600],
    "is_discounted": [False, True, False, False, True, True, False, False]
})

# 1. Group by region with named aggregations
regional_report = df_orders.groupby("region").agg(
    total_spend=("order_amount", "sum"),
    avg_order_value=("order_amount", "mean"),
    orders_count=("order_amount", "count")
)
print("1. Regional Performance:\n", regional_report)

# 2. Pivot table
discount_pivot = df_orders.pivot_table(
    index="region",
    columns="is_discounted",
    values="order_amount",
    aggfunc="sum",
    fill_value=0,
    margins=True
)
print("\n2. Discount Usage Pivot Table:\n", discount_pivot)

# 3. Customer Lifetime Value (CLV)
customer_clv = df_orders.groupby("customer_id")["order_amount"].sum().sort_values(ascending=False)
print("\n3. Customer Lifetime Value Ranking:\n", customer_clv)
print("\n Module 4 Completed Successfully!")
