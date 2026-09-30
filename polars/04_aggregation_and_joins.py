"""
Module 4: Aggregation, Window Functions & Relational Joins
=========================================================
Topics Covered:
- 4.1 Core Aggregation Functions & pl.len()
- 4.2 group_by() & Parallel Aggregation with .agg()
- 4.3 Filtered Aggregations within group_by() (SQL FILTER WHERE equivalent)
- 4.4 Relational Joins (Inner, Left, Full, Cross)
- 4.5 Filtering Joins (Semi-Join & Anti-Join for Set Operations)
- 4.6 Concatenation: Vertical, Horizontal, and Diagonal (Schema Unification)
- 4.7 Reshaping: pivot() and unpivot() (Long <-> Wide)
- 4.8 Window Functions with .over() (SQL OVER PARTITION BY equivalent)
- 4.9 Ranking, Cumulative Operations & Lags (rank, cum_sum, shift, pct_change)
- 4.10 Statistical Correlation & Covariance (corr, cov)
- 4.11 Hands-on Business Challenge: Customer Cohort Retention & Top-N per Group
"""

import sys
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


# Baseline datasets for joins and aggregations
transactions = pl.DataFrame(
    {
        "tx_id": [101, 102, 103, 104, 105, 106, 107, 108],
        "user_id": [1, 2, 1, 3, 2, 4, 1, 2],
        "category": ["Electronics", "Electronics", "Groceries", "Books", "Groceries", "Books", "Electronics", "Books"],
        "amount": [500.0, 120.0, 45.0, 30.0, 85.0, 22.0, 750.0, 65.0],
        "is_returned": [False, False, False, False, True, False, False, False],
    }
)

users = pl.DataFrame(
    {
        "user_id": [1, 2, 3, 5],  # user 4 is missing, user 5 has no transactions
        "name": ["Alice", "Bob", "Charlie", "Diana"],
        "country": ["IN", "US", "IN", "UK"],
        "plan": ["enterprise", "pro", "free", "pro"],
    }
)

print("Transactions Data:\n", transactions)
print("\nUsers Data:\n", users)


# ----------------------------------------------------------------------
# 4.1 Core Aggregations & group_by()
# ----------------------------------------------------------------------
section("4.1 Core Aggregations & group_by()")

# Grouping by category and computing multiple metrics in one parallel pass:
category_summary = transactions.group_by("category").agg(
    pl.len().alias("order_count"),
    pl.col("amount").sum().alias("total_revenue"),
    pl.col("amount").mean().round(2).alias("avg_order_value"),
    pl.col("amount").max().alias("max_single_sale"),
    pl.col("amount").std().round(2).alias("revenue_std_dev"),
).sort("total_revenue", descending=True)

print("Category Performance Summary:\n", category_summary)


# ----------------------------------------------------------------------
# 4.2 Filtered Aggregations (SQL FILTER WHERE Equivalent)
# ----------------------------------------------------------------------
section("4.2 Filtered Aggregations inside .agg()")

# Superpower of Polars: You can filter expressions directly INSIDE the aggregation,
# without needing separate subqueries or pre-filtering!
revenue_health = transactions.group_by("category").agg(
    pl.col("amount").sum().alias("gross_revenue"),
    pl.col("amount").filter(~pl.col("is_returned")).sum().alias("net_settled_revenue"),
    pl.col("amount").filter(pl.col("is_returned")).sum().fill_null(0.0).alias("refunded_revenue"),
    (pl.col("is_returned").sum() / pl.len()).round(4).alias("return_rate"),
)
print("Filtered Aggregations (Net vs Refunded):\n", revenue_health)


# ----------------------------------------------------------------------
# 4.3 Relational Joins: Inner, Left, Full, Cross
# ----------------------------------------------------------------------
section("4.3 Relational Joins (Inner, Left, Full, Cross)")

# Inner Join: Only matching user_id in both tables
inner_joined = transactions.join(users, on="user_id", how="inner")
print("Inner Join (user 4 omitted because missing from users):\n", inner_joined.select("tx_id", "user_id", "name", "country", "amount").head(4))

# Left Join: Keep all transactions, fill missing user info with null
left_joined = transactions.join(users, on="user_id", how="left")
print("\nLeft Join (user 4 included with null name & country):\n", left_joined.filter(pl.col("user_id") == 4).select("tx_id", "user_id", "name", "amount"))

# Full Outer Join: Keep all transactions and users (even user 5 who made 0 orders)
full_joined = transactions.join(users, on="user_id", how="full")
print("\nFull Join (includes Diana with null tx_id):\n", full_joined.filter(pl.col("user_id") == 5).select("tx_id", "user_id", "name", "plan"))


# ----------------------------------------------------------------------
# 4.4 Filtering Joins: Semi-Join and Anti-Join
# ----------------------------------------------------------------------
section("4.4 Filtering Joins (Semi-Join & Anti-Join)")

# Semi-Join: Filter users who HAVE made at least one purchase (without duplicating rows or joining transaction columns)
active_buyers = users.join(transactions, on="user_id", how="semi")
print("Semi-Join (Users who have purchased at least once):\n", active_buyers)

# Anti-Join: Find users who have NEVER made a purchase
churned_or_new_users = users.join(transactions, on="user_id", how="anti")
print("\nAnti-Join (Users with zero transactions):\n", churned_or_new_users)


# ----------------------------------------------------------------------
# 4.5 Concatenation: Vertical, Horizontal, Diagonal
# ----------------------------------------------------------------------
section("4.5 Concatenation (Vertical, Horizontal, Diagonal)")

df_q1 = pl.DataFrame({"quarter": ["Q1", "Q1"], "sales": [100, 150]})
df_q2 = pl.DataFrame({"quarter": ["Q2", "Q2"], "sales": [200, 220]})

# Vertical concatenation (UNION ALL)
combined_v = pl.concat([df_q1, df_q2], how="vertical")
print("Vertical Concat (UNION ALL):\n", combined_v)

# Diagonal concatenation (Schema unification when columns differ)
df_extra = pl.DataFrame({"quarter": ["Q3"], "sales": [310], "marketing_spend": [45]})
combined_diag = pl.concat([df_q1, df_extra], how="diagonal")
print("\nDiagonal Concat (Union with missing column filling nulls):\n", combined_diag)


# ----------------------------------------------------------------------
# 4.6 Reshaping: pivot() and unpivot() (Long <-> Wide)
# ----------------------------------------------------------------------
section("4.6 Reshaping with pivot() and unpivot()")

# Wide summary using pivot()
pivoted = transactions.pivot(
    index="category",
    on="user_id",
    values="amount",
    aggregate_function="sum",
).fill_null(0.0)
print("Pivoted Table (Category x User Matrix):\n", pivoted)

# Long format using unpivot()
unpivoted = pivoted.unpivot(
    index="category",
    variable_name="user_id",
    value_name="total_amount",
)
print("\nUnpivoted back to Long format:\n", unpivoted.head(4))


# ----------------------------------------------------------------------
# 4.7 Window Functions with .over() (SQL OVER PARTITION BY)
# ----------------------------------------------------------------------
section("4.7 Window Functions with .over()")

# Window functions allow computing group-level metrics WITHOUT collapsing individual rows!
# (SQL: amount / SUM(amount) OVER (PARTITION BY category))
windowed_df = transactions.with_columns(
    pl.col("amount").sum().over("category").alias("category_total"),
    (pl.col("amount") / pl.col("amount").sum().over("category") * 100).round(1).alias("category_share_pct"),
    pl.col("amount").mean().over("user_id").round(2).alias("user_avg_order"),
)
print("Window Function Results:\n", windowed_df.select("tx_id", "user_id", "category", "amount", "category_total", "category_share_pct"))


# ----------------------------------------------------------------------
# 4.8 Ranking, Cumulative Operations & Lags
# ----------------------------------------------------------------------
section("4.8 Ranking, Cumulative Sums, Lags & Shifts")

ranked_df = transactions.with_columns(
    pl.col("amount").rank(descending=True).over("category").alias("rank_in_category"),
    pl.col("amount").cum_sum().alias("running_total"),
    pl.col("amount").shift(1).alias("prev_tx_amount"),
).with_columns(
    (pl.col("amount") - pl.col("prev_tx_amount")).alias("diff_from_prev")
)

print("Rank and Running Total:\n", ranked_df.select("tx_id", "category", "amount", "rank_in_category", "running_total", "diff_from_prev"))


# ----------------------------------------------------------------------
# 4.9 Statistical Correlation & Covariance
# ----------------------------------------------------------------------
section("4.9 Statistical Correlation & Covariance")

corr_df = pl.DataFrame(
    {
        "ad_spend": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
        "organic_reach": [500.0, 750.0, 600.0, 900.0, 1100.0],
        "sales": [12000.0, 21000.0, 29000.0, 42000.0, 51000.0],
    }
)

corr_result = corr_df.select(
    pl.corr("ad_spend", "sales").alias("pearson_corr_spend_vs_sales"),
    pl.cov("ad_spend", "sales").alias("covariance"),
)
print("Correlation and Covariance Matrix:\n", corr_result)

print("\n Module 4 Completed Successfully!")
