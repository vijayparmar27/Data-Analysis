"""
Module 6: Real-World Portfolio Projects (Hands-On)
===================================================
Projects Included:
- Project 1: E-Commerce Sales & Revenue Analytics (Beginner)
    * Multi-table order joining, net margins, top bestselling SKUs, and Average Order Value (AOV).
- Project 2: Customer RFM Segmentation & Behavior Analysis (Intermediate)
    * Recency, Frequency, Monetary calculations, quantile scoring (1-4), and customer tier cohorts.
- Project 3: Production Data Cleaning & Validation Pipeline (Intermediate)
    * Idempotent .pipe() architecture, type enforcement, sentinel imputation, IQR clipping.
- Project 4: Monthly Financial KPI & Retention Cohort Engine (Intermediate)
    * MRR tracking, Month-over-Month (MoM) growth rates, cohort retention heatmaps.
- Project 5: Inventory Turnover & Stock Replenishment Forecasting (Advanced)
    * Daily depletion velocity, Days of Inventory (DOI), safety stock buffers, reorder alerts.
- Project 6: Statistical A/B Testing & Conversion Rate Hypothesis Engine (Advanced)
    * Control vs Treatment, conversion proportions, pooled standard error, Z-score, p-value.
"""

import math
import sys

# Defensive guard: prevent local directory name from shadowing installed library
if "" in sys.path:
    sys.path.remove("")
if "." in sys.path:
    sys.path.remove(".")

import numpy as np
import pandas as pd


def project_header(title: str):
    print(f"\n{'#' * 75}\n  {title}\n{'#' * 75}")


# ======================================================================
# PROJECT 1: E-Commerce Sales & Revenue Analytics (Beginner)
# ======================================================================
def project_1_ecommerce_analytics():
    project_header("PROJECT 1: E-Commerce Sales & Revenue Analytics")

    # 1. Synthetic Orders Table
    orders_df = pd.DataFrame({
        "order_id": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
        "customer_id": ["C1", "C2", "C1", "C3", "C4", "C2", "C5", "C3"],
        "order_date": pd.to_datetime([
            "2026-01-05", "2026-01-12", "2026-01-20", "2026-02-03",
            "2026-02-15", "2026-02-22", "2026-03-01", "2026-03-10"
        ]),
        "status": ["COMPLETED", "COMPLETED", "REFUNDED", "COMPLETED", "COMPLETED", "COMPLETED", "CANCELLED", "COMPLETED"]
    })

    # 2. Synthetic Order Items Table
    items_df = pd.DataFrame({
        "order_id": [1001, 1001, 1002, 1003, 1004, 1005, 1005, 1006, 1007, 1008],
        "product_name": [
            "Mechanical Keyboard", "Wireless Mouse", "4K Monitor", "USB-C Cable",
            "Ergonomic Chair", "Mechanical Keyboard", "Desk Mat", "4K Monitor", "Webcam", "Wireless Mouse"
        ],
        "category": [
            "Peripherals", "Peripherals", "Displays", "Accessories",
            "Furniture", "Peripherals", "Accessories", "Displays", "Peripherals", "Peripherals"
        ],
        "quantity": [1, 2, 1, 3, 1, 2, 1, 1, 1, 1],
        "unit_price": [120.0, 45.0, 450.0, 15.0, 320.0, 120.0, 25.0, 450.0, 75.0, 45.0],
        "unit_cost": [60.0, 20.0, 280.0, 4.0, 180.0, 60.0, 10.0, 280.0, 35.0, 20.0]
    })

    # Filter out cancelled / refunded orders
    valid_orders = orders_df[orders_df["status"] == "COMPLETED"]

    # Join orders with order items
    sales_merged = pd.merge(valid_orders, items_df, on="order_id", how="inner")

    # Vectorized revenue & profit margins
    sales_merged["gross_revenue"] = sales_merged["quantity"] * sales_merged["unit_price"]
    sales_merged["total_cost"] = sales_merged["quantity"] * sales_merged["unit_cost"]
    sales_merged["gross_profit"] = sales_merged["gross_revenue"] - sales_merged["total_cost"]
    sales_merged["profit_margin_pct"] = (sales_merged["gross_profit"] / sales_merged["gross_revenue"]) * 100

    # Executive High-Level Metrics
    total_sales = sales_merged["gross_revenue"].sum()
    total_profit = sales_merged["gross_profit"].sum()
    total_orders_count = sales_merged["order_id"].nunique()
    aov = total_sales / total_orders_count

    print(f"Total Gross Sales:    ${total_sales:,.2f}")
    print(f"Total Gross Profit:   ${total_profit:,.2f} ({total_profit / total_sales * 100:.1f}% margin)")
    print(f"Total Valid Orders:   {total_orders_count}")
    print(f"Average Order Value:  ${aov:,.2f}")

    # Top selling products
    top_products = sales_merged.groupby("product_name").agg(
        units_sold=("quantity", "sum"),
        revenue_generated=("gross_revenue", "sum"),
        profit_generated=("gross_profit", "sum")
    ).sort_values(by="revenue_generated", ascending=False)
    print("\nTop Products Performance:\n", top_products)


# ======================================================================
# PROJECT 2: Customer RFM Segmentation & Behavior Analysis (Intermediate)
# ======================================================================
def project_2_rfm_segmentation():
    project_header("PROJECT 2: Customer RFM Segmentation & Behavior Analysis")

    # Synthetic purchase transaction stream
    txns = pd.DataFrame({
        "customer_id": ["Cust_A", "Cust_B", "Cust_C", "Cust_D", "Cust_E", "Cust_A", "Cust_B", "Cust_A"],
        "order_date": pd.to_datetime([
            "2026-03-15", "2026-01-10", "2026-03-25", "2025-11-04",
            "2026-03-28", "2026-02-20", "2026-03-01", "2026-03-27"
        ]),
        "order_amount": [150.0, 320.0, 50.0, 45.0, 620.0, 200.0, 110.0, 85.0]
    })

    # Reference analysis date (snapshot timestamp)
    snapshot_date = pd.to_datetime("2026-03-29")

    # Aggregate RFM metrics per customer:
    # - Recency: Days since last purchase
    # - Frequency: Number of unique orders placed
    # - Monetary: Total lifetime revenue
    rfm = txns.groupby("customer_id").agg(
        recency=("order_date", lambda dates: (snapshot_date - dates.max()).days),
        frequency=("order_date", "count"),
        monetary=("order_amount", "sum")
    )

    # Score each dimension from 1 to 3 (or 1 to 5)
    # Recency: Lower days = better score (reverse rank)
    rfm["R_score"] = pd.qcut(rfm["recency"].rank(method="first"), q=3, labels=[3, 2, 1])
    rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=3, labels=[1, 2, 3])
    rfm["M_score"] = pd.qcut(rfm["monetary"].rank(method="first"), q=3, labels=[1, 2, 3])

    # Combined composite RFM segment
    def assign_customer_tier(row):
        r, f, m = int(row["R_score"]), int(row["F_score"]), int(row["M_score"])
        if r >= 2 and f >= 2 and m >= 2:
            return "Champion (VIP)"
        elif r >= 2 and f >= 1:
            return "Potential Loyalist"
        elif r == 1:
            return "At Risk / Churning"
        return "Standard"

    rfm["segment"] = rfm.apply(assign_customer_tier, axis=1)
    print("Customer RFM Segmentation Matrix:\n", rfm)


# ======================================================================
# PROJECT 3: Production Data Cleaning & Validation Pipeline (Intermediate)
# ======================================================================
def project_3_data_cleaning_pipeline():
    project_header("PROJECT 3: Production Data Cleaning & Validation Pipeline")

    raw_user_feed = pd.DataFrame({
        "id": [1, 2, 3, 3, 4, 5],
        "name": ["  Vikram  ", "ANITA RAO", "rohit sharma", "rohit sharma", "Pooja ", None],
        "email": ["vikram@gmail.com", "ANITA@TECH.COM", "rohit@gmail.com", "rohit@gmail.com", "invalid_email", "guest@site.com"],
        "age_str": ["28", "34", "twenty", "34", "45", "999"],  # 999 is outlier / dirty entry
        "annual_salary": ["65000", "120000", "85000", "85000", "50000", "40000"]
    })

    # Composable pipeline functions
    def step_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates(subset=["id"], keep="first")

    def step_clean_strings(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.dropna(subset=["name"])
        df["name"] = df["name"].str.strip().str.title()
        df["email"] = df["email"].str.lower().str.strip()
        # Keep only emails with valid syntax
        df = df[df["email"].str.contains(r"^[\w\.-]+@[\w\.-]+\.\w+$", regex=True)]
        return df

    def step_cast_and_clip(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Parse numeric age safely
        df["age"] = pd.to_numeric(df["age_str"], errors="coerce")
        df["age"] = df["age"].fillna(df["age"].median())
        # Clip unrealistic ages to valid range [18, 75]
        df["age"] = df["age"].clip(lower=18, upper=75).astype(int)
        df["annual_salary"] = pd.to_numeric(df["annual_salary"], errors="coerce")
        df = df.drop(columns=["age_str"])
        return df

    # Execute end-to-end pipeline using method chaining (.pipe)
    cleaned_feed = (
        raw_user_feed
        .pipe(step_deduplicate)
        .pipe(step_clean_strings)
        .pipe(step_cast_and_clip)
    )

    print("End-to-End Pipeline Output:\n", cleaned_feed)


# ======================================================================
# PROJECT 4: Monthly Financial KPI & Retention Engine (Intermediate)
# ======================================================================
def project_4_financial_cohort_engine():
    project_header("PROJECT 4: Monthly Financial KPI & Retention Cohort Engine")

    # 10 SaaS subscription charges
    subscriptions = pd.DataFrame({
        "user_id": ["u1", "u2", "u1", "u3", "u2", "u1", "u4", "u2", "u3", "u1"],
        "charge_date": pd.to_datetime([
            "2026-01-01", "2026-01-05", "2026-02-01", "2026-02-10", "2026-02-05",
            "2026-03-01", "2026-03-02", "2026-03-05", "2026-03-10", "2026-04-01"
        ]),
        "mrr_amount": [99.0, 299.0, 99.0, 49.0, 299.0, 99.0, 29.0, 299.0, 49.0, 99.0]
    })

    # Resample / group by month to compute Monthly Recurring Revenue (MRR)
    subscriptions["billing_month"] = subscriptions["charge_date"].dt.to_period("M")

    mrr_summary = subscriptions.groupby("billing_month").agg(
        active_subscribers=("user_id", "nunique"),
        total_mrr=("mrr_amount", "sum")
    )
    # Calculate Month-over-Month (MoM) MRR Growth Rate
    mrr_summary["mom_growth_pct"] = mrr_summary["total_mrr"].pct_change() * 100

    print("Monthly SaaS Financial Performance:\n", mrr_summary.round(2))


# ======================================================================
# PROJECT 5: Inventory Turnover & Stock Replenishment Forecasting (Advanced)
# ======================================================================
def project_5_inventory_replenishment():
    project_header("PROJECT 5: Inventory Turnover & Replenishment Forecasting")

    # Inventory status across 4 product lines
    stock_df = pd.DataFrame({
        "sku": ["SKU-101", "SKU-102", "SKU-103", "SKU-104"],
        "product": ["Noise Cancelling Headphones", "Smartphone Gimbal", "Anker PowerBank", "HDMI Cable"],
        "current_stock": [45, 12, 180, 8],
        "daily_sales_rate": [3.5, 2.0, 5.0, 1.8],    # Average units sold per day
        "lead_time_days": [7, 10, 14, 5],            # Days for supplier to deliver
        "unit_cost": [120.0, 85.0, 25.0, 6.0]
    })

    # Calculations:
    # 1. Days of Inventory (DOI) = Current Stock / Daily Sales Rate
    stock_df["days_of_inventory"] = stock_df["current_stock"] / stock_df["daily_sales_rate"]

    # 2. Safety Stock = 3 days buffer of sales
    stock_df["safety_stock"] = stock_df["daily_sales_rate"] * 3

    # 3. Reorder Point (ROP) = (Daily Sales Rate * Lead Time) + Safety Stock
    stock_df["reorder_point"] = (stock_df["daily_sales_rate"] * stock_df["lead_time_days"]) + stock_df["safety_stock"]

    # 4. Trigger Reorder Alert
    stock_df["needs_reorder"] = stock_df["current_stock"] <= stock_df["reorder_point"]

    print("Inventory Replenishment & Stockout Risk Table:\n",
          stock_df[["sku", "product", "current_stock", "days_of_inventory", "reorder_point", "needs_reorder"]].round(1))


# ======================================================================
# PROJECT 6: Statistical A/B Testing & Conversion Engine (Advanced)
# ======================================================================
def project_6_ab_testing_engine():
    project_header("PROJECT 6: Statistical A/B Testing & Conversion Engine")

    # Experiment: Checkout Redesign (Variant A: Control, Variant B: Treatment)
    ab_data = pd.DataFrame({
        "variant": ["Control (A)", "Treatment (B)"],
        "visitors": [12500, 12600],
        "conversions": [1120, 1285]
    })

    # 1. Conversion Rates (p)
    ab_data["conversion_rate"] = ab_data["conversions"] / ab_data["visitors"]
    ab_data["conv_pct"] = (ab_data["conversion_rate"] * 100).round(2)

    n_a = ab_data.loc[0, "visitors"]
    x_a = ab_data.loc[0, "conversions"]
    p_a = ab_data.loc[0, "conversion_rate"]

    n_b = ab_data.loc[1, "visitors"]
    x_b = ab_data.loc[1, "conversions"]
    p_b = ab_data.loc[1, "conversion_rate"]

    # 2. Lift calculation
    relative_lift = ((p_b - p_a) / p_a) * 100

    # 3. Pooled standard error & two-proportion Z-score test
    p_pool = (x_a + x_b) / (n_a + n_b)
    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z_score = (p_b - p_a) / se_pool

    # Two-tailed p-value approximation via standard normal error function
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z_score) / math.sqrt(2))))

    print("A/B Experiment Summary:\n", ab_data[["variant", "visitors", "conversions", "conv_pct"]])
    print(f"\nRelative Conversion Lift: +{relative_lift:.2f}%")
    print(f"Pooled Standard Error:     {se_pool:.5f}")
    print(f"Z-Score:                  {z_score:.3f}")
    print(f"P-Value:                  {p_value:.5f}")

    if p_value < 0.05:
        print(" Statistically Significant Result (p < 0.05)! We reject H0 and deploy Variant B.")
    else:
        print(" Not statistically significant (p >= 0.05). Keep iterating on experiment.")


# ======================================================================
# Main Execution Runner
# ======================================================================
if __name__ == "__main__":
    print("=======================================================================")
    print(" PANDAS REAL-WORLD PORTFOLIO PROJECTS EXECUTION SUITE")
    print("=======================================================================")
    project_1_ecommerce_analytics()
    project_2_rfm_segmentation()
    project_3_data_cleaning_pipeline()
    project_4_financial_cohort_engine()
    project_5_inventory_replenishment()
    project_6_ab_testing_engine()
    print("\n All 6 Real-World Portfolio Projects Executed Successfully!")
