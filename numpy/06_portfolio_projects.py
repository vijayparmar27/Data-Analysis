"""
Module 6: Real-World Portfolio Projects (Hands-On)
===================================================
Projects Included:
- Project 1: Sales & Revenue Analysis (Beginner)
    * Gross/Net revenue aggregations, monthly trends, top-selling products.
- Project 2: Customer Purchase Behavior & RFM Segmentation (Intermediate)
    * Recency, Frequency, Monetary value calculation, quantile scoring, customer tiers.
- Project 3: End-to-End Data Cleaning & Normalization Pipeline (Intermediate)
    * Sentinel handling, NaN median imputation, IQR outlier clipping, train/test split.
- Project 4: Statistical A/B Testing Engine (Intermediate)
    * Two-sample proportion test, standard error, Z-score, p-value, and confidence interval.
- Project 5: Inventory & Supply Demand Forecasting (Intermediate)
    * 7-day rolling moving averages (convolution), safety stock, Reorder Point (ROP).
- Project 6: Image Array Manipulation (Advanced)
    * 3D RGB ndarray (H, W, C), luminance grayscale, brightness/contrast filter, flip & crop.
- Project 7: Monte Carlo Business Risk Simulation (Advanced)
    * 10,000-iteration cash flow simulation, insolvency risk probability, and 95% Value-at-Risk (VaR).
"""

import math
import numpy as np


def project_header(title: str):
    print(f"\n{'#' * 70}\n  {title}\n{'#' * 70}")


# ======================================================================
# PROJECT 1: Sales & Revenue Analysis (Beginner)
# ======================================================================
def project_1_sales_analysis():
    project_header("PROJECT 1: Sales & Revenue Analysis")

    # Synthetic Transaction Table: 10 transactions
    # Columns: [TxID, Month (1-4), ProductID (101-105), Quantity, UnitPrice, DiscountPct]
    transactions = np.array([
        [1, 1, 101, 15, 25.0, 0.05],
        [2, 1, 102,  8, 80.0, 0.10],
        [3, 2, 101, 20, 25.0, 0.00],
        [4, 2, 103,  5, 150.0, 0.15],
        [5, 2, 104, 12, 45.0, 0.05],
        [6, 3, 102, 10, 80.0, 0.10],
        [7, 3, 105, 30, 15.0, 0.00],
        [8, 4, 103,  8, 150.0, 0.20],
        [9, 4, 104, 14, 45.0, 0.00],
        [10, 4, 105, 25, 15.0, 0.05],
    ])

    qty = transactions[:, 3]
    price = transactions[:, 4]
    discount = transactions[:, 5]

    # Vectorized Revenue Calculations:
    gross_revenue = qty * price
    discount_amount = gross_revenue * discount
    net_revenue = gross_revenue - discount_amount

    total_gross = np.sum(gross_revenue)
    total_net = np.sum(net_revenue)

    print(f"Total Transactions Processed: {len(transactions)}")
    print(f"Total Gross Revenue        : ${total_gross:,.2f}")
    print(f"Total Discounts Given       : ${np.sum(discount_amount):,.2f}")
    print(f"Total Net Revenue           : ${total_net:,.2f}")

    # Monthly Trend Analysis:
    months = np.unique(transactions[:, 1].astype(int))
    print("\n--- Monthly Performance Breakdown ---")
    print(f"{'Month':<8} {'Transactions':<14} {'Gross Revenue':<16} {'Net Revenue'}")
    print("-" * 52)
    for m in months:
        m_mask = transactions[:, 1] == m
        m_count = np.sum(m_mask)
        m_gross = np.sum(gross_revenue[m_mask])
        m_net = np.sum(net_revenue[m_mask])
        print(f"Month {m:<2} {m_count:<14} ${m_gross:<15.2f} ${m_net:<15.2f}")

    # Top-Performing Products by Net Revenue:
    product_ids = np.unique(transactions[:, 2].astype(int))
    prod_revenues = np.array([np.sum(net_revenue[transactions[:, 2] == p]) for p in product_ids])
    sort_idx = np.argsort(prod_revenues)[::-1]

    print("\n--- Top Products Ranked by Net Revenue ---")
    for rank, idx in enumerate(sort_idx, start=1):
        print(f"#{rank} Product {product_ids[idx]}: ${prod_revenues[idx]:,.2f}")


# ======================================================================
# PROJECT 2: Customer Purchase Behavior & Segmentation (RFM Analytics)
# ======================================================================
def project_2_rfm_segmentation():
    project_header("PROJECT 2: Customer RFM Segmentation Engine")

    # Order History: [OrderID, CustomerID, DaysAgo, OrderValue]
    orders = np.array([
        [1001, 1,  5,  240.0],
        [1002, 2, 45,   80.0],
        [1003, 1, 18,  310.0],
        [1004, 3, 90,   50.0],
        [1005, 4,  2, 1200.0],
        [1006, 2, 60,   95.0],
        [1007, 4, 12,  850.0],
        [1008, 4, 25,  600.0],
        [1009, 5,  8,  150.0],
        [1010, 5, 14,  180.0],
        [1011, 3, 120,  40.0],
    ])

    unique_custs = np.unique(orders[:, 1].astype(int))
    n_custs = len(unique_custs)

    # Compute R, F, M vectors
    recency = np.empty(n_custs)
    frequency = np.empty(n_custs)
    monetary = np.empty(n_custs)

    for i, cid in enumerate(unique_custs):
        mask = orders[:, 1] == cid
        recency[i] = np.min(orders[mask, 2])          # Minimum days ago = most recent
        frequency[i] = np.sum(mask)                   # Number of orders
        monetary[i] = np.sum(orders[mask, 3])         # Total spend

    # Quantile thresholds (Median split for demonstration: High vs Low)
    r_median = np.median(recency)
    f_median = np.median(frequency)
    m_median = np.median(monetary)

    # Recency: Lower days ago is BETTER (R_score: True if <= median)
    r_score = recency <= r_median
    f_score = frequency >= f_median
    m_score = monetary >= m_median

    # Multi-condition segmentation using np.select
    conditions = [
        r_score & f_score & m_score,                  # Best on all 3: Champions
        r_score & (f_score | m_score),                # Recent + active: Loyal / Promising
        (~r_score) & (f_score | m_score),             # High historical value but inactive: At Risk
    ]
    labels = ["Champion", "Loyal / Active", "At Risk"]
    segments = np.select(conditions, labels, default="Lost / Hibernating")

    print(f"{'CustID':<8} {'Recency (Days)':<16} {'Frequency':<12} {'Total Spend':<14} {'Segment'}")
    print("-" * 65)
    for i in range(n_custs):
        print(f"User {unique_custs[i]:<3} {recency[i]:<16.0f} {frequency[i]:<12.0f} ${monetary[i]:<13.2f} {segments[i]}")


# ======================================================================
# PROJECT 3: CSV Data Cleaning & Normalization Pipeline (Intermediate)
# ======================================================================
def project_3_data_cleaning_pipeline():
    project_header("PROJECT 3: CSV Data Cleaning & Normalization Pipeline")

    # Raw dirty dataset representing tabular features [Age, Income, CreditScore, LoanAmount]
    # Contains: Sentinel missing values (-999.0), standard NaNs, and extreme outliers
    raw_data = np.array([
        [25.0,  45000.0,  680.0,  12000.0],
        [38.0, -999.0,    710.0,  25000.0],  # Sentinel in Income
        [45.0,  85000.0,  np.nan, 40000.0],  # NaN in CreditScore
        [29.0,  52000.0,  690.0,  15000.0],
        [54.0, 110000.0,  790.0, 850000.0],  # Extreme outlier loan amount
        [33.0,  61000.0,  720.0,  18000.0],
        [62.0,  92000.0,  750.0,  30000.0],
        [22.0,  31000.0,  640.0,   8000.0],
    ])

    print(f"Original Raw Matrix shape: {raw_data.shape}")

    # Stage 1: Convert sentinel codes (-999.0) into IEEE NaNs
    data = np.where(raw_data == -999.0, np.nan, raw_data)
    print(f"Stage 1 - Converted sentinels to NaN. Total NaNs: {np.sum(np.isnan(data))}")

    # Stage 2: NaN-Safe Column Median Imputation
    col_medians = np.nanmedian(data, axis=0)
    for col in range(data.shape[1]):
        nan_mask = np.isnan(data[:, col])
        data[nan_mask, col] = col_medians[col]
    print(f"Stage 2 - Imputed missing values with column medians: {np.round(col_medians, 1)}")

    # Stage 3: Outlier Capping (Winsorization) using 1.5 * IQR per column
    q25 = np.percentile(data, 25, axis=0)
    q75 = np.percentile(data, 75, axis=0)
    iqr = q75 - q25
    lower_limits = q25 - 1.5 * iqr
    upper_limits = q75 + 1.5 * iqr

    data_clipped = np.clip(data, lower_limits, upper_limits)
    print(f"Stage 3 - Outliers clipped. Loan amount outlier 850,000 capped to: ${data_clipped[4, 3]:,.2f}")

    # Stage 4: Feature Scaling (Z-Score Standard Scaling)
    # Scaled = (X - Mean) / Std
    means = data_clipped.mean(axis=0, keepdims=True)
    stds = data_clipped.std(axis=0, keepdims=True)
    standardized = (data_clipped - means) / stds

    # Stage 5: Train / Test Split (75% Train, 25% Test) without data leakage
    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(len(standardized))
    split_point = int(0.75 * len(standardized))
    train_idx, test_idx = indices[:split_point], indices[split_point:]

    X_train = standardized[train_idx]
    X_test = standardized[test_idx]

    print(f"Stage 4 & 5 - Train Set shape: {X_train.shape} | Test Set shape: {X_test.shape}")
    print("Cleaned & Standardized Train Sample (First 2 rows):\n", np.round(X_train[:2], 3))


# ======================================================================
# PROJECT 4: Statistical A/B Testing Engine (Intermediate)
# ======================================================================
def project_4_ab_testing_engine():
    project_header("PROJECT 4: Statistical A/B Testing Engine")

    # Experiment: New Checkout Page (Variant B) vs Control (Variant A)
    # Control A
    visitors_A = 12_500
    conversions_A = 1_200   # 9.60% conversion rate

    # Treatment B
    visitors_B = 12_500
    conversions_B = 1_375   # 11.00% conversion rate

    p_A = conversions_A / visitors_A
    p_B = conversions_B / visitors_B
    observed_uplift = (p_B - p_A) / p_A

    # Pooled Probability for Two-Sample Proportion Test:
    p_pool = (conversions_A + conversions_B) / (visitors_A + visitors_B)

    # Standard Error (SE) under Null Hypothesis H0 (p_A == p_B):
    se_pool = np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / visitors_A + 1.0 / visitors_B))

    # Test Statistic: Z-Score
    z_score = (p_B - p_A) / se_pool

    # Two-Tailed P-Value calculation using standard normal approximation:
    # Vectorized math via error function math.erf: CDF(z) = 0.5 * (1 + erf(z / sqrt(2)))
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_score) / np.sqrt(2.0))))

    # 95% Confidence Interval for Absolute Difference (p_B - p_A):
    se_diff = np.sqrt((p_A * (1 - p_A) / visitors_A) + (p_B * (1 - p_B) / visitors_B))
    ci_lower = (p_B - p_A) - 1.96 * se_diff
    ci_upper = (p_B - p_A) + 1.96 * se_diff

    print(f"Variant A (Control)   : {conversions_A:,} / {visitors_A:,} ({p_A * 100:.2f}%)")
    print(f"Variant B (Treatment) : {conversions_B:,} / {visitors_B:,} ({p_B * 100:.2f}%)")
    print(f"Observed Relative Uplift: +{observed_uplift * 100:.2f}%")
    print(f"Z-Score               : {z_score:.4f}")
    print(f"P-Value               : {p_value:.6f}")
    print(f"95% Confidence Interval: [{ci_lower * 100:+.2f}%, {ci_upper * 100:+.2f}%]")

    # Decision Logic:
    alpha = 0.05
    if p_value < alpha:
        print(f"✅ RESULT: Statistically Significant (p < {alpha}). Safe to deploy Variant B!")
    else:
        print(f"❌ RESULT: Inconclusive (p >= {alpha}). Insufficient evidence to reject H0.")


# ======================================================================
# PROJECT 5: Inventory & Supply Demand Forecasting (Intermediate)
# ======================================================================
def project_5_inventory_forecasting():
    project_header("PROJECT 5: Inventory & Supply Demand Forecasting")

    # Simulate 30 days of product customer demand
    rng = np.random.default_rng(seed=101)
    daily_demand = rng.poisson(lam=45, size=30)  # average 45 units/day

    # 1. 7-Day Moving Average using 1D Convolution (np.convolve)
    # A rolling 7-day average filter is simply a uniform kernel of size 7: [1/7, ..., 1/7]
    window = 7
    kernel = np.ones(window) / window
    # mode='valid' computes the convolution only where the window completely overlaps
    moving_avg_7d = np.convolve(daily_demand, kernel, mode='valid')

    print(f"Daily Demand Sample (Days 1..10): {daily_demand[:10]}")
    print(f"7-Day Moving Averages (Days 7..12) : {np.round(moving_avg_7d[:6], 1)}")

    # 2. Safety Stock & Reorder Point (ROP) Calculation
    # Supplier Lead Time (L) = 5 days
    lead_time_days = 5
    avg_daily_demand = np.mean(daily_demand)
    std_daily_demand = np.std(daily_demand, ddof=1)

    # Lead time demand = Average Demand * Lead Time
    lead_time_demand = avg_daily_demand * lead_time_days

    # Safety Stock formula for 95% service level (Z = 1.65):
    # Safety Stock = Z * sqrt(LeadTime) * StdDailyDemand
    z_service_level = 1.65
    safety_stock = z_service_level * np.sqrt(lead_time_days) * std_daily_demand

    # Reorder Point (ROP): Place a supplier order when stock drops to this level!
    reorder_point = lead_time_demand + safety_stock

    print("\n--- Inventory Replenishment Policy ---")
    print(f"Average Daily Demand     : {avg_daily_demand:.1f} units")
    print(f"Daily Demand Volatility  : ±{std_daily_demand:.2f} units")
    print(f"Expected Lead Time Demand: {lead_time_demand:.1f} units (over {lead_time_days} days)")
    print(f"Required Safety Stock    : {math.ceil(safety_stock)} units (Buffer against stockout)")
    print(f"🎯 Recommended Reorder Pt: {math.ceil(reorder_point)} units")


# ======================================================================
# PROJECT 6: Image Array Manipulation (Advanced)
# ======================================================================
def project_6_image_array_manipulation():
    project_header("PROJECT 6: Image Array Manipulation (3D RGB ndarray)")

    # Synthesize a 3D RGB Image Tensor: shape (Height=6, Width=8, Channels=3)
    # In NumPy/Computer Vision:
    # - Axis 0: Rows / Height
    # - Axis 1: Columns / Width
    # - Axis 2: Color Channels (Red, Green, Blue) with dtype=uint8 (0 to 255)
    H, W = 6, 8
    img = np.zeros((H, W, 3), dtype=np.uint8)

    # Create distinct color quadrants:
    img[:3, :4] = [255, 0, 0]      # Top-Left: Bright Red
    img[:3, 4:] = [0, 255, 0]      # Top-Right: Bright Green
    img[3:, :4] = [0, 0, 255]      # Bottom-Left: Bright Blue
    img[3:, 4:] = [255, 255, 0]    # Bottom-Right: Yellow (Red + Green)

    print(f"RGB Image Tensor shape: {img.shape} | dtype: {img.dtype} | nbytes: {img.nbytes} bytes")

    # 1. Grayscale Conversion (Luminance formula: Y = 0.2989 R + 0.5870 G + 0.1140 B)
    luminance_weights = np.array([0.2989, 0.5870, 0.1140])
    # Dot product along color channel axis (axis 2):
    grayscale = np.dot(img, luminance_weights).astype(np.uint8)
    print(f"\n1. Converted to Grayscale shape {grayscale.shape}:\n{grayscale}")

    # 2. Brightness & Contrast Adjustment
    # Formula: New_Img = clip(alpha * Img + beta, 0, 255)
    alpha = 1.2   # Contrast multiplier (+20%)
    beta = 25     # Brightness offset (+25)
    adjusted = np.clip(alpha * img.astype(float) + beta, 0, 255).astype(np.uint8)
    print(f"\n2. Contrast/Brightness adjusted (Top-left Red pixel: {img[0, 0]} -> {adjusted[0, 0]})")

    # 3. Flips and Slicing Transformations:
    horizontal_flip = img[:, ::-1]  # Reverse columns
    vertical_flip = img[::-1, :]    # Reverse rows
    center_crop = img[2:4, 2:6]     # Height 2..3, Width 2..5

    print(f"3. Original Top-Left pixel : {img[0, 0]}")
    print(f"   H-Flipped Top-Left pixel: {horizontal_flip[0, 0]} (now green from top-right!)")
    print(f"   Center crop shape       : {center_crop.shape}")


# ======================================================================
# PROJECT 7: Monte Carlo Business Simulation (Advanced)
# ======================================================================
def project_7_monte_carlo_simulation():
    project_header("PROJECT 7: Monte Carlo Business Risk Simulation")

    # Scenario: 10,000 parallel 12-month financial trajectories for a SaaS startup.
    # Initial Cash Reserves: $500,000
    N_SIMULATIONS = 10_000
    MONTHS = 12
    initial_cash = 500_000.0

    rng = np.random.default_rng(seed=2026)

    # Vectorized random variables across all 10,000 simulations x 12 months:
    # 1. Monthly Revenue Growth: Normal(mean=5%, std=3%)
    monthly_growth = rng.normal(loc=0.05, scale=0.03, size=(N_SIMULATIONS, MONTHS))

    # Base month-0 revenue = $80,000
    # Cumulative revenue multiplier across time: cumprod(1 + growth)
    revenue_trajectories = 80_000.0 * np.cumprod(1.0 + monthly_growth, axis=1)

    # 2. Monthly Operating Expenses: Normal(mean=$90,000, std=$8,000)
    operating_expenses = rng.normal(loc=90_000.0, scale=8_000.0, size=(N_SIMULATIONS, MONTHS))

    # 3. Rare Unexpected Legal/Server Outage Event: Poisson(λ = 0.1 incidents/mo) * $30,000 cost
    incidents = rng.poisson(lam=0.1, size=(N_SIMULATIONS, MONTHS))
    unexpected_costs = incidents * 30_000.0

    # Total Monthly Net Cash Flow = Revenue - Operating Expenses - Unexpected Costs
    monthly_cash_flows = revenue_trajectories - operating_expenses - unexpected_costs

    # Cumulative cash balance over the 12 months: initial_cash + cumsum(monthly_cash_flows)
    cash_balances = initial_cash + np.cumsum(monthly_cash_flows, axis=1)

    ending_balances = cash_balances[:, -1]
    minimum_cash_reached = np.min(cash_balances, axis=1)

    # 1. Probability of Insolvency (Cash balance ever dropped below $0):
    insolvent_runs = np.sum(minimum_cash_reached < 0)
    insolvency_probability = (insolvent_runs / N_SIMULATIONS) * 100.0

    # 2. Key Summary Financial Metrics:
    mean_ending_cash = np.mean(ending_balances)
    median_ending_cash = np.median(ending_balances)

    # 3. Value-at-Risk (VaR) at 95% Confidence (5th percentile):
    var_95_ending_cash = np.percentile(ending_balances, 5.0)
    var_99_ending_cash = np.percentile(ending_balances, 1.0)

    print(f"Simulations Executed       : {N_SIMULATIONS:,} trajectories across {MONTHS} months")
    print(f"Initial Starting Cash      : ${initial_cash:,.2f}")
    print(f"Mean Ending Cash at Month 12: ${mean_ending_cash:,.2f}")
    print(f"Median Ending Cash         : ${median_ending_cash:,.2f}")
    print(f"\n--- Risk Analysis ---")
    print(f"🚨 Probability of Cash Insolvency: {insolvency_probability:.2f}% ({insolvent_runs:,} / {N_SIMULATIONS:,})")
    print(f"📉 95% Value-at-Risk (5th %ile) : ${var_95_ending_cash:,.2f}")
    print(f"📉 99% Value-at-Risk (1st %ile) : ${var_99_ending_cash:,.2f}")

    if insolvency_probability < 5.0:
        print("💡 Financial Health: STRONG (< 5% risk of default).")
    else:
        print("⚠️ Financial Health: AT RISK (> 5% default probability). Capital raise recommended.")


# ======================================================================
# MASTER EXECUTION ENTRYPOINT
# ======================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  NUMPY MODULE 6: REAL-WORLD PORTFOLIO PROJECTS MASTER RUNNER")
    print("=" * 70)

    project_1_sales_analysis()
    project_2_rfm_segmentation()
    project_3_data_cleaning_pipeline()
    project_4_ab_testing_engine()
    project_5_inventory_forecasting()
    project_6_image_array_manipulation()
    project_7_monte_carlo_simulation()

    print("\n" + "=" * 70)
    print("  🎉 ALL 7 PORTFOLIO PROJECTS SUCCESSFULLY EXECUTED!")
    print("=" * 70 + "\n")
