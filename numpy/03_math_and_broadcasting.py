"""
Module 3: Mathematical Operations & Broadcasting
================================================
Topics Covered:
- 3.1 Element-wise arithmetic operations (+, -, *, /, //, **, %)
- 3.2 Universal functions (ufuncs): sqrt, exp, log, abs, sin, cos
- 3.3 Basic aggregations: sum, prod, min, max, ptp
- 3.4 Statistical aggregations: mean, median, var, std, percentiles (ddof explained)
- 3.5 Directional operations with axis (axis=0 columns vs axis=1 rows) & keepdims
- 3.6 Broadcasting rules & dimension compatibility explained visually
- 3.7 Broadcasting in practice (Z-score normalization, Min-Max scaling)
- 3.8 Comparison & logical operations: all, any, isclose, allclose
- 3.9 Numerical rounding: round, floor, ceil, trunc
- 3.10 Cumulative operations: cumsum, cumprod, diff
- 3.11 Comprehensive Hands-on Practice Challenge & Solution (Financial Portfolio Analysis)
"""

import numpy as np


def section(title: str):
    print(f"\n{'=' * 65}\n  {title}\n{'=' * 65}")


# ----------------------------------------------------------------------
# 3.1 Element-wise Arithmetic Operations
# ----------------------------------------------------------------------
section("3.1 Element-Wise Arithmetic Operations")

a = np.array([10, 20, 30, 40])
b = np.array([2, 4, 5, 8])

print(f"Array a: {a}")
print(f"Array b: {b}")

# Standard operators map directly to vectorized C-level ufuncs:
print(f"a + b  (Addition)       : {a + b}")        # np.add(a, b)
print(f"a - b  (Subtraction)    : {a - b}")        # np.subtract(a, b)
print(f"a * b  (Multiplication) : {a * b}")        # np.multiply(a, b)
print(f"a / b  (Float Division) : {a / b}")        # np.divide(a, b)
print(f"a // b (Floor Division) : {a // b}")       # np.floor_divide(a, b)
print(f"a ** 2 (Power / Square) : {a ** 2}")       # np.power(a, 2)
print(f"a % b  (Modulo)         : {a % b}")        # np.mod(a, b)

# Arithmetic with scalars (broadcasts the scalar automatically to every element):
print(f"a * 1.05 (+5% markup)   : {a * 1.05}")


# ----------------------------------------------------------------------
# 3.2 Universal Functions (ufuncs)
# ----------------------------------------------------------------------
section("3.2 Universal Functions (ufuncs)")

# ufuncs are compiled C functions that execute element-by-element loops at CPU speed
vals = np.array([1, 4, 9, 16, 25], dtype=float)
print(f"Values: {vals}")

print(f"np.sqrt(vals) : {np.sqrt(vals)}")
print(f"np.exp(vals)  : {np.exp(np.array([0, 1, 2]))}")          # e^x
print(f"np.log(vals)  : {np.log(vals)}")                         # Natural log (ln)
print(f"np.log10(vals): {np.log10(np.array([1, 10, 100, 1000]))}")

# Absolute values & trigonometry
mixed = np.array([-3.5, 4.2, -8.9, 0.0])
print(f"\nnp.abs({mixed}) : {np.abs(mixed)}")

angles = np.array([0, np.pi / 4, np.pi / 2, np.pi])
print(f"np.sin([0, π/4, π/2, π]): {np.round(np.sin(angles), 4)}")


# ----------------------------------------------------------------------
# 3.3 Basic Aggregations: sum, prod, min, max, ptp
# ----------------------------------------------------------------------
section("3.3 Basic Aggregations")

numbers = np.array([12, 45, 7, 89, 23, 56])
print(f"Numbers: {numbers}")

# You can call as numpy function np.func(arr) or method arr.func()
print(f"Sum         : {numbers.sum()}  (or np.sum)")
print(f"Product     : {np.prod(np.array([1, 2, 3, 4, 5]))}")
print(f"Minimum     : {numbers.min()} at index {numbers.argmin()}")
print(f"Maximum     : {numbers.max()} at index {numbers.argmax()}")
print(f"Peak-to-Peak: {np.ptp(numbers)} (Range: max - min = 89 - 7 = 82)")


# ----------------------------------------------------------------------
# 3.4 Statistical Aggregations & Quantiles
# ----------------------------------------------------------------------
section("3.4 Statistical Aggregations (mean, median, var, std, quantiles)")

salaries = np.array([45000, 52000, 61000, 58000, 120000, 49000, 55000])
print(f"Salaries: {salaries}")

print(f"Mean (Average)      : ${salaries.mean():,.2f}")
print(f"Median (50th %ile)  : ${np.median(salaries):,.2f}  (Robust to outlier $120k!)")

# Variance and Standard Deviation
# ddof=0: Population (default in NumPy)
# ddof=1: Sample (unbiased estimator, default in Pandas / statistics)
print(f"Std Dev (Population): ${np.std(salaries, ddof=0):,.2f}")
print(f"Std Dev (Sample)    : ${np.std(salaries, ddof=1):,.2f}")
print(f"Variance            : {np.var(salaries):,.2f}")

# Percentiles & Quantiles
p25, p50, p75 = np.percentile(salaries, [25, 50, 75])
print(f"\n25th Percentile (Q1) : ${p25:,.2f}")
print(f"50th Percentile (Q2) : ${p50:,.2f}")
print(f"75th Percentile (Q3) : ${p75:,.2f}")
print(f"IQR (Spread)         : ${p75 - p25:,.2f}")


# ----------------------------------------------------------------------
# 3.5 Directional Operations with the axis Parameter & keepdims
# ----------------------------------------------------------------------
section("3.5 Directional Operations (axis=0 vs axis=1) & keepdims")

# Think of 2D array shape as (Rows, Columns) -> (Axis 0, Axis 1)
# RULE: The axis you specify is the axis that gets COLLAPSED (reduced)!
# - axis=0: Collapses across ROWS    -> gives result for each COLUMN (vertical reduction)
# - axis=1: Collapses across COLUMNS -> gives result for each ROW    (horizontal reduction)

exam_scores = np.array([
    # Math, Science, English
    [85, 90, 78],  # Student 0
    [70, 65, 80],  # Student 1
    [95, 92, 88],  # Student 2
    [60, 75, 70],  # Student 3
])
print(f"Exam Scores (4 Students x 3 Subjects):\n{exam_scores}")

# Average score per SUBJECT (collapse rows -> axis=0)
subject_averages = exam_scores.mean(axis=0)
print(f"\nSubject Averages [Math, Science, English] (axis=0): {subject_averages}")

# Average score per STUDENT (collapse cols -> axis=1)
student_averages = exam_scores.mean(axis=1)
print(f"Student Averages [Student 0..3] (axis=1)          : {student_averages}")

# The 'keepdims=True' parameter:
# Without keepdims: student_averages shape is (4,)
# With keepdims=True: shape is preserved as (4, 1), allowing seamless broadcasting back!
student_avg_2d = exam_scores.mean(axis=1, keepdims=True)
print(f"\nShape without keepdims: {student_averages.shape}")
print(f"Shape with keepdims   : {student_avg_2d.shape}")
print(f"Differences from student average:\n{exam_scores - student_avg_2d}")


# ----------------------------------------------------------------------
# 3.6 Broadcasting Rules & Dimension Compatibility
# ----------------------------------------------------------------------
section("3.6 Broadcasting Rules & Dimension Compatibility")

print("""
The Two Golden Rules of NumPy Broadcasting:
1. Arrays are aligned starting from their TRAILING (rightmost) dimensions.
2. Two dimensions are compatible if:
   a) They are EQUAL, OR
   b) One of them is 1 (the size-1 dimension is stretched to match).
""")

# Example 1: (3, 3) matrix + (3,) vector
# Alignment:
# Matrix: (3, 3)
# Vector: (   3) -> matches trailing dimension 3! Vector is broadcast across all 3 rows.
mat3x3 = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
row_vec = np.array([10, 20, 30])
print(f"Matrix (3, 3):\n{mat3x3}")
print(f"Vector (3,):\n{row_vec}")
print(f"Result of Matrix + Vector:\n{mat3x3 + row_vec}")

# Example 2: Broadcasting a Column Vector (3, 1) across Columns
col_vec = np.array([[100], [200], [300]])  # shape (3, 1)
print(f"\nColumn Vector shape {col_vec.shape}:\n{col_vec}")
print(f"Matrix + Column Vector:\n{mat3x3 + col_vec}")

# Example 3: Incompatible dimensions raise ValueError
incompatible_vec = np.array([1, 2])  # shape (2,) cannot broadcast with (3, 3)
try:
    _ = mat3x3 + incompatible_vec
except ValueError as e:
    print(f"\n⚠️ Incompatible Broadcast caught: {e}")

# Creating new dimensions with np.newaxis or None:
v = np.array([1, 2, 3])            # shape (3,)
v_col = v[:, np.newaxis]           # shape (3, 1)
v_row = v[np.newaxis, :]           # shape (1, 3)
print(f"\nOriginal vector shape: {v.shape}")
print(f"Expanded to column vector via np.newaxis: {v_col.shape}")
print(f"Outer product table (v_col * v_row):\n{v_col * v_row}")


# ----------------------------------------------------------------------
# 3.7 Broadcasting in Practice: Data Normalization
# ----------------------------------------------------------------------
section("3.7 Broadcasting in Practice (Feature Scaling)")

# Raw dataset: 4 samples with 3 features (e.g. Age, Income, Credit Score)
features = np.array([
    [25.0,  50000.0, 650.0],
    [45.0, 120000.0, 780.0],
    [35.0,  75000.0, 710.0],
    [52.0,  95000.0, 690.0]
])
print(f"Raw Features (4 samples x 3 features):\n{features}")

# 1. Mean Centering (Demeaning): X - mean
col_means = features.mean(axis=0, keepdims=True)  # shape (1, 3)
centered = features - col_means
print(f"\nColumn Means: {col_means}")
print(f"Centered Data (mean of each col is now ~0):\n{np.round(centered, 2)}")

# 2. Standard Scaling (Z-Score): (X - mean) / std
col_stds = features.std(axis=0, keepdims=True)    # shape (1, 3)
standardized = (features - col_means) / col_stds
print(f"\nZ-Score Standardized Data (mean=0, std=1):\n{np.round(standardized, 2)}")

# 3. Min-Max Normalization: (X - min) / (max - min) -> maps all features to [0, 1]
col_mins = features.min(axis=0, keepdims=True)
col_maxs = features.max(axis=0, keepdims=True)
min_max_scaled = (features - col_mins) / (col_maxs - col_mins)
print(f"\nMin-Max Scaled to [0, 1]:\n{np.round(min_max_scaled, 3)}")


# ----------------------------------------------------------------------
# 3.8 Comparison & Logical Operations: all, any, isclose
# ----------------------------------------------------------------------
section("3.8 Comparison & Logical Operations")

arr1 = np.array([10, 20, 30, 40])
arr2 = np.array([10, 25, 30, 35])

print(f"arr1: {arr1}")
print(f"arr2: {arr2}")
print(f"Element-wise equal (arr1 == arr2): {arr1 == arr2}")
print(f"Element-wise greater (arr1 > arr2): {arr1 > arr2}")

# np.all: True if EVERY element satisfies condition
# np.any: True if AT LEAST ONE element satisfies condition
print(f"\nAre ALL elements in arr1 > 0?  : {np.all(arr1 > 0)}")
print(f"Are ALL elements identical?    : {np.all(arr1 == arr2)}")
print(f"Is ANY element in arr1 > 35?   : {np.any(arr1 > 35)}")

# Floating point precision gotcha:
x = 0.1 + 0.2
y = 0.3
print(f"\nPure Python float test (0.1 + 0.2 == 0.3): {x == y} (0.1+0.2 is actually {x})")
# np.isclose and np.allclose use absolute & relative tolerances for safe float comparison:
print(f"np.isclose(0.1 + 0.2, 0.3)               : {np.isclose(x, y)} (Safe float check!)")

float_vec_a = np.array([1.0000001, 2.0, 3.0])
float_vec_b = np.array([1.0000002, 2.0, 3.0])
print(f"np.allclose(vec_a, vec_b)                : {np.allclose(float_vec_a, float_vec_b)}")


# ----------------------------------------------------------------------
# 3.9 Numerical Rounding: round, floor, ceil, trunc
# ----------------------------------------------------------------------
section("3.9 Numerical Rounding (round, floor, ceil, trunc)")

decimals = np.array([-2.7, -1.2, 0.0, 1.5, 2.3, 2.8])
print(f"Original Decimals: {decimals}")

# np.round(): Rounds to nearest integer (uses IEEE banker's rounding to nearest even)
print(f"np.round(decimals) : {np.round(decimals)}")

# np.floor(): Rounds DOWN towards -infinity
print(f"np.floor(decimals) : {np.floor(decimals)}")

# np.ceil(): Rounds UP towards +infinity
print(f"np.ceil(decimals)  : {np.ceil(decimals)}")

# np.trunc(): Truncates (chops off) decimals towards ZERO
print(f"np.trunc(decimals) : {np.trunc(decimals)}")


# ----------------------------------------------------------------------
# 3.10 Cumulative Operations: cumsum, cumprod, diff
# ----------------------------------------------------------------------
section("3.10 Cumulative Operations (cumsum, cumprod, diff)")

# Daily transactions / cash flows
cash_flow = np.array([1000, -200, 450, -100, 300])
print(f"Daily cash flows: {cash_flow}")

# np.cumsum: Running total / account balance over time
balance_over_time = np.cumsum(cash_flow)
print(f"Account Balance Over Time (cumsum): {balance_over_time}")

# np.cumprod: Compound interest / investment growth multiplier
# e.g. Daily returns: +2%, -1%, +3% -> factors 1.02, 0.99, 1.03
returns_factors = np.array([1.02, 0.99, 1.03, 1.01])
portfolio_growth = np.cumprod(returns_factors)
print(f"Portfolio Multiplier Over Time (cumprod): {np.round(portfolio_growth, 4)}")

# np.diff: Discrete difference between consecutive elements (arr[i+1] - arr[i])
# Note: Output array length is (N - 1)
stock_prices = np.array([150.0, 153.5, 151.0, 158.0, 162.5])
daily_changes = np.diff(stock_prices)
print(f"\nStock Prices: {stock_prices}")
print(f"Daily Price Changes (diff): {daily_changes}")


# ----------------------------------------------------------------------
# 3.11 Comprehensive Hands-on Practice Challenge & Solution
# ----------------------------------------------------------------------
section("3.11 Practice Challenge: Financial Asset Portfolio Analysis")
print("""
Scenario:
You are analyzing the performance of 3 Tech Stocks [AAPL, MSFT, GOOG] over 5 trading days.
Given the 5x3 price matrix:
1. Daily Percentage Returns:
   Calculate the day-over-day price returns using np.diff() and price indexing:
   Return = (Price[t] - Price[t-1]) / Price[t-1]
2. Summary Performance:
   Compute the Mean Return and Volatility (Standard Deviation) for each stock across the 4 return days.
3. Z-Score Standardized Returns:
   Use broadcasting with keepdims=True to standardize returns for each stock (mean=0, std=1).
4. Market Alignment Check:
   Check if there were ANY days where ALL 3 stocks had positive returns (> 0).
5. Top Performing Asset:
   Identify which stock had the highest cumulative return by the end of the period.
""")

# Prices over 5 days for [AAPL, MSFT, GOOG]
prices = np.array([
    [150.0, 310.0, 130.0],  # Day 0
    [153.0, 312.0, 128.5],  # Day 1
    [151.5, 315.0, 132.0],  # Day 2
    [156.0, 320.0, 135.0],  # Day 3
    [160.5, 318.0, 139.0],  # Day 4
])
stock_names = ["AAPL", "MSFT", "GOOG"]

# Step 1: Calculate daily percentage returns
# np.diff(prices, axis=0) gives price changes. Divide by prices[:-1] (prices of prior days)
daily_price_change = np.diff(prices, axis=0)
daily_returns = daily_price_change / prices[:-1]
print("Step 1 - Daily Percentage Returns (4 trading days x 3 stocks):")
print(np.round(daily_returns * 100, 2), "%\n")

# Step 2: Mean return and Volatility (Std Dev) per stock
mean_returns = daily_returns.mean(axis=0)
volatilities = daily_returns.std(axis=0, ddof=1)
for i, name in enumerate(stock_names):
    print(f"Step 2 - {name:4s} -> Mean Return: {mean_returns[i]*100:+.2f}% | Volatility (Std): {volatilities[i]*100:.2f}%")

# Step 3: Standardize returns via broadcasting with keepdims=True
# Standardized = (Returns - Mean) / Volatility
standardized_returns = (daily_returns - daily_returns.mean(axis=0, keepdims=True)) / daily_returns.std(axis=0, ddof=1, keepdims=True)
print("\nStep 3 - Standardized Returns (Z-scores):\n", np.round(standardized_returns, 2))

# Step 4: Check if any days had all positive returns
all_positive_days = np.all(daily_returns > 0, axis=1)  # condition evaluated across stocks (axis=1)
print(f"\nStep 4 - Positive on all stocks per day? {all_positive_days}")
print(f"Was there ANY day where all 3 stocks were positive? {np.any(all_positive_days)}")

# Step 5: Cumulative return from Day 0 to Day 4
# Cumulative Return = (Final Price - Initial Price) / Initial Price
total_return = (prices[-1] - prices[0]) / prices[0]
best_stock_idx = np.argmax(total_return)
print(f"\nStep 5 - Total 5-Day Returns:")
for i, name in enumerate(stock_names):
    print(f"  {name}: {total_return[i]*100:+.2f}%")
print(f"🏆 Best Performing Stock: {stock_names[best_stock_idx]} with {total_return[best_stock_idx]*100:+.2f}% total return!")
