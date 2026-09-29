"""
Module 4: Data Processing & Business Analysis
=============================================
Topics Covered:
- 4.1 Sorting: np.sort(), in-place .sort(), and index-based sorting (np.argsort)
- 4.2 Searching: np.where(), np.nonzero(), np.argmax(), np.argmin()
- 4.3 Unique values & value counts: np.unique(..., return_counts=True)
- 4.4 Handling missing & invalid data: np.nan, np.isnan, np.nanmean, np.nan_to_num
- 4.5 Conditional replacement & clipping: np.where, np.select (SQL CASE WHEN), np.clip
- 4.6 Array-based data cleaning patterns: Outlier capping (IQR rule) & sentinel value handling
- 4.7 Datetime arrays: np.datetime64, np.timedelta64, date ranges & business day math
- 4.8 Modern random number generation: np.random.default_rng(seed)
- 4.9 Sampling & distributions: uniform, normal, binomial, Poisson
- 4.10 Shuffling, permutations & bootstrapping: rng.choice, rng.shuffle
- 4.11 Hands-on Business Challenge & Solution (E-Commerce Customer Analytics & Monte Carlo Bootstrapping)
"""

import numpy as np


def section(title: str):
    print(f"\n{'=' * 65}\n  {title}\n{'=' * 65}")


# ----------------------------------------------------------------------
# 4.1 Sorting: np.sort(), in-place sort, and np.argsort()
# ----------------------------------------------------------------------
section("4.1 Sorting (np.sort, in-place sort, np.argsort)")

raw_scores = np.array([42, 15, 88, 63, 27, 95, 54])
print(f"Raw scores: {raw_scores}")

# 1. np.sort() returns a sorted COPY:
sorted_copy = np.sort(raw_scores)
print(f"np.sort() (ascending copy) : {sorted_copy}")
print(f"Descending sort via slice  : {sorted_copy[::-1]}")

# 2. In-place sorting (mutates array without extra memory):
arr_copy = raw_scores.copy()
arr_copy.sort()
print(f"In-place arr.sort()        : {arr_copy}")

# 3. np.argsort(): Returns the INDICES that would sort the array!
# Database Equivalent: ORDER BY column_name across an entire table.
sort_indices = np.argsort(raw_scores)
print(f"\nSort Indices (np.argsort)  : {sort_indices}")
print(f"Reconstructed via indices  : {raw_scores[sort_indices]}")

# Sorting a 2D table by a specific column (e.g. sort employees by salary, Col 1):
# [EmployeeID, Salary, DepartmentCode]
employees = np.array([
    [101, 85000, 2],
    [102, 52000, 1],
    [103, 110000, 2],
    [104, 64000, 1]
])
print(f"\nEmployees Table [ID, Salary, Dept]:\n{employees}")

# Order rows by Salary (Col 1) ascending:
salary_sort_order = np.argsort(employees[:, 1])
sorted_by_salary = employees[salary_sort_order]
print(f"Sorted by Salary ascending (SQL ORDER BY Salary):\n{sorted_by_salary}")


# ----------------------------------------------------------------------
# 4.2 Searching: np.where(), np.nonzero(), np.argmax(), np.argmin()
# ----------------------------------------------------------------------
section("4.2 Searching (np.where, np.nonzero, argmax, argmin)")

revenue = np.array([1200, 4500, 3100, 8900, 2400, 8900, 1500])
print(f"Daily Revenue: {revenue}")

# np.where(condition) without replacement returns indices matching condition
high_rev_idx = np.where(revenue > 4000)[0]
print(f"Days where Revenue > 4000 (indices): {high_rev_idx}")
print(f"Values on those days: {revenue[high_rev_idx]}")

# np.nonzero() returns indices of non-zero (or True) elements
active_mask = (revenue >= 3000)
print(f"np.nonzero(revenue >= 3000): {np.nonzero(active_mask)[0]}")

# argmax / argmin: return the index of the first occurrence of maximum/minimum
max_day = np.argmax(revenue)
min_day = np.argmin(revenue)
print(f"Peak revenue day index (argmax): Day {max_day} (${revenue[max_day]})")
print(f"Lowest revenue day index (argmin): Day {min_day} (${revenue[min_day]})")

# In 2D matrices: find row/col of maximum along axes
grid = np.array([
    [10, 80, 30],
    [90, 20, 40]
])
print(f"\n2D Grid:\n{grid}")
print(f"Argmax across columns (axis=0) -> row of max per col: {np.argmax(grid, axis=0)}")
print(f"Argmax across rows    (axis=1) -> col of max per row: {np.argmax(grid, axis=1)}")


# ----------------------------------------------------------------------
# 4.3 Unique Values & Value Counts: np.unique()
# ----------------------------------------------------------------------
section("4.3 Unique Values & Value Counts (SQL DISTINCT / GROUP BY)")

customer_cities = np.array(["NYC", "LA", "Chicago", "NYC", "LA", "NYC", "Miami", "LA", "NYC"])
print(f"Customer Cities: {customer_cities}")

# 1. SQL SELECT DISTINCT:
unique_cities = np.unique(customer_cities)
print(f"Distinct Cities: {unique_cities}")

# 2. SQL SELECT city, COUNT(*) GROUP BY city:
cities, counts = np.unique(customer_cities, return_counts=True)
print("\nCity Counts (GROUP BY city COUNT(*)):")
for city, count in zip(cities, counts):
    print(f"  {city:8s} : {count} orders")

# 3. Unique rows in a 2D table (e.g. deduplicating user interaction events):
events = np.array([
    [1, 100],  # User 1, Product 100
    [2, 200],  # User 2, Product 200
    [1, 100],  # Duplicate
    [3, 100]
])
unique_events = np.unique(events, axis=0)
print(f"\nDeduplicated unique events (axis=0):\n{unique_events}")


# ----------------------------------------------------------------------
# 4.4 Handling Missing & Invalid Data: np.nan, np.isnan, np.nanmean
# ----------------------------------------------------------------------
section("4.4 Handling Missing & Invalid Data (NaNs & Infs)")

# What is np.nan? IEEE 754 float representation for 'Not a Number'
# GOTCHA: In IEEE 754, np.nan != np.nan (evaluates to False!)
print(f"np.nan == np.nan test: {np.nan == np.nan}  (Must use np.isnan!)")

measurements = np.array([23.5, np.nan, 28.1, np.nan, 25.4, 30.2])
print(f"\nMeasurements with NaNs: {measurements}")

# Checking for missing values:
nan_mask = np.isnan(measurements)
print(f"Boolean mask of NaNs (np.isnan)   : {nan_mask}")
print(f"Total missing values              : {np.sum(nan_mask)}")

# Gotcha: Standard aggregations get poisoned by NaNs!
print(f"Standard mean (poisoned by NaN)   : {np.mean(measurements)}")

# Solution: NaN-safe functions (ignore missing values during aggregation):
clean_mean = np.nanmean(measurements)
clean_median = np.nanmedian(measurements)
clean_std = np.nanstd(measurements)
print(f"NaN-safe Mean (np.nanmean)        : {clean_mean:.2f}")
print(f"NaN-safe Median (np.nanmedian)    : {clean_median:.2f}")
print(f"NaN-safe Std Dev (np.nanstd)      : {clean_std:.2f}")

# Imputation (replacing NaNs):
# Method A: np.nan_to_num(arr, nan=replacement_val)
imputed_zero = np.nan_to_num(measurements, nan=0.0)
print(f"\nImputed with 0.0 (nan_to_num)     : {imputed_zero}")

# Method B: Impute with column/dataset median (best practice for skewed data):
imputed_median = measurements.copy()
imputed_median[np.isnan(imputed_median)] = clean_median
print(f"Imputed with Median ({clean_median:.2f})       : {np.round(imputed_median, 2)}")


# ----------------------------------------------------------------------
# 4.5 Conditional Replacement & Clipping: np.where, np.select, np.clip
# ----------------------------------------------------------------------
section("4.5 Conditional Logic & Clipping (SQL CASE WHEN / np.select)")

# 1. Simple Binary condition: np.where(cond, true_val, false_val)
scores = np.array([45, 82, 67, 91, 58])
grade_status = np.where(scores >= 70, "PASS", "FAIL")
print(f"Scores: {scores}")
print(f"Pass/Fail status: {grade_status}")

# 2. Multi-Condition Categorization: np.select()
# Direct equivalent of SQL:
# CASE
#   WHEN score >= 90 THEN 'Tier 1 (Platinum)'
#   WHEN score >= 75 THEN 'Tier 2 (Gold)'
#   WHEN score >= 60 THEN 'Tier 3 (Silver)'
#   ELSE 'Tier 4 (Bronze)'
# END
conditions = [
    scores >= 90,
    scores >= 75,
    scores >= 60
]
choices = [
    "Tier 1 (Platinum)",
    "Tier 2 (Gold)",
    "Tier 3 (Silver)"
]
customer_tiers = np.select(conditions, choices, default="Tier 4 (Bronze)")
print(f"\nCustomer Tiers (np.select):\n{list(zip(scores, customer_tiers))}")

# 3. np.clip(arr, min, max): Restrict values within a safe range
unbounded_metrics = np.array([-15, 5, 45, 80, 120, 250, 95])
clipped = np.clip(unbounded_metrics, a_min=0, a_max=100)
print(f"\nUnbounded metrics : {unbounded_metrics}")
print(f"Clipped to [0, 100]: {clipped}")


# ----------------------------------------------------------------------
# 4.6 Array-Based Data Cleaning Patterns (Outlier Capping via IQR)
# ----------------------------------------------------------------------
section("4.6 Real-World Data Cleaning: Outlier Capping (IQR Rule)")

# Sensor / Transaction values containing anomalies:
raw_transactions = np.array([45, 52, 48, 55, 49, 51, 50, 47, 53, 500, -80, 52, 46])
print(f"Raw transactions (with outliers 500 and -80):\n{raw_transactions}")

# 1. Handle sentinel missing values (e.g., -999 from legacy systems)
raw_with_sentinel = np.array([10.5, 20.0, -999.0, 35.2, -999.0, 42.1])
cleaned_sentinel = np.where(raw_with_sentinel == -999.0, np.nan, raw_with_sentinel)
print(f"Sentinel -999 converted to NaN: {cleaned_sentinel}")

# 2. Outlier Capping via 1.5 * IQR Rule (Winsorization)
q25 = np.percentile(raw_transactions, 25)
q75 = np.percentile(raw_transactions, 75)
iqr = q75 - q25
lower_bound = q25 - 1.5 * iqr
upper_bound = q75 + 1.5 * iqr

print(f"\nQ1 (25%): {q25:.1f} | Q3 (75%): {q75:.1f} | IQR: {iqr:.1f}")
print(f"Valid range: [{lower_bound:.1f}, {upper_bound:.1f}]")

# Cap the outliers rather than dropping them (preserves data points):
capped_transactions = np.clip(raw_transactions, lower_bound, upper_bound)
print(f"Transactions after IQR outlier capping:\n{np.round(capped_transactions, 1)}")


# ----------------------------------------------------------------------
# 4.7 Datetime Arrays & Date Math: np.datetime64, np.timedelta64
# ----------------------------------------------------------------------
section("4.7 Datetime Arrays & Date Math (np.datetime64, np.timedelta64)")

# Create ISO-8601 timestamps with unit precision ('D'=day, 's'=second, 'ms'=millisecond):
order_date = np.datetime64('2026-09-29', 'D')
print(f"Order Date: {order_date} (dtype: {order_date.dtype})")

# Generate continuous date ranges (like PostgreSQL generate_series):
q1_dates = np.arange('2026-01-01', '2026-01-06', dtype='datetime64[D]')
print(f"\nDate sequence (5 days):\n{q1_dates}")

# Date arithmetic using np.timedelta64:
due_dates = q1_dates + np.timedelta64(14, 'D')  # +14 days payment term
print(f"Due Dates (+14 Days):\n{due_dates}")

# Calculating difference between dates:
start = np.datetime64('2026-09-01')
end = np.datetime64('2026-09-29')
days_passed = end - start
print(f"\nElapsed time between {start} and {end}: {days_passed}")

# Business Day calculation:
work_days = np.busday_count('2026-09-01', '2026-09-29')
print(f"Working / Business days in that window: {work_days} business days")


# ----------------------------------------------------------------------
# 4.8 Modern Random Number Generation: np.random.default_rng()
# ----------------------------------------------------------------------
section("4.8 Modern Random Number Generation (default_rng)")

# Best Practice: Use default_rng(seed) instead of legacy np.random.seed()
# default_rng uses the PCG-64 bit generator (faster, statistically superior, thread-safe).
rng = np.random.default_rng(seed=42)

# Generate floats in [0.0, 1.0):
random_floats = rng.random(size=(2, 3))
print(f"Random floats (2x3):\n{np.round(random_floats, 4)}")

# Generate random integers: rng.integers(low, high, size, endpoint=False)
random_ints = rng.integers(low=1, high=100, size=5)
print(f"\nRandom integers in [1, 99]: {random_ints}")


# ----------------------------------------------------------------------
# 4.9 Sampling & Probability Distributions
# ----------------------------------------------------------------------
section("4.9 Sampling & Statistical Distributions")

# 1. Uniform Distribution: Equal probability between [min, max]
# Real-world use: continuous baseline simulations, random discount percentages
uniform_sample = rng.uniform(low=5.0, high=15.0, size=4)
print(f"Uniform [5.0, 15.0]: {np.round(uniform_sample, 2)}")

# 2. Normal (Gaussian) Distribution: Bell curve (loc=mean, scale=std_dev)
# Real-world use: user session duration, sensor error, daily stock returns
normal_sample = rng.normal(loc=100.0, scale=15.0, size=4)
print(f"Normal (mean=100, std=15): {np.round(normal_sample, 2)}")

# 3. Binomial Distribution: Coin flips / A/B Test conversion simulation
# Real-world use: Out of n=100 ad impressions with p=0.05 conversion rate, how many convert?
ab_test_conversions = rng.binomial(n=100, p=0.05, size=5)
print(f"A/B Test conversions (5 experiments of 100 users @ 5%): {ab_test_conversions}")

# 4. Poisson Distribution: Rate of events per unit time
# Real-world use: Server requests per second (lam = expected average rate)
server_requests_per_sec = rng.poisson(lam=25, size=5)
print(f"Server requests per second (λ=25): {server_requests_per_sec}")


# ----------------------------------------------------------------------
# 4.10 Shuffling, Permutations, and Bootstrapping: rng.choice & rng.shuffle
# ----------------------------------------------------------------------
section("4.10 Shuffling, Permutations & Bootstrapping (rng.choice)")

items = np.array(["Apples", "Bananas", "Cherries", "Dates", "Elderberries"])
print(f"Original items: {items}")

# 1. rng.shuffle() modifies the array IN-PLACE:
shuffled_items = items.copy()
rng.shuffle(shuffled_items)
print(f"In-place shuffled: {shuffled_items}")

# 2. rng.permutation() returns a new permuted copy:
permuted = rng.permutation(items)
print(f"Permuted copy    : {permuted}")

# 3. rng.choice(): Random sampling (with or without replacement, optional weights)
# Sample 3 items WITHOUT replacement (like picking raffle winners):
winners = rng.choice(items, size=3, replace=False)
print(f"\nRandom 3 winners (no replacement): {winners}")

# Weighted sampling (e.g. 70% mobile traffic, 25% desktop, 5% tablet):
devices = np.array(["Mobile", "Desktop", "Tablet"])
traffic_weights = [0.70, 0.25, 0.05]
sampled_traffic = rng.choice(devices, size=8, replace=True, p=traffic_weights)
print(f"Weighted traffic simulation: {sampled_traffic}")


# ----------------------------------------------------------------------
# 4.11 Comprehensive Hands-on Practice Challenge & Solution
# ----------------------------------------------------------------------
section("4.11 Hands-on Challenge: Customer Analytics & Bootstrapping")
print("""
Business Analytics Scenario:
You are analyzing an e-commerce order table with 8 customers:
Columns: [CustomerID, RawOrderAmount, CustomerSatisfaction (1-5, with NaNs)]

Your Tasks:
1. Handle Missing Data:
   Compute the nan-safe median of Customer Satisfaction and impute missing values.
2. Outlier Capping (Winsorization):
   Find outliers in RawOrderAmount using IQR rule and cap them into a clean column.
3. Customer Segmentation (SQL CASE WHEN equivalent):
   Use np.select to assign VIP status:
     - 'Platinum': Cleaned Order Amount >= 200 AND Satisfaction >= 4.0
     - 'Gold'    : Cleaned Order Amount >= 100
     - 'Standard': All other orders
4. Sort by Revenue:
   Use np.argsort() to display the customer table sorted by Cleaned Order Amount descending.
5. Statistical Bootstrapping (Monte Carlo):
   Simulate 1,000 bootstrap resamples (with replacement) of Cleaned Order Amount
   to compute the 95% Confidence Interval for the true mean order value!
""")

# [CustomerID, OrderAmount, Satisfaction]
data = np.array([
    [101,  150.0, 4.5],
    [102,   45.0, 2.0],
    [103,  850.0, 5.0],   # Outlier amount
    [104,  120.0, np.nan],# Missing satisfaction
    [105,   80.0, 3.5],
    [106,  250.0, 4.0],
    [107,   30.0, np.nan],# Missing satisfaction
    [108,  310.0, 4.8],
])

# Step 1: Impute missing satisfaction with median
sat_col = data[:, 2]
median_sat = np.nanmedian(sat_col)
clean_sat = sat_col.copy()
clean_sat[np.isnan(clean_sat)] = median_sat
print(f"Step 1 - Satisfaction Median: {median_sat:.1f}")
print(f"Step 1 - Imputed Satisfaction: {clean_sat}")

# Step 2: Outlier Capping on Order Amounts via IQR
amounts = data[:, 1]
q25, q75 = np.percentile(amounts, [25, 75])
iqr = q75 - q25
upper_limit = q75 + 1.5 * iqr
lower_limit = q25 - 1.5 * iqr
clean_amounts = np.clip(amounts, lower_limit, upper_limit)
print(f"\nStep 2 - Amount Upper Limit: ${upper_limit:.2f}")
print(f"Step 2 - Capped Amounts: {clean_amounts}")

# Step 3: Segment Customers with np.select
conditions = [
    (clean_amounts >= 200) & (clean_sat >= 4.0),
    (clean_amounts >= 100)
]
segment_choices = ["Platinum", "Gold"]
segments = np.select(conditions, segment_choices, default="Standard")
print(f"\nStep 3 - Segments: {segments}")

# Step 4: Sort Table Descending by Cleaned Amount
# Order descending: argsort on negative values or reverse slice
sort_order = np.argsort(clean_amounts)[::-1]
print("\nStep 4 - Sorted Customer Report (Highest Revenue First):")
print(f"{'CustID':<8} {'Clean Amount':<14} {'Satisfaction':<14} {'Segment'}")
print("-" * 48)
for idx in sort_order:
    cust_id = int(data[idx, 0])
    amt = clean_amounts[idx]
    sat = clean_sat[idx]
    seg = segments[idx]
    print(f"{cust_id:<8} ${amt:<13.2f} {sat:<14.1f} {seg}")

# Step 5: Monte Carlo Bootstrap Simulation (95% Confidence Interval of Mean Revenue)
sim_rng = np.random.default_rng(seed=2026)
n_bootstraps = 1000
bootstrap_means = np.empty(n_bootstraps)

for b in range(n_bootstraps):
    resample = sim_rng.choice(clean_amounts, size=len(clean_amounts), replace=True)
    bootstrap_means[b] = resample.mean()

ci_lower = np.percentile(bootstrap_means, 2.5)
ci_upper = np.percentile(bootstrap_means, 97.5)
observed_mean = clean_amounts.mean()

print(f"\nStep 5 - Monte Carlo Bootstrap Results (1,000 iterations):")
print(f"Observed Sample Mean Order Amount : ${observed_mean:.2f}")
print(f"95% Bootstrap Confidence Interval : [${ci_lower:.2f}, ${ci_upper:.2f}]")
