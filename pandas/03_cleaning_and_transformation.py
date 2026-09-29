"""
Module 3: Data Cleaning & Transformation
========================================
Topics Covered:
- 3.1 Handling Missing Values (isna, notna, fillna, dropna with subsets & thresholds)
- 3.2 Detecting & Removing Duplicates (duplicated, drop_duplicates)
- 3.3 Type Conversions (astype, pd.to_numeric, pd.to_datetime with error coercion)
- 3.4 Vectorized String Operations (.str.strip, .lower, .contains, .replace, regex extract)
- 3.5 Applying Functions (.apply, .map, .transform)
- 3.6 Filtering Rows (boolean masks, compound conditions &, |, ~, and df.query())
- 3.7 Sorting & Ranking (sort_values, multi-column sorting, rank)
- 3.8 Replacing Values & Conditional Masking (.replace, .where, .mask, np.select)
- 3.9 Renaming & Standardizing Dirty Data (clean column headers, strip whitespace)
- 3.10 Outlier Detection & Treatment (IQR fence clipping, Z-score thresholds)
- 3.11 Hands-on Practice Challenge with complete solution!
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
# 3.1 Handling Missing Values (NaN / None)
# ----------------------------------------------------------------------
section("3.1 Handling Missing Values: Detection, Imputation & Dropping")

# Real-world dirty dataset: Customer registrations
raw_data = {
    "customer_id": [101, 102, 103, 104, 105, 106],
    "email": ["alice@gmail.com", "BOB@YAHOO.COM", None, "david@outlook.com", "eva@gmail.com", None],
    "age": [25, np.nan, 32, 45, np.nan, 29],
    "income": [65000.0, 72000.0, np.nan, 120000.0, 58000.0, 61000.0],
    "signup_channel": ["Google", "Facebook", "Google", None, "Referral", "Google"]
}
df = pd.DataFrame(raw_data)
print("Raw Dirty Dataset:\n", df)

# 1. Detection: Count missing values per column
print("\nMissing values per column:\n", df.isna().sum())

# 2. Imputation: Fill missing numeric 'age' with column median
median_age = df["age"].median()
df["age_filled"] = df["age"].fillna(median_age)
print(f"\nImputed age (median = {median_age:.1f}):\n", df[["age", "age_filled"]])

# Forward fill / Backward fill (useful for time series & sensor logs):
df["signup_channel_ffill"] = df["signup_channel"].ffill()
print("\nForward-filled signup channel:\n", df[["signup_channel", "signup_channel_ffill"]])

# 3. Removal: Drop rows where critical primary identifiers (email) are missing
df_clean_email = df.dropna(subset=["email"])
print(f"\nRows remaining after dropping missing emails: {len(df_clean_email)} / {len(df)}")


# ----------------------------------------------------------------------
# 3.2 Duplicate Detection and Removal
# ----------------------------------------------------------------------
section("3.2 Duplicate Records: Detection & Removal")

orders_with_dupes = pd.DataFrame({
    "order_id": ["ORD-1", "ORD-2", "ORD-2", "ORD-3", "ORD-3", "ORD-3"],
    "user_id": [10, 20, 20, 30, 30, 30],
    "amount": [150.0, 200.0, 200.0, 75.0, 75.0, 95.0]  # Note: 3rd ORD-3 has different amount
})
print("Orders with duplicate records:\n", orders_with_dupes)

# Check which rows are duplicates:
print("\nDuplicate mask (is row identical to previous?):\n", orders_with_dupes.duplicated())

# Drop exact duplicate rows:
exact_deduped = orders_with_dupes.drop_duplicates()
print("\nExact deduped rows:\n", exact_deduped)

# Drop duplicates based on specific business key (order_id), keeping the first:
deduped_by_id = orders_with_dupes.drop_duplicates(subset=["order_id"], keep="first")
print("\nDeduped by order_id (keeping first):\n", deduped_by_id)


# ----------------------------------------------------------------------
# 3.3 Type Conversions (astype, pd.to_numeric, pd.to_datetime)
# ----------------------------------------------------------------------
section("3.3 Type Conversions & Safe Parsing with error coercion")

dirty_types = pd.DataFrame({
    "price_str": ["$19.99", "$45.50", "Free", "$89.00", "N/A"],
    "date_str": ["2026-01-15", "15/02/2026", "2026-03-30", "invalid_date", "2026-05-12"],
    "is_active_int": [1, 0, 1, 1, 0]
})
print("Dirty Types DataFrame:\n", dirty_types)

# 1. Clean string currency and safely parse to float (coerce invalid strings like 'Free' to NaN):
clean_prices = dirty_types["price_str"].str.replace("$", "", regex=False)
dirty_types["price_numeric"] = pd.to_numeric(clean_prices, errors="coerce")
print("\nSafely parsed numeric prices (errors='coerce'):\n", dirty_types[["price_str", "price_numeric"]])

# 2. Parse mixed dates safely (coercing 'invalid_date' to NaT - Not a Time):
dirty_types["parsed_date"] = pd.to_datetime(dirty_types["date_str"], format="mixed", errors="coerce")
print("\nSafely parsed datetimes:\n", dirty_types[["date_str", "parsed_date"]])

# 3. Boolean casting:
dirty_types["is_active"] = dirty_types["is_active_int"].astype(bool)
print("\nCast integer 1/0 to bool:\n", dirty_types[["is_active_int", "is_active"]])


# ----------------------------------------------------------------------
# 3.4 Vectorized String Operations (.str Accessor)
# ----------------------------------------------------------------------
section("3.4 Vectorized String Operations with .str")

contacts = pd.DataFrame({
    "raw_name": ["  Alice Johnson ", "BOB SMITH", "Charlie Brown  ", "  diana Prince "],
    "email": ["alice@gmail.com", "bob@corporate.co.uk", "charlie@gmail.com", "diana@yahoo.com"],
    "phone": ["+1-555-0192", "+1-555-0144", "+44-20-7946", "+1-555-0189"]
})

# Standardize names: strip leading/trailing whitespace, convert to Title Case
contacts["clean_name"] = contacts["raw_name"].str.strip().str.title()

# Filter emails containing 'gmail':
gmail_users = contacts[contacts["email"].str.contains("gmail", case=False, na=False)]
print("Gmail users filtered via .str.contains():\n", gmail_users[["clean_name", "email"]])

# Extract domain using regex capture group:
contacts["email_domain"] = contacts["email"].str.extract(r"@([\w\.-]+)")
print("\nExtracted Email Domains:\n", contacts[["clean_name", "email", "email_domain"]])


# ----------------------------------------------------------------------
# 3.5 Applying Functions (.apply, .map, .transform)
# ----------------------------------------------------------------------
section("3.5 Applying Functions (.apply, .map, .transform)")

scores_df = pd.DataFrame({
    "student": ["Alex", "Beth", "Chris", "Dave"],
    "score": [88, 92, 54, 76],
    "department": ["CS", "CS", "Math", "Math"]
})

# .map() with a dictionary: Map department code to full department name
dept_names = {"CS": "Computer Science", "Math": "Mathematics"}
scores_df["dept_full"] = scores_df["department"].map(dept_names)

# .apply() with custom business logic (e.g. grading system):
def grade_classifier(score: float) -> str:
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    return "F"

scores_df["grade"] = scores_df["score"].apply(grade_classifier)
print("Scores with mapped department and applied grades:\n", scores_df)

# .transform() for group-level operations (e.g., score deviation from department mean):
scores_df["dept_mean"] = scores_df.groupby("department")["score"].transform("mean")
scores_df["score_diff_from_dept"] = scores_df["score"] - scores_df["dept_mean"]
print("\nDepartment relative performance with .transform():\n",
      scores_df[["student", "score", "dept_full", "dept_mean", "score_diff_from_dept"]])


# ----------------------------------------------------------------------
# 3.6 Filtering with Compound Conditions and df.query()
# ----------------------------------------------------------------------
section("3.6 Filtering: Boolean Masks vs. SQL-like df.query()")

inventory = pd.DataFrame({
    "item": ["Laptop", "Monitor", "Keyboard", "Mouse", "Desk", "Chair"],
    "category": ["Electronics", "Electronics", "Electronics", "Electronics", "Furniture", "Furniture"],
    "price": [1200.0, 300.0, 85.0, 30.0, 450.0, 220.0],
    "in_stock": [True, True, False, True, True, False]
})

# Method 1: Boolean mask with bitwise & (AND), | (OR), ~ (NOT)
# Note: Always wrap each condition in parentheses!
mask = (inventory["category"] == "Electronics") & (inventory["price"] > 100.0) & inventory["in_stock"]
print("Filtered via Boolean Mask:\n", inventory[mask])

# Method 2: Expressive SQL-like df.query() (Cleaner syntax for backend developers!)
query_result = inventory.query("category == 'Electronics' and price > 100.0 and in_stock == True")
print("\nFiltered via df.query():\n", query_result)


# ----------------------------------------------------------------------
# 3.7 Sorting & Ranking
# ----------------------------------------------------------------------
section("3.7 Sorting & Ranking Data")

# Sort by multiple columns: category ascending, price descending
sorted_inventory = inventory.sort_values(by=["category", "price"], ascending=[True, False])
print("Sorted by Category ASC, Price DESC:\n", sorted_inventory)

# Rank items by price within each category:
inventory["price_rank_in_cat"] = inventory.groupby("category")["price"].rank(ascending=False, method="dense")
print("\nRanked within category by price:\n", inventory[["category", "item", "price", "price_rank_in_cat"]])


# ----------------------------------------------------------------------
# 3.8 Conditional Replacement: .where, .mask, and np.select
# ----------------------------------------------------------------------
section("3.8 Conditional Replacement (.where, .mask, np.select)")

sales_reps = pd.DataFrame({
    "rep": ["John", "Sara", "Mike", "Anna"],
    "deals_closed": [12, 4, 18, 8],
    "revenue": [120000, 35000, 210000, 80000]
})

# np.select() is the Python equivalent of SQL CASE WHEN ... THEN ... ELSE
conditions = [
    sales_reps["revenue"] >= 150000,
    sales_reps["revenue"] >= 75000,
    sales_reps["revenue"] < 75000
]
choices = ["Tier 1 (Elite)", "Tier 2 (Core)", "Tier 3 (Developing)"]
sales_reps["performance_tier"] = np.select(conditions, choices, default="Unranked")
print("SQL CASE WHEN equivalent via np.select():\n", sales_reps)


# ----------------------------------------------------------------------
# 3.9 Outlier Detection & Treatment (IQR Rule & Clipping)
# ----------------------------------------------------------------------
section("3.9 Outlier Detection & Treatment (IQR & Clipping)")

# Salary dataset with extreme outliers (e.g. CEO compensation or data entry errors)
salaries = pd.Series([45000, 48000, 52000, 50000, 55000, 53000, 49000, 2500000, 15000])  # 2.5M is an outlier!

# Compute Interquartile Range (IQR)
q1 = salaries.quantile(0.25)
q3 = salaries.quantile(0.75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

print(f"Q1: {q1}, Q3: {q3}, IQR: {iqr}")
print(f"Valid Non-Outlier Range: [{lower_fence:.1f}, {upper_fence:.1f}]")

# Outliers identified:
outliers = salaries[(salaries < lower_fence) | (salaries > upper_fence)]
print("Detected Outliers:\n", outliers)

# Treat outliers by capping/clipping to the upper and lower fences:
salaries_clipped = salaries.clip(lower=lower_fence, upper=upper_fence)
print("\nSalaries after .clip(lower_fence, upper_fence):\n", salaries_clipped)


# ----------------------------------------------------------------------
# 3.10 Hands-on Practice Challenge
# ----------------------------------------------------------------------
section("3.10 Hands-on Practice Challenge")
print("""
PRACTICE EXERCISE:
Given a messy e-commerce customer transaction log:
df_messy = pd.DataFrame({
    "trans_id": [1, 2, 2, 3, 4, 5],
    "user_email": ["john@work.com", "SARA@OUTLOOK.COM", "sara@outlook.com", None, "mike@work.com", "invalid_email"],
    "amount": ["$120.50", "$45.00", "$45.00", "$999.00", "Refunded", "$35.20"],
    "discount": [10.0, np.nan, np.nan, 20.0, 0.0, np.nan]
})

Your Task:
1. Drop duplicate rows based on 'trans_id', keeping the first occurrence.
2. Remove rows where 'user_email' is missing (None/NaN) or does not contain '@'.
3. Clean 'amount': strip '$', coerce invalid values ('Refunded') to NaN, and fill NaN with 0.0.
4. Fill missing 'discount' values with 0.0.
5. Add a boolean column 'is_corporate_user' that is True if user_email ends with '@work.com'.
""")

# --- Challenge Solution ---
df_messy = pd.DataFrame({
    "trans_id": [1, 2, 2, 3, 4, 5],
    "user_email": ["john@work.com", "SARA@OUTLOOK.COM", "sara@outlook.com", None, "mike@work.com", "invalid_email"],
    "amount": ["$120.50", "$45.00", "$45.00", "$999.00", "Refunded", "$35.20"],
    "discount": [10.0, np.nan, np.nan, 20.0, 0.0, np.nan]
})

# 1. Dedup
df_clean = df_messy.drop_duplicates(subset=["trans_id"], keep="first").copy()

# 2. Email validation
df_clean = df_clean.dropna(subset=["user_email"])
df_clean = df_clean[df_clean["user_email"].str.contains("@", na=False)]
df_clean["user_email"] = df_clean["user_email"].str.lower()

# 3. Clean amount
clean_amt = df_clean["amount"].str.replace("$", "", regex=False)
df_clean["amount_num"] = pd.to_numeric(clean_amt, errors="coerce").fillna(0.0)

# 4. Fill discount
df_clean["discount"] = df_clean["discount"].fillna(0.0)

# 5. Add is_corporate_user
df_clean["is_corporate_user"] = df_clean["user_email"].str.endswith("@work.com")

print("Cleaned Production-Ready Dataset:\n",
      df_clean[["trans_id", "user_email", "amount_num", "discount", "is_corporate_user"]])
print("\n Module 3 Completed Successfully!")
