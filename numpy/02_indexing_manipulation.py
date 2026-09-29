"""
Module 2: Array Indexing & Manipulation
=======================================
Topics Covered:
- 2.1 Basic 1D indexing and negative indexing
- 2.2 1D array slicing syntax (start:stop:step) & view vs copy semantics
- 2.3 2D & multi-dimensional indexing and slicing (arr[row, col])
- 2.4 Boolean indexing & conditional masks (arr[arr > threshold])
- 2.5 Fancy / integer array indexing (indexing with coordinate lists)
- 2.6 Combining conditions with bitwise operators (&, |, ~) and np.where
- 2.7 Reshaping arrays (reshape and shape inference with -1)
- 2.8 Flattening arrays: flatten() (copy) vs ravel() (view)
- 2.9 Transposing & swapping axes (.T, transpose, swapaxes)
- 2.10 Concatenation & stacking (concatenate, vstack, hstack, stack)
- 2.11 Splitting arrays (split, vsplit, hsplit)
- 2.12 Mutating arrays (insert, delete, append, resize)
- 2.13 Comprehensive Practice Challenge & Solution
"""

import numpy as np


def section(title: str):
    print(f"\n{'=' * 65}\n  {title}\n{'=' * 65}")


# ----------------------------------------------------------------------
# 2.1 Basic 1D Indexing & Negative Indexing
# ----------------------------------------------------------------------
section("2.1 Basic 1D Indexing & Negative Indexing")

arr_1d = np.array([10, 25, 40, 55, 70, 85, 100])
print(f"Original 1D array: {arr_1d}")

# Zero-based indexing from the front
print(f"First element (index 0)   : {arr_1d[0]}")
print(f"Fourth element (index 3)  : {arr_1d[3]}")

# Negative indexing from the end (-1 is last, -2 is second to last)
print(f"Last element (index -1)   : {arr_1d[-1]}")
print(f"Second to last (index -2) : {arr_1d[-2]}")

# In-place element modification via index
arr_1d[0] = 999
print(f"After modifying index 0   : {arr_1d}")
arr_1d[0] = 10  # restore


# ----------------------------------------------------------------------
# 2.2 1D Array Slicing Syntax (start:stop:step) & Views vs Copies
# ----------------------------------------------------------------------
section("2.2 1D Array Slicing Syntax & View vs Copy Semantics")

# Syntax: arr[start:stop:step] (stop is EXCLUSIVE)
data = np.arange(10, 110, 10)  # [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
print(f"Original array: {data}")

print(f"Slice data[2:6]     (indices 2..5)     : {data[2:6]}")
print(f"Slice data[:4]      (first 4 elements) : {data[:4]}")
print(f"Slice data[6:]      (from index 6 on)  : {data[6:]}")
print(f"Slice data[::2]     (every 2nd item)   : {data[::2]}")
print(f"Slice data[::-1]    (reversed array)   : {data[::-1]}")
print(f"Slice data[-4:-1]   (negative slice)   : {data[-4:-1]}")

# CRITICAL NUMPY CONCEPT: Slices create VIEWS, not copies!
# Modifying a slice directly mutates the original parent array in memory.
slice_view = data[0:3]
slice_view[:] = -1
print("\n⚠️ Mutated slice_view[:] = -1")
print(f"Original array is modified!: {data}")

# To prevent modifying original, explicitly use .copy()
data[0:3] = [10, 20, 30]  # restore
safe_copy = data[0:3].copy()
safe_copy[:] = 0
print(f"Original array after modifying .copy(): {data} (safe & unchanged!)")


# ----------------------------------------------------------------------
# 2.3 2D & Multi-Dimensional Indexing and Slicing
# ----------------------------------------------------------------------
section("2.3 2D & Multi-Dimensional Indexing and Slicing")

# In standard Python lists: nested_list[row][col] (inefficient double lookup)
# In NumPy: arr[row, col] (single optimized memory lookup)
matrix = np.array([
    [10, 20, 30, 40],
    [50, 60, 70, 80],
    [90, 100, 110, 120]
])
print(f"2D Matrix (3x4):\n{matrix}")

# Single element access
print(f"\nElement at row 1, col 2 -> matrix[1, 2]: {matrix[1, 2]}")
print(f"Bottom-right element    -> matrix[-1, -1]: {matrix[-1, -1]}")

# Slicing rows & columns: matrix[row_slice, col_slice]
# Use ':' alone to select all elements along that axis
print(f"\nAll columns of Row 0    -> matrix[0, :] : {matrix[0, :]}")
print(f"All rows of Column 2    -> matrix[:, 2] : {matrix[:, 2]}")
print(f"Sub-matrix (Rows 0-1, Cols 1-2):\n{matrix[0:2, 1:3]}")
print(f"Every other row & col   -> matrix[::2, ::2]:\n{matrix[::2, ::2]}")

# 3D Tensor Indexing: shape (batch, row, col)
tensor = np.arange(1, 13).reshape(2, 2, 3)
print(f"\n3D Tensor (2 batches, 2 rows, 3 cols):\n{tensor}")
print(f"Batch 0, Row 1, Col 2 -> tensor[0, 1, 2]: {tensor[0, 1, 2]}")
print(f"All batches, Col 0 only -> tensor[:, :, 0]:\n{tensor[:, :, 0]}")


# ----------------------------------------------------------------------
# 2.4 Boolean Indexing & Conditional Masks
# ----------------------------------------------------------------------
section("2.4 Boolean Indexing & Conditional Masks")

scores = np.array([45, 88, 72, 95, 30, 64, 82])
print(f"Scores: {scores}")

# Step 1: Create a boolean condition mask
passing_mask = scores >= 70
print(f"Boolean mask (scores >= 70): {passing_mask}")

# Step 2: Filter array using the mask (returns a 1D copy of matching elements)
passing_scores = scores[passing_mask]
print(f"Passing scores: {passing_scores}")

# Direct inline masking:
high_scorers = scores[scores > 85]
print(f"High scorers (>85): {high_scorers}")

# Conditional modification (e.g. ReLU activation / clipping values)
raw_signals = np.array([-5, 12, -20, 45, 0, -8, 30])
print(f"\nRaw signals: {raw_signals}")
raw_signals[raw_signals < 0] = 0  # clamp negative values to 0
print(f"Signals clamped (negatives set to 0): {raw_signals}")


# ----------------------------------------------------------------------
# 2.5 Fancy / Integer Array Indexing
# ----------------------------------------------------------------------
section("2.5 Fancy / Integer Array Indexing")

# Fancy indexing = passing a list/array of integer indices to select elements.
# IMPORTANT: Unlike slicing, Fancy Indexing ALWAYS creates a COPY!
values = np.array([100, 200, 300, 400, 500, 600, 700])
indices = [0, 2, 4, 6]
print(f"Array: {values}")
print(f"Selecting indices {indices}: {values[indices]}")

# Permuting / re-ordering elements:
reordered = values[[3, 0, 2, 1]]
print(f"Re-ordered with [3, 0, 2, 1]: {reordered}")

# 2D Fancy Indexing:
grid = np.arange(10, 100, 10).reshape(3, 3)
print(f"\nGrid 3x3:\n{grid}")

# Select specific rows in order:
print(f"Select row 2 then row 0:\n{grid[[2, 0]]}")

# Select coordinate pairs: (row 0, col 1) and (row 2, col 2)
# Pass row indices as 1st list, col indices as 2nd list
coord_values = grid[[0, 2], [1, 2]]
print(f"Points at (0, 1) and (2, 2) -> grid[[0, 2], [1, 2]]: {coord_values}")


# ----------------------------------------------------------------------
# 2.6 Combining Conditions with Bitwise Operators (&, |, ~) and np.where
# ----------------------------------------------------------------------
section("2.6 Combining Conditions (&, |, ~) and np.where")

prices = np.array([12, 45, 78, 110, 150, 25, 95])
print(f"Prices: {prices}")

# NOTE: Python's standard 'and', 'or', 'not' keywords DO NOT work on NumPy arrays!
# You MUST use bitwise operators:
#   & : Element-wise AND
#   | : Element-wise OR
#   ~ : Element-wise NOT (inversion)
# IMPORTANT: Parentheses around EACH condition are mandatory due to operator precedence!

budget_items = prices[(prices >= 30) & (prices <= 100)]
print(f"Items between $30 and $100: {budget_items}")

extreme_items = prices[(prices < 20) | (prices > 120)]
print(f"Items < $20 OR > $120: {extreme_items}")

inverted = prices[~(prices > 50)]  # NOT (> 50) => <= 50
print(f"Items NOT > $50: {inverted}")

# np.where(condition, value_if_true, value_if_false): Vectorized ternary operator
labels = np.where(prices >= 100, "Expensive", "Affordable")
print(f"np.where classification:\n{list(zip(prices, labels))}")


# ----------------------------------------------------------------------
# 2.7 Reshaping Arrays: reshape() and Shape Inference with -1
# ----------------------------------------------------------------------
section("2.7 Reshaping Arrays & Shape Inference (-1)")

flat = np.arange(1, 13)
print(f"Flat array (12 items): {flat}")

# Reshape to (3, 4) - 3 rows, 4 columns
matrix_3x4 = flat.reshape(3, 4)
print(f"Reshaped to (3, 4):\n{matrix_3x4}")

# Reshape to (2, 6)
print(f"Reshaped to (2, 6):\n{flat.reshape(2, 6)}")

# Automatic dimension inference using -1:
# NumPy automatically calculates the missing dimension based on total size!
auto_cols = flat.reshape(4, -1)   # 4 rows, NumPy infers cols = 12 / 4 = 3
print(f"Reshaped with (4, -1) -> inferred shape {auto_cols.shape}:\n{auto_cols}")

# Convert 1D vector to 2D column vector (crucial for ML feature inputs!)
col_vector = flat.reshape(-1, 1)  # shape (12, 1)
print(f"Column vector shape (-1, 1): {col_vector.shape}")


# ----------------------------------------------------------------------
# 2.8 Flattening Arrays: flatten() (copy) vs ravel() (view)
# ----------------------------------------------------------------------
section("2.8 Flattening: flatten() (copy) vs ravel() (view)")

original = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Original 2D array:\n{original}")

# ravel() creates a flattened VIEW whenever possible (fast, no memory duplicated)
rav = original.ravel()
# flatten() ALWAYS creates a new, independent COPY in memory
flat_copy = original.flatten()

print(f"ravel()   : {rav} | base is original? {rav.base is original}")
print(f"flatten() : {flat_copy} | base is original? {flat_copy.base is original}")

# Proof of mutation:
rav[0] = 999
print(f"\nAfter modifying rav[0] = 999:")
print(f"Original array is modified!: {original[0, 0]} (because ravel is a view!)")

flat_copy[1] = 888
print(f"After modifying flat_copy[1] = 888:")
print(f"Original array is UNCHANGED: {original[0, 1]} (because flatten is a copy!)")
original[0, 0] = 1  # restore


# ----------------------------------------------------------------------
# 2.9 Transposing & Swapping Axes: .T, transpose(), swapaxes()
# ----------------------------------------------------------------------
section("2.9 Transposing & Swapping Axes (.T, transpose, swapaxes)")

mat = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
print(f"Original Matrix shape {mat.shape}:\n{mat}")

# Transpose flips rows into columns (shape: 2x3 -> 3x2)
transposed = mat.T  # or np.transpose(mat)
print(f"Transposed (.T) shape {transposed.shape}:\n{transposed}")

# Multi-dimensional tensor transposition (common in computer vision: (H, W, C) -> (C, H, W))
img_tensor = np.zeros(shape=(1080, 1920, 3))  # Height, Width, Channels
print(f"\nImage tensor shape (H, W, C): {img_tensor.shape}")

# Reorder axes using np.transpose(arr, axes)
chw_tensor = np.transpose(img_tensor, (2, 0, 1))  # (Channels, Height, Width)
print(f"PyTorch format (C, H, W)     : {chw_tensor.shape}")

# swapaxes() swaps exactly two specific axes
swapped = np.swapaxes(mat, 0, 1)
print(f"swapaxes(0, 1) matches .T    : {np.array_equal(swapped, mat.T)}")


# ----------------------------------------------------------------------
# 2.10 Concatenation & Stacking: concatenate, vstack, hstack, stack
# ----------------------------------------------------------------------
section("2.10 Concatenation & Stacking")

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

print(f"Array A:\n{a}")
print(f"Array B:\n{b}")

# concatenate along existing axis:
concat_row = np.concatenate([a, b], axis=0)  # vertical (extend rows)
concat_col = np.concatenate([a, b], axis=1)  # horizontal (extend cols)
print(f"\nconcatenate(axis=0) [extend rows]:\n{concat_row}")
print(f"concatenate(axis=1) [extend cols]:\n{concat_col}")

# Convenience helpers:
# np.vstack: vertical stack (row-wise)
# np.hstack: horizontal stack (column-wise)
print(f"\nnp.vstack([a, b]):\n{np.vstack([a, b])}")
print(f"np.hstack([a, b]):\n{np.hstack([a, b])}")

# np.stack creates a BRAND NEW dimension/axis!
# E.g. stacking two 2D arrays along axis=0 yields a 3D array of shape (2, 2, 2)
stacked_new_axis = np.stack([a, b], axis=0)
print(f"\nnp.stack([a, b], axis=0) -> shape {stacked_new_axis.shape}:\n{stacked_new_axis}")

# Stacking 1D arrays into columns (useful when assembling dataset feature tables):
f1 = np.array([1, 2, 3])
f2 = np.array([4, 5, 6])
table = np.column_stack([f1, f2])
print(f"\nnp.column_stack([f1, f2]):\n{table}")


# ----------------------------------------------------------------------
# 2.11 Splitting Arrays: split, vsplit, hsplit
# ----------------------------------------------------------------------
section("2.11 Splitting Arrays (split, vsplit, hsplit)")

matrix_to_split = np.arange(16).reshape(4, 4)
print(f"Matrix (4x4):\n{matrix_to_split}")

# Split into 2 equal parts vertically (row-wise) -> 2 matrices of shape (2, 4)
top, bottom = np.vsplit(matrix_to_split, 2)
print(f"\nvsplit (Top half):\n{top}")
print(f"vsplit (Bottom half):\n{bottom}")

# Split into 2 equal parts horizontally (col-wise) -> 2 matrices of shape (4, 2)
left, right = np.hsplit(matrix_to_split, 2)
print(f"\nhsplit (Left half):\n{left}")
print(f"hsplit (Right half):\n{right}")

# Real-world ML application: Splitting features X and target label y
dataset = np.array([
    [5.1, 3.5, 1.4, 0],
    [4.9, 3.0, 1.4, 0],
    [6.2, 3.4, 5.4, 1],
    [5.9, 3.0, 5.1, 1],
])
# Split all columns up to index 3 as X, index 3 onwards as y
X, y = np.hsplit(dataset, [3])
print(f"\nDataset feature matrix X shape {X.shape}:\n{X}")
print(f"Dataset target labels y shape {y.shape}:\n{y.ravel()}")


# ----------------------------------------------------------------------
# 2.12 Mutating Arrays: insert, delete, append, resize
# ----------------------------------------------------------------------
section("2.12 Mutating Arrays (insert, delete, append, resize)")

base_arr = np.array([10, 20, 30, 40, 50])
print(f"Base array: {base_arr}")

# np.append returns a NEW array (NumPy arrays have fixed memory buffers)
appended = np.append(base_arr, [60, 70])
print(f"np.append: {appended}")

# np.insert(array, index, values, axis=None)
inserted = np.insert(base_arr, 2, [999, 888])  # insert at index 2
print(f"np.insert at index 2: {inserted}")

# np.delete(array, obj, axis=None)
deleted = np.delete(base_arr, [1, 3])  # delete indices 1 and 3
print(f"np.delete indices 1 & 3: {deleted}")

# np.resize returns a new array with repeated elements if size increases
resized = np.resize(base_arr, (2, 4))
print(f"np.resize to (2, 4) [repeats data if needed]:\n{resized}")


# ----------------------------------------------------------------------
# 2.13 Comprehensive Practice Challenge & Solution
# ----------------------------------------------------------------------
section("2.13 Hands-on Practice Challenge")
print("""
Real-World Data Cleaning & Preprocessing Scenario:
You are given a raw sensor telemetry stream containing 24 readings from 2 IoT sensors:
- The stream represents 6 time steps.
- Each time step recorded 4 measurements: [Sensor1_Temp, Sensor1_Pressure, Sensor2_Temp, Sensor2_Pressure].

Your Tasks:
1. Reshape the 1D raw stream into a 2D matrix of shape (6, 4).
2. Corrupted Sensor readings: Any negative value is faulty sensor noise. Replace all negative values with 0.0 using boolean masking.
3. Feature Extraction:
   - Extract only Sensor 1's data (Cols 0 and 1) into `sensor_1`.
   - Extract only Sensor 2's data (Cols 2 and 3) into `sensor_2`.
4. High Temperature Alert:
   - Find all time steps (rows) where EITHER Sensor 1 Temp (Col 0) > 40 OR Sensor 2 Temp (Col 2) > 40.
5. Merge & Stack:
   - Add a newly arrived time step readings [35.0, 101.2, 42.0, 100.8] to the bottom of the cleaned 2D matrix.
""")

# --- Solution ---
raw_stream = np.array([
    22.5, 101.3, 24.1, 101.1,
    -9.9, 100.8, 25.0, 100.9,   # faulty negative reading
    38.2, 102.1, 41.5, 101.8,   # Sensor 2 Temp > 40
    42.0, 103.0, 39.1, 102.2,   # Sensor 1 Temp > 40
    -1.0, 100.2, -5.0,  99.8,   # multiple faulty readings
    29.4, 101.5, 30.1, 101.4
])

# Step 1: Reshape into (6, 4)
telemetry = raw_stream.reshape(6, -1)
print("Step 1 - Reshaped Telemetry (6, 4):\n", telemetry)

# Step 2: Clean corrupted values (< 0 -> 0.0)
faulty_mask = telemetry < 0
telemetry[faulty_mask] = 0.0
print("\nStep 2 - Telemetry after cleaning negatives:\n", telemetry)

# Step 3: Extract Sensor 1 & Sensor 2 features
sensor_1 = telemetry[:, :2]  # cols 0, 1
sensor_2 = telemetry[:, 2:]  # cols 2, 3
print("\nStep 3 - Sensor 1 (Cols 0, 1):\n", sensor_1)
print("Step 3 - Sensor 2 (Cols 2, 3):\n", sensor_2)

# Step 4: High Temperature Alert (Col 0 > 40 OR Col 2 > 40)
high_temp_rows = telemetry[(telemetry[:, 0] > 40) | (telemetry[:, 2] > 40)]
print("\nStep 4 - Time steps with High Temp Alert (>40°C on either sensor):\n", high_temp_rows)

# Step 5: Append new reading
new_reading = np.array([[35.0, 101.2, 42.0, 100.8]])
final_telemetry = np.vstack([telemetry, new_reading])
print("\nStep 5 - Final Telemetry with new reading stacked:\n", final_telemetry)
print(f"Final shape: {final_telemetry.shape}")
