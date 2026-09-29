"""
Module 1: NumPy Fundamentals
=============================
Topics Covered:
- 1.1 Why NumPy? (Memory layout & speed vs Python lists)
- 1.2 Creating arrays (np.array, zeros, ones, full)
- 1.3 Array dimensions (1D, 2D, 3D)
- 1.4 Core attributes (shape, ndim, size, dtype, itemsize, nbytes)
- 1.5 Data types and explicit casting (astype)
- 1.6 Sequence generators (arange, linspace)
- 1.7 Identity & special matrices (eye, diag)
- 1.8 Practice Challenge for you to solve!
"""

import time
import numpy as np


def section(title: str):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


# ----------------------------------------------------------------------
# 1.1 Why NumPy? (Speed & Memory Architecture)
# ----------------------------------------------------------------------
section("1.1 Why NumPy? (Speed Benchmark)")

# Python lists store pointers to individual PyObject items (scattered in heap memory).
# NumPy ndarrays store raw data in a single contiguous block of C memory (cache friendly!).

size = 1_000_000

# Pure Python approach
py_list = list(range(size))
t0 = time.perf_counter()
py_result = [x * 2 for x in py_list]
t_python = time.perf_counter() - t0

# NumPy vectorized approach
np_arr = np.arange(size)
t0 = time.perf_counter()
np_result = np_arr * 2
t_numpy = time.perf_counter() - t0

print(f"Pure Python list (1M items): {t_python:.4f} seconds")
print(f"NumPy ndarray    (1M items): {t_numpy:.4f} seconds")
print(f"⚡ NumPy is {t_python / t_numpy:.1f}x faster!")


# ----------------------------------------------------------------------
# 1.2 Array Creation Basics & Placeholders
# ----------------------------------------------------------------------
section("1.2 Array Creation & Placeholders")

# From Python lists:
vec_1d = np.array([10, 20, 30, 40])
print("1D array from list:\n", vec_1d)

# Common utility constructors:
zeros = np.zeros(shape=(2, 4), dtype=int)       # 2 rows, 4 columns of 0
ones = np.ones(shape=(3, 2), dtype=float)       # 3 rows, 2 columns of 1.0
filled = np.full(shape=(2, 3), fill_value=99)   # Filled with specific value

print("\nzeros(2, 4):\n", zeros)
print("\nones(3, 2):\n", ones)
print("\nfull(2, 3, fill_value=99):\n", filled)


# ----------------------------------------------------------------------
# 1.3 Array Dimensions (1D, 2D, 3D)
# ----------------------------------------------------------------------
section("1.3 Array Dimensions (1D Vector, 2D Matrix, 3D Tensor)")

# 1D: Vector (e.g. a single series of prices)
d1 = np.array([100, 105, 102, 110])

# 2D: Matrix / Table (Rows x Columns - like a SQL query result)
d2 = np.array([
    [10, 20, 30],
    [40, 50, 60]
])

# 3D: Batch or Tensor (e.g. 2 batches of 3x3 matrices, or RGB images)
d3 = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])

print("1D shape:", d1.shape, "| ndim:", d1.ndim)
print("2D shape:", d2.shape, "| ndim:", d2.ndim)
print("3D shape:", d3.shape, "| ndim:", d3.ndim)


# ----------------------------------------------------------------------
# 1.4 Core Array Attributes
# ----------------------------------------------------------------------
section("1.4 Core Array Attributes")

sample = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]])

print(f"Array:\n{sample}")
print(f"- .shape    : {sample.shape}     (rows, columns)")
print(f"- .ndim     : {sample.ndim}          (number of axes/dimensions)")
print(f"- .size     : {sample.size}          (total elements)")
print(f"- .dtype    : {sample.dtype}    (element data type)")
print(f"- .itemsize : {sample.itemsize} bytes    (bytes per element: 64 bits = 8 bytes)")
print(f"- .nbytes   : {sample.nbytes} bytes   (total memory footprint)")


# ----------------------------------------------------------------------
# 1.5 Data Types (dtype) & Type Conversion (.astype)
# ----------------------------------------------------------------------
section("1.5 Data Types & Explicit Casting (astype)")

# NumPy enforces homogeneous types (all items must share the same dtype)
int_arr = np.array([1, 2, 3, 4], dtype=np.int32)
print("Explicit int32:", int_arr, "| dtype:", int_arr.dtype)

# Convert integer array to float32 (crucial in ML/analytics to optimize memory)
float_arr = int_arr.astype(np.float32)
print("Casted to float32:", float_arr, "| dtype:", float_arr.dtype)

# String arrays to numeric
str_prices = np.array(["19.99", "25.50", "99.00"])
num_prices = str_prices.astype(np.float64)
print("Casted from strings:", num_prices, "| dtype:", num_prices.dtype)


# ----------------------------------------------------------------------
# 1.6 Sequence Generation: arange() and linspace()
# ----------------------------------------------------------------------
section("1.6 Sequence Generation (arange vs linspace)")

# np.arange(start, stop, step) - like Python's range(), supports floats
grid_step = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
print("arange(0, 10, 2):\n", grid_step)

# np.linspace(start, stop, num_points) - creates evenly spaced numbers (endpoint inclusive)
# Essential for plotting functions, simulation steps, and statistical bins
linear_space = np.linspace(0, 1, 5)  # 5 numbers from 0.0 to 1.0
print("\nlinspace(0, 1, 5):\n", linear_space)


# ----------------------------------------------------------------------
# 1.7 Identity & Diagonal Matrices
# ----------------------------------------------------------------------
section("1.7 Identity & Diagonal Matrices")

# Identity matrix (useful for linear algebra transformations)
identity = np.eye(3)
print("np.eye(3):\n", identity)

# Extract or construct diagonal
diag_matrix = np.diag([10, 20, 30])
print("\nnp.diag([10, 20, 30]):\n", diag_matrix)


# ----------------------------------------------------------------------
# 1.8 Hands-on Practice Exercise
# ----------------------------------------------------------------------
section("1.8 Practice Challenge")
print("""
Exercise 1:
1. Create a 2D array of shape (3, 4) containing float numbers from 1.0 to 12.0 using np.arange().
   (Hint: np.arange(1.0, 13.0).reshape(3, 4))
2. Print its shape, data type, and total bytes consumed in memory.
3. Cast this array to np.int16 and check how .nbytes changes.
""")

# --- Solution ---
# Step 1: Create a 2D array of shape (3, 4) with floats from 1.0 to 12.0
arr = np.arange(1.0, 13.0).reshape(3, 4)
print("Created Array:\n", arr)

# Step 2: Print shape, dtype, and total bytes consumed
print(f"\nShape: {arr.shape}")
print(f"Data Type: {arr.dtype}")
print(f"Memory consumed: {arr.nbytes} bytes ({arr.itemsize} bytes per element x {arr.size} items)")

# Step 3: Cast to np.int16 and check memory change
arr_int16 = arr.astype(np.int16)
print("\nArray cast to int16:\n", arr_int16)
print(f"New Data Type: {arr_int16.dtype}")
print(f"New Memory consumed: {arr_int16.nbytes} bytes ({arr_int16.itemsize} bytes per element x {arr_int16.size} items)")
print(f"⚡ Memory reduction: {arr.nbytes} bytes -> {arr_int16.nbytes} bytes ({arr.nbytes / arr_int16.nbytes:.0f}x less memory!)")
