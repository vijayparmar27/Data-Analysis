"""
Module 5: Advanced NumPy & Performance
======================================
Topics Covered:
- 5.1 Vectorization efficiency: benchmarking pure Python vs. NumPy
- 5.2 Views vs. Copies: base, shares_memory, flags, and avoiding accidental mutations
- 5.3 Memory layout & strides: C-order (row-major) vs Fortran-order (column-major) & cache locality
- 5.4 Advanced broadcasting: Pairwise Euclidean distance matrix (N x M without loops)
- 5.5 Structured arrays & record arrays (heterogeneous column records like C structs/SQL rows)
- 5.6 Linear algebra basics: dot product, modern @ operator (np.matmul), vector norms
- 5.7 Matrix inversion & solving linear systems: np.linalg.solve vs np.linalg.inv
- 5.8 Matrix decompositions: Determinants, Eigenvalues (eig), and SVD (low-rank matrix compression)
- 5.9 Reproducible simulations with modern SeedSequence & independent RNG streams
- 5.10 Persistent storage: .npy, compressed .npz archives, and memory mapping (mmap_mode)
- 5.11 Comprehensive Hands-on Practice Challenge & Solution (OLS Linear Regression from Scratch via Normal Equation)
"""

import os
import tempfile
import time
import numpy as np


def section(title: str):
    print(f"\n{'=' * 65}\n  {title}\n{'=' * 65}")


# ----------------------------------------------------------------------
# 5.1 Vectorization Efficiency: Python vs NumPy Benchmark
# ----------------------------------------------------------------------
section("5.1 Vectorization Efficiency Benchmark")

# Formula: Sigmoid function 1 / (1 + exp(-x)) evaluated over 2,000,000 points
N = 2_000_000

# 1. Pure Python approach
py_data = [float(i) / 100_000 for i in range(N)]
t0 = time.perf_counter()
import math
py_result = [1.0 / (1.0 + math.exp(-x)) for x in py_data]
t_py = time.perf_counter() - t0

# 2. NumPy vectorized approach
np_data = np.arange(N, dtype=np.float64) / 100_000
t0 = time.perf_counter()
np_result = 1.0 / (1.0 + np.exp(-np_data))
t_np = time.perf_counter() - t0

print(f"Dataset Size          : {N:,} elements")
print(f"Pure Python Loop      : {t_py:.4f} seconds")
print(f"NumPy Vectorized C/SIMD: {t_np:.4f} seconds")
print(f"⚡ NumPy Speedup      : {t_py / t_np:.1f}x faster!")


# ----------------------------------------------------------------------
# 5.2 Views vs. Copies: base, shares_memory, and Flags
# ----------------------------------------------------------------------
section("5.2 Views vs. Copies Internals (.base, shares_memory)")

original = np.array([10, 20, 30, 40, 50])
slice_view = original[1:4]
fancy_copy = original[[1, 2, 3]]
explicit_copy = original[1:4].copy()

# Inspecting memory ownership:
# .base is None when the array owns its own contiguous memory buffer.
# If .base points to another ndarray, it is a VIEW.
print(f"original.base is None     : {original.base is None} (Owns memory)")
print(f"slice_view.base is original: {slice_view.base is original} (Points to original)")
print(f"fancy_copy.base is None   : {fancy_copy.base is None} (Owns independent memory)")
print(f"explicit_copy.base is None: {explicit_copy.base is None} (Owns independent memory)")

# Safe memory sharing check with np.shares_memory():
print(f"\nnp.shares_memory(original, slice_view) : {np.shares_memory(original, slice_view)}")
print(f"np.shares_memory(original, fancy_copy) : {np.shares_memory(original, fancy_copy)}")

# Array flags:
print("\nArray memory flags for slice_view:")
print(f"  OWNDATA     : {slice_view.flags['OWNDATA']}")
print(f"  C_CONTIGUOUS: {slice_view.flags['C_CONTIGUOUS']}")


# ----------------------------------------------------------------------
# 5.3 Memory Layout & Strides: C-Order vs Fortran-Order
# ----------------------------------------------------------------------
section("5.3 Memory Layout & Strides (CPU Cache Locality)")

# Strides tell NumPy how many BYTES in RAM to skip to advance 1 step in each dimension.
# In a 2D float64 (8 bytes each) array of shape (3, 4):
# - C-order (Row-Major, default in C/Python): Consecutive row elements are adjacent in RAM.
#   Strides: (32, 8) -> 32 bytes (4 elements * 8) to next row, 8 bytes to next col.
# - Fortran-order (Column-Major, used in R, MATLAB, Fortran): Consecutive col elements are adjacent in RAM.
#   Strides: (8, 24) -> 8 bytes to next row, 24 bytes (3 elements * 8) to next col.

c_arr = np.zeros((3, 4), dtype=np.float64, order='C')
f_arr = np.zeros((3, 4), dtype=np.float64, order='F')

print(f"C-order shape {c_arr.shape} strides: {c_arr.strides} bytes (Row-Major)")
print(f"F-order shape {f_arr.shape} strides: {f_arr.strides} bytes (Col-Major)")

# CPU Cache Locality Benchmark:
# Iterating along the contiguous direction is 2x to 5x faster because CPU caches pull adjacent bytes!
big_c = np.ones((5000, 5000), dtype=np.float64, order='C')

# Summing rows (contiguous in C-order):
t0 = time.perf_counter()
_ = big_c.sum(axis=1)  # moves row-by-row along cache lines
t_cache_hit = time.perf_counter() - t0

# Summing columns (striding across rows in C-order):
t0 = time.perf_counter()
_ = big_c.sum(axis=0)  # jumps 40,000 bytes per step -> more cache misses!
t_cache_miss = time.perf_counter() - t0

print(f"\nC-order row-wise sum (Cache friendly) : {t_cache_hit:.4f} s")
print(f"C-order col-wise sum (Cache jumping)  : {t_cache_miss:.4f} s")
print(f"⚡ Cache Locality speedup              : {t_cache_miss / t_cache_hit:.2f}x")


# ----------------------------------------------------------------------
# 5.4 Advanced Broadcasting: Pairwise Distance Matrix
# ----------------------------------------------------------------------
section("5.4 Advanced Broadcasting (Pairwise Distance Matrix without Loops)")

# Scenario: We have 3 customer delivery points and 2 warehouse hubs in 2D coordinates (x, y).
# Points A: shape (3, 2)
# Hubs B  : shape (2, 2)
# We want the Euclidean distance between EVERY customer and EVERY warehouse without a single loop!

customers = np.array([
    [1.0, 2.0],  # Customer 0
    [4.0, 6.0],  # Customer 1
    [7.0, 1.0],  # Customer 2
])  # shape (3, 2)

warehouses = np.array([
    [2.0, 3.0],  # Warehouse 0
    [6.0, 5.0],  # Warehouse 1
])  # shape (2, 2)

# Reshape with np.newaxis:
# customers[:, np.newaxis, :]  -> shape (3, 1, 2)
# warehouses[np.newaxis, :, :] -> shape (1, 2, 2)
# Difference broadcast shape   -> (3, 2, 2)
diff = customers[:, np.newaxis, :] - warehouses[np.newaxis, :, :]
# Distance = sqrt(sum(diff ** 2, axis=-1))
distances = np.sqrt(np.sum(diff ** 2, axis=-1))  # shape (3, 2)

print(f"Customers shape : {customers.shape}")
print(f"Warehouses shape: {warehouses.shape}")
print(f"Pairwise Distance Matrix (3 Customers x 2 Warehouses):\n{np.round(distances, 2)}")
print(f"Closest warehouse for Customer 1: Warehouse {np.argmin(distances[1])}")


# ----------------------------------------------------------------------
# 5.5 Structured Arrays & Record Arrays
# ----------------------------------------------------------------------
section("5.5 Structured Arrays & Record Arrays (C Structs / SQL Rows)")

# Structured arrays store heterogeneous data types in a single contiguous C buffer!
# Perfect for high-performance tabular storage without Python object overhead.

dtype_definition = [
    ('emp_id', np.int32),
    ('name', 'U15'),          # Unicode string up to 15 characters
    ('salary', np.float64),
    ('is_manager', np.bool_)
]

staff = np.array([
    (101, 'Alice Smith', 85000.0, True),
    (102, 'Bob Jones',   62000.0, False),
    (103, 'Charlie Ray', 74000.0, False)
], dtype=dtype_definition)

print(f"Structured array:\n{staff}")
print(f"Dtype structure : {staff.dtype}")
print(f"Salaries column : {staff['salary']}")
print(f"Row 0 record    : {staff[0]}")

# Record array (allows dot notation access e.g. rec.salary):
rec_staff = staff.view(np.recarray)
print(f"Record array dot access -> rec_staff.name: {rec_staff.name}")


# ----------------------------------------------------------------------
# 5.6 Linear Algebra Basics: Dot Product & Modern @ Operator
# ----------------------------------------------------------------------
section("5.6 Linear Algebra Basics (@ Operator, Norms)")

# 1. Vector Dot Product: sum(x_i * y_i)
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
print(f"v1 . v2 (Dot product)       : {np.dot(v1, v2)} (1*4 + 2*5 + 3*6 = 32)")

# 2. Matrix Multiplication using the @ operator (PEP 465, equivalent to np.matmul)
# A: shape (2, 3) | B: shape (3, 2) -> A @ B: shape (2, 2)
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
B = np.array([
    [7, 8],
    [9, 1],
    [2, 3]
])
product = A @ B  # or np.matmul(A, B)
print(f"\nMatrix A (2x3):\n{A}")
print(f"Matrix B (3x2):\n{B}")
print(f"Matrix Product A @ B (2x2):\n{product}")

# 3. Vector Norm (Euclidean / L2 norm: sqrt(sum(x^2))):
vector = np.array([3.0, 4.0])
l2_norm = np.linalg.norm(vector)
print(f"\nL2 Norm of [3.0, 4.0] (sqrt(3^2 + 4^2)): {l2_norm}")


# ----------------------------------------------------------------------
# 5.7 Matrix Inversion & Solving Linear Systems (solve vs inv)
# ----------------------------------------------------------------------
section("5.7 Solving Linear Systems (np.linalg.solve vs inv)")

# Problem: Solve the system of linear equations:
#   2x + 1y = 8
#   1x + 3y = 13
# In matrix form: A * [x, y]^T = b
matrix_A = np.array([[2.0, 1.0], [1.0, 3.0]])
vector_b = np.array([8.0, 13.0])

# BEST PRACTICE: Always use np.linalg.solve(A, b)!
# It uses LU decomposition (LAPACK), which is much faster, numerically stable,
# and avoids calculating the explicit inverse.
solution = np.linalg.solve(matrix_A, vector_b)
print(f"Matrix A:\n{matrix_A}")
print(f"Vector b: {vector_b}")
print(f"Solution [x, y] via np.linalg.solve: {solution}")
print(f"Verification A @ solution: {matrix_A @ solution}")

# Explicit inverse (np.linalg.inv) - USE WITH CAUTION:
inv_A = np.linalg.inv(matrix_A)
print(f"\nMatrix Inverse A^(-1):\n{inv_A}")
# Condition number: measures numerical stability (values close to 1 are well-conditioned)
print(f"Condition Number (linalg.cond): {np.linalg.cond(matrix_A):.2f}")


# ----------------------------------------------------------------------
# 5.8 Matrix Decompositions: Determinants, Eigenvalues, SVD
# ----------------------------------------------------------------------
section("5.8 Matrix Decompositions (det, eig, SVD)")

M = np.array([
    [4.0, 2.0],
    [1.0, 3.0]
])

# 1. Determinant (det):
determinant = np.linalg.det(M)
print(f"Determinant of M: {determinant:.2f} (4*3 - 2*1 = 10)")

# 2. Eigenvalues and Eigenvectors:
eigenvalues, eigenvectors = np.linalg.eig(M)
print(f"\nEigenvalues : {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# 3. Singular Value Decomposition (SVD): M = U @ diag(S) @ Vt
# Used in Principal Component Analysis (PCA), recommendation algorithms, and compression.
U, S, Vt = np.linalg.svd(M)
print(f"\nSVD Singular values: {S}")

# Reconstructing original matrix from SVD components:
reconstructed = U @ np.diag(S) @ Vt
print(f"Reconstruction close to original? {np.allclose(M, reconstructed)}")


# ----------------------------------------------------------------------
# 5.9 Reproducible Simulations & Independent RNG Streams
# ----------------------------------------------------------------------
section("5.9 Reproducible Simulations (SeedSequence & Streams)")

# In modern NumPy, SeedSequence allows generating independent, statistically non-overlapping
# random number generators for parallel / multi-process workflows!
seed_seq = np.random.SeedSequence(12345)
# Spawn 2 independent child streams
child_seeds = seed_seq.spawn(2)
rng_worker1 = np.random.default_rng(child_seeds[0])
rng_worker2 = np.random.default_rng(child_seeds[1])

draws_1 = rng_worker1.integers(1, 100, size=3)
draws_2 = rng_worker2.integers(1, 100, size=3)
print(f"Worker 1 Random Sample: {draws_1}")
print(f"Worker 2 Random Sample: {draws_2} (Independent & fully reproducible)")


# ----------------------------------------------------------------------
# 5.10 Persistent Storage: .npy, .npz, and Memory Mapping
# ----------------------------------------------------------------------
section("5.10 Persistent Storage (.npy, .npz, mmap_mode)")

with tempfile.TemporaryDirectory() as temp_dir:
    npy_path = os.path.join(temp_dir, "dataset.npy")
    npz_path = os.path.join(temp_dir, "archive.npz")

    # 1. np.save & np.load (.npy - binary format preserving dtype & shape):
    data_to_save = np.arange(1000, dtype=np.int32).reshape(100, 10)
    np.save(npy_path, data_to_save)
    loaded_npy = np.load(npy_path)
    print(f"Saved & Loaded .npy: shape={loaded_npy.shape}, dtype={loaded_npy.dtype}")

    # 2. np.savez_compressed (.npz - zip archive of multiple arrays):
    labels = np.array(["train", "test", "val"])
    np.savez_compressed(npz_path, features=data_to_save, split_labels=labels)

    with np.load(npz_path) as archive:
        print(f"Keys inside .npz archive: {archive.files}")
        print(f"Retrieved 'split_labels' : {archive['split_labels']}")

    # 3. Memory-Mapping (mmap_mode='r'):
    # Reads gigantic arrays directly from disk on-demand without loading into RAM!
    mmap_arr = np.load(npy_path, mmap_mode='r')
    print(f"\nMemory-mapped read row 5: {mmap_arr[5, :3]} (Instant, zero RAM overhead!)")


# ----------------------------------------------------------------------
# 5.11 Comprehensive Hands-on Practice Challenge: OLS Linear Regression
# ----------------------------------------------------------------------
section("5.11 Hands-on Challenge: Multiple Linear Regression from Scratch")
print("""
Applied Data Science & Linear Algebra Scenario:
Implement Ordinary Least Squares (OLS) Multiple Linear Regression from scratch
using the Normal Equation:
   β = (X^T @ X)^(-1) @ (X^T @ y)

Numerically Stable Formulation:
Solve: (X^T @ X) @ β = (X^T @ y) using np.linalg.solve()!

Your Tasks:
1. Feature Matrix Preparation:
   Given 6 real estate samples with 2 features [SquareFootage, Bedrooms]:
   Add a bias column (column of 1.0s) to X using np.column_stack.
2. Normal Equation Solver:
   Calculate weights β = [Intercept, Weight_SqFt, Weight_Bedrooms] using np.linalg.solve.
3. Make Predictions:
   Compute predicted prices: y_hat = X @ β
4. Metrics:
   Compute Root Mean Squared Error (RMSE) and R-squared (Coefficient of Determination).
5. New Property Prediction:
   Predict the price of a 1,800 sq ft, 3 bedroom home.
""")

# Features: [SquareFootage, Bedrooms]
X_raw = np.array([
    [1200.0, 2.0],
    [1500.0, 3.0],
    [2100.0, 3.0],
    [2400.0, 4.0],
    [1850.0, 3.0],
    [2900.0, 4.0]
])

# Actual Sale Prices (in $1,000s):
y = np.array([250.0, 310.0, 420.0, 480.0, 370.0, 560.0])

# Step 1: Add bias/intercept column of 1.0s
ones = np.ones((X_raw.shape[0], 1))
X = np.column_stack([ones, X_raw])
print("Step 1 - Design Matrix X (first 3 rows):\n", X[:3])

# Step 2: Solve Normal Equation (X.T @ X) @ beta = X.T @ y
XT_X = X.T @ X
XT_y = X.T @ y
beta = np.linalg.solve(XT_X, XT_y)

intercept, w_sqft, w_beds = beta
print(f"\nStep 2 - Learned Parameters (Weights β):")
print(f"  Intercept (β0)         : ${intercept:,.2f}k")
print(f"  Square Footage (β1)    : ${w_sqft:,.4f}k per sq ft")
print(f"  Bedrooms (β2)          : ${w_beds:,.2f}k per bedroom")

# Step 3: Predictions y_hat = X @ beta
y_pred = X @ beta
print(f"\nStep 3 - Actual vs Predicted Prices:")
for actual, pred in zip(y, y_pred):
    print(f"  Actual: ${actual:5.1f}k  |  Predicted: ${pred:5.1f}k  |  Error: ${actual - pred:+5.1f}k")

# Step 4: Model Evaluation Metrics (RMSE & R2)
residuals = y - y_pred
rmse = np.sqrt(np.mean(residuals ** 2))
ss_tot = np.sum((y - np.mean(y)) ** 2)
ss_res = np.sum(residuals ** 2)
r2_score = 1.0 - (ss_res / ss_tot)

print(f"\nStep 4 - Model Performance:")
print(f"  RMSE     : ${rmse:.2f}k")
print(f"  R² Score : {r2_score:.4f} (Explains {r2_score*100:.1f}% of variance!)")

# Step 5: Predict for a new property: 1,800 sq ft, 3 bedrooms
new_house = np.array([1.0, 1800.0, 3.0])  # [bias, sqft, beds]
predicted_price = new_house @ beta
print(f"\nStep 5 - Prediction for 1,800 sq ft, 3-bed home: ${predicted_price:,.2f}k (${predicted_price*1000:,.0f})")
