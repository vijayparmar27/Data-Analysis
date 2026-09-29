# NumPy Learning Roadmap (Beginner to Advanced)

> **Context**: Tailored for a Full-Stack & Backend Developer (Python, PostgreSQL, FastAPI, TypeScript/Node.js). We focus on memory layout, vectorization, real-world data analysis, and bridging concepts to familiar database/backend equivalents.

---

## 📊 Learning Progress: `0 / 52 Topics Completed`

---

## Module 1: NumPy Fundamentals (Beginner)

_Goal: Understand memory structures, ndarray internals, creation routines, and data types._

- [ ] **1.1** What is NumPy and why use it? (CPython lists vs. C contiguous memory buffers)
- [ ] **1.2** Environment setup and verification (`numpy` in `.venv`)
- [ ] **1.3** Importing conventions & namespace (`import numpy as np`)
- [ ] **1.4** Array creation basics: `np.array()`
- [ ] **1.5** Constant & placeholder arrays: `np.zeros()`, `np.ones()`, `np.full()`, `np.empty()`
- [ ] **1.6** Array dimensions & hierarchy: 1D (vectors), 2D (matrices/tables), 3D+ (tensors)
- [ ] **1.7** Core array attributes: `shape`, `ndim`, `size`, `itemsize`, `nbytes`
- [ ] **1.8** Data types (`dtype`): int, float, bool, string, and type casting (`astype()`)
- [ ] **1.9** Numerical sequence generation: `np.arange()`, `np.linspace()`
- [ ] **1.10** Identity and diagonal matrices: `np.eye()`, `np.identity()`, `np.diag()`

---

## Module 2: Array Indexing & Manipulation (Beginner → Intermediate)

_Goal: Master data access, slicing, dimensional reshaping, and array reorganization without unnecessary memory copies._

- [ ] **2.1** Basic 1D indexing and negative indexing
- [ ] **2.2** 1D array slicing syntax (`start:stop:step`)
- [ ] **2.3** 2D and multi-dimensional indexing & slicing (`arr[row, col]`)
- [ ] **2.4** Boolean indexing & conditional masks (`arr[arr > 100]`)
- [ ] **2.5** Fancy / integer array indexing
- [ ] **2.6** Combining conditions with bitwise operators (`&`, `|`, `~`)
- [ ] **2.7** Reshaping arrays: `reshape()` and shape inference with `-1`
- [ ] **2.8** Flattening arrays: `flatten()` (copy) vs `ravel()` (view)
- [ ] **2.9** Transposing & swapping axes: `T`, `transpose()`, `swapaxes()`
- [ ] **2.10** Concatenation & stacking: `concatenate()`, `vstack()`, `hstack()`, `stack()`
- [ ] **2.11** Splitting arrays: `split()`, `vsplit()`, `hsplit()`
- [ ] **2.12** Mutating arrays: `insert()`, `delete()`, `append()`, `resize()`

---

## Module 3: Mathematical Operations & Broadcasting (Intermediate)

_Goal: Eliminate slow Python loops with vectorized ufuncs, aggregations, and broadcasting mechanics._

- [ ] **3.1** Element-wise arithmetic operations (`+`, `-`, `*`, `/`, `//`, `**`, `%`)
- [ ] **3.2** Universal functions (ufuncs): `np.sqrt()`, `np.exp()`, `np.log()`, `np.abs()`
- [ ] **3.3** Basic aggregations: `sum()`, `prod()`, `min()`, `max()`
- [ ] **3.4** Statistical aggregations: `mean()`, `median()`, `var()`, `std()`
- [ ] **3.5** Directional operations with `axis` parameter (`axis=0` columns vs `axis=1` rows)
- [ ] **3.6** Broadcasting rules & dimension compatibility
- [ ] **3.7** Broadcasting in practice (normalizing data, subtracting column means)
- [ ] **3.8** Comparison and logical operations: `np.all()`, `np.any()`, `np.isclose()`
- [ ] **3.9** Numerical rounding: `np.round()`, `np.floor()`, `np.ceil()`, `np.trunc()`
- [ ] **3.10** Cumulative operations: `cumsum()`, `cumprod()`, `diff()`

---

## Module 4: Data Processing & Business Analysis (Intermediate)

_Goal: Apply NumPy to tabular-style cleaning, filtering, missing values, and probabilistic sampling._

- [ ] **4.1** Sorting: `np.sort()`, in-place sort, and index-based sorting `np.argsort()`
- [ ] **4.2** Searching: `np.where()`, `np.nonzero()`, `np.argmax()`, `np.argmin()`
- [ ] **4.3** Unique values and value counts: `np.unique(..., return_counts=True)`
- [ ] **4.4** Handling missing & invalid data: `np.nan`, `np.isnan()`, `np.nan_to_num()`, `np.nanmean()`
- [ ] **4.5** Conditional replacement & clipping: `np.where()`, `np.select()`, `np.clip()`
- [ ] **4.6** Array-based data cleaning patterns (outlier capping, standard scaling)
- [ ] **4.7** Datetime arrays: `np.datetime64`, `np.timedelta64`, and date math
- [ ] **4.8** Modern random number generation: `np.random.default_rng()`
- [ ] **4.9** Sampling & distributions: uniform, normal, binomial, Poisson
- [ ] **4.10** Shuffling, permutations, and bootstrapping: `rng.choice()`, `rng.shuffle()`

---

## Module 5: Advanced NumPy & Performance (Advanced)

_Goal: Deep-dive into memory internals, cache locality, linear algebra, and optimized vectorization._

- [ ] **5.1** Vectorization efficiency: benchmarking pure Python vs. NumPy
- [ ] **5.2** Views vs. Copies: `base`, `.copy()`, and avoiding accidental mutations
- [ ] **5.3** Memory layout & strides: C-order (row-major) vs Fortran-order (column-major)
- [ ] **5.4** Advanced broadcasting techniques and `np.newaxis` / `None`
- [ ] **5.5** Structured arrays & record arrays (heterogeneous column records)
- [ ] **5.6** Linear algebra basics with `numpy.linalg`: dot product, matrix multiplication (`@`)
- [ ] **5.7** Matrix inversion & solving linear systems: `np.linalg.inv()`, `np.linalg.solve()`
- [ ] **5.8** Matrix decompositions: Eigenvalues (`eig()`), Determinants (`det()`), SVD
- [ ] **5.9** Reproducible simulations with RNG Seeds
- [ ] **5.10** Persistent storage: saving & loading `.npy`, compressed `.npz`, and text/CSV

---

## Module 6: Real-World Portfolio Projects (Hands-On)

_Goal: Synthesize skills into standalone analytical projects._

- [ ] **Project 1: Sales & Revenue Analysis (Beginner)**
  - Revenue aggregations, monthly trends, top-performing product identification.
- [ ] **Project 2: Customer Purchase Behavior & Segmentation (Intermediate)**
  - RFM (Recency, Frequency, Monetary) metric calculations using vectorized arrays.
- [ ] **Project 3: CSV Data Cleaning & Normalization Pipeline (Intermediate)**
  - Vectorized missing-value imputation, outlier trimming, z-score feature scaling.
- [ ] **Project 4: Statistical A/B Testing Engine (Intermediate)**
  - Two-sample hypothesis testing, conversion rate variance, p-value calculation.
- [ ] **Project 5: Inventory & Supply Demand Forecasting (Intermediate)**
  - Moving averages, safety stock buffers, stockout probability estimation.
- [ ] **Project 6: Image Array Manipulation (Advanced)**
  - Loading image as 3D ndarray `(H, W, C)`, grayscale conversion, brightness filters, slicing.
- [ ] **Project 7: Monte Carlo Business Simulation (Advanced)**
  - 10,000-iteration cash flow & risk probability forecasting.

---

## 🗓️ Recommended Schedule

| Phase       | Topics                            | Suggested Focus                                  | Time     |
| :---------- | :-------------------------------- | :----------------------------------------------- | :------- |
| **Phase 1** | Module 1: Fundamentals            | ndarray internals, creation, dtypes, attributes  | 2–3 Days |
| **Phase 2** | Module 2: Indexing & Manipulation | Slicing, boolean masks, reshape, stacking        | 2–3 Days |
| **Phase 3** | Module 3: Math & Broadcasting     | ufuncs, axis aggregations, broadcasting rules    | 2–3 Days |
| **Phase 4** | Module 4: Data Processing         | Cleaning, NaN handling, sorting, random sampling | 3–4 Days |
| **Phase 5** | Module 5: Advanced NumPy          | Strides, views vs copies, `linalg`, file I/O     | 4–5 Days |
| **Phase 6** | Module 6: Real-World Projects     | Projects 1 through 7                             | 5–7 Days |
