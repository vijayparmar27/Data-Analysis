Searched for "2.9 Transposing"
Viewed 02_indexing_manipulation.ipynb:650-720
Viewed 02_indexing_manipulation.ipynb:710-790
Viewed 02_indexing_manipulation.ipynb:780-850

In NumPy, **transposing** and **swapping axes** refer to rearranging the dimensions (axes) of an array. While in 2D linear algebra transposing simply means turning rows into columns, in multi-dimensional data science (Computer Vision, Deep Learning, Time Series) these operations are essential for aligning tensor shapes.

Here is a comprehensive breakdown of `.T`, `np.transpose()`, and `np.swapaxes()`.

---

### Conceptual Overview: How NumPy Indexes Axes

For an $N$-dimensional array, axes are $0$-indexed:

- **2D Matrix:** `axis 0` = Rows, `axis 1` = Columns
- **3D Tensor (e.g., Image):** `axis 0` = Height, `axis 1` = Width, `axis 2` = Channels (RGB)
- **4D Tensor (e.g., Video/Batch):** `axis 0` = Batch, `axis 1` = Frames/Channels, `axis 2` = Height, `axis 3` = Width

Transposition changes **how indices map to elements**:
$$\text{Original: } A[i, j, k] \quad \longrightarrow \quad \text{Transposed: } A^T[\text{permuted}(i, j, k)]$$

---

### 1. `.T` (The Transpose Attribute)

`.T` is a convenience attribute that **completely reverses the order of all axes**.

```python
import numpy as np

mat = np.array([
    [1, 2, 3],
    [4, 5, 6]
])  # shape: (2, 3) -> 2 rows, 3 cols

transposed = mat.T  # shape: (3, 2) -> 3 rows, 2 cols
# [[1, 4],
#  [2, 5],
#  [3, 6]]
```

#### Important Behaviors & Gotchas with `.T`:

1. **1D Arrays (The Biggest Beginner Trap):**

   ```python
   vec = np.array([1, 2, 3])  # shape: (3,)
   print(vec.T.shape)         # Still (3,)! Does nothing!
   ```

   **Why?** A 1D array has only 1 axis (`axis 0`). Reversing `(0,)` yields `(0,)`. To transpose a 1D vector into a column vector, you must first promote it to 2D:

   ```python
   col_vec = vec[:, np.newaxis]  # shape: (3, 1)
   print(col_vec.T.shape)        # shape: (1, 3)
   ```

2. **3D and Higher Dimensions:**
   For a 3D array of shape `(2, 5, 10)`, `.T` reverses _all_ axes:
   $$\text{Shape: } (2, 5, 10) \xrightarrow{.T} (10, 5, 2)$$
   $$\text{Axes: } (0, 1, 2) \xrightarrow{.T} (2, 1, 0)$$

---

### 2. `np.transpose(arr, axes=None)` or `arr.transpose(*axes)`

`np.transpose` is the **generalized, fully customizable** version. It lets you specify an arbitrary permutation of axes.

#### Signature:

```python
np.transpose(arr, axes=None)  # or arr.transpose((...))
```

- If `axes=None`: Exactly equivalent to `arr.T` (reverses all axes).
- If `axes` is a tuple/list: Specifies the **new order of original axis indices**.

#### Example: Computer Vision Format Conversion (HWC $\to$ CHW)

In Deep Learning:

- **OpenCV / TensorFlow** store images as `(Height, Width, Channels)` $\to$ `(H, W, C)`
- **PyTorch** expects images as `(Channels, Height, Width)` $\to$ `(C, H, W)`

```python
# Original image tensor: Height=1080, Width=1920, Channels=3
# Axis index:           0=H,         1=W,          2=C
img = np.zeros((1080, 1920, 3))

# We want new order: (Channels, Height, Width) -> (Axis 2, Axis 0, Axis 1)
chw_img = np.transpose(img, (2, 0, 1))

print(chw_img.shape)  # Output: (3, 1080, 1920)
```

#### How the `axes` Argument Works:

Each number in `axes=(a, b, c)` represents **which original axis index** moves to position 0, position 1, position 2:

| New Position | Specified Axis Index | Meaning                                         |
| :----------- | :------------------- | :---------------------------------------------- |
| Index `0`    | `2`                  | Old axis 2 (Channels) becomes the 1st dimension |
| Index `1`    | `0`                  | Old axis 0 (Height) becomes the 2nd dimension   |
| Index `2`    | `1`                  | Old axis 1 (Width) becomes the 3rd dimension    |

---

### 3. `np.swapaxes(arr, axis1, axis2)` or `arr.swapaxes(axis1, axis2)`

`np.swapaxes` is designed for situations where you want to **swap exactly two axes** and leave all other axes untouched in their original relative positions.

#### Signature:

```python
np.swapaxes(arr, axis1, axis2)  # or arr.swapaxes(axis1, axis2)
```

#### When to use `swapaxes` over `transpose`?

When dealing with higher-dimensional tensors (e.g., 4D or 5D), writing full axis tuples in `transpose()` is error-prone. If you only need to swap two dimensions, `swapaxes` is simpler and more readable.

#### Example: NLP / Time Series (Batch-First vs. Time-First)

In Sequence Models (RNNs/LSTMs/Transformers):

- Shape: `(Batch_Size, Sequence_Length, Embedding_Dim)` = `(32, 100, 512)`
- Some architectures require time-first: `(100, 32, 512)`

```python
batch_first = np.zeros((32, 100, 512))  # (Batch=0, Seq=1, Embed=2)

# Swap axis 0 (Batch) and axis 1 (Seq)
time_first = np.swapaxes(batch_first, 0, 1)

print(time_first.shape)  # Output: (100, 32, 512)
```

Equivalent with `np.transpose`:

```python
time_first = np.transpose(batch_first, (1, 0, 2))  # Does the exact same thing
```

---

### 4. Side-by-Side Comparison

| Feature           | `.T`                       | `np.transpose(arr, axes)`                       | `np.swapaxes(arr, ax1, ax2)`                                               |
| :---------------- | :------------------------- | :---------------------------------------------- | :------------------------------------------------------------------------- |
| **Scope**         | Reverses _all_ axes        | Arbitrary permutation of all axes               | Swaps _only two_ specific axes                                             |
| **Parameters**    | None                       | Tuple of new axis order (`axes`)                | Two axis indices (`axis1`, `axis2`)                                        |
| **1D Array**      | No-op (does nothing)       | No-op (does nothing)                            | Raises error if axis $\ge 1$                                               |
| **2D Matrix**     | Equivalent to transpose    | Equivalent when `axes=(1, 0)`                   | Equivalent when `axis1=0, axis2=1`                                         |
| **Best Used For** | Quick 2D matrix transposes | Full multi-axis reordering (e.g. HWC $\to$ CHW) | Swapping 2 dimensions in $N$-D tensors (e.g., Batch $\leftrightarrow$ Seq) |

---

### 5. Under the Hood: Memory Layout & Views (Crucial Internals)

#### 1. Instant $O(1)$ Operation: No Data is Copied

None of `.T`, `np.transpose()`, or `np.swapaxes()` copy data in memory. They return a **view** that points to the original memory buffer with updated **strides**.

```python
a = np.array([[1, 2, 3], [4, 5, 6]])
b = a.T

# Verify b is a view of a
print(b.base is a)  # True

# Modifying b modifies a!
b[0, 0] = 999
print(a[0, 0])      # 999
```

#### 2. Strides & Contiguity Gotcha

NumPy arrays by default are stored in row-major (**C-contiguous**) format in physical memory. Transposing inverts or permutes the strides:

```python
a = np.arange(6, dtype=np.int32).reshape(2, 3)
print(a.strides)  # (12, 4) -> 12 bytes to jump a row, 4 bytes to jump a col
print(a.flags['C_CONTIGUOUS'])  # True

a_t = a.T
print(a_t.strides)  # (4, 12) -> Strides are flipped!
print(a_t.flags['C_CONTIGUOUS'])  # False (It is now Fortran-contiguous)
```

#### Why This Matters (The `reshape` Error):

Because `a_t` is not C-contiguous, operations like `a_t.reshape(...)` may either raise a `ValueError` (in older versions) or be forced to make an unexpected **deep copy**:

```python
# To make it contiguous again explicitly:
a_contiguous = np.ascontiguousarray(a.T)
print(a_contiguous.flags['C_CONTIGUOUS'])  # True
```

---

### Summary Rule of Thumb

- Use **`.T`** for standard 2D mathematical matrices.
- Use **`np.transpose(arr, (new_order))`** when you need to rearrange multiple axes of a high-dimensional tensor (e.g., CV image pipelines).
- Use **`np.swapaxes(arr, ax1, ax2)`** when you only want to flip two specific dimensions without touching the rest (e.g., NLP sequences).
