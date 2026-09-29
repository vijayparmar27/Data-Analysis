## Module 2: Array Indexing & Manipulation

Here is a detailed breakdown of **`matrix[0:2, 1:3]`**, explaining each part separately and then showing how they combine.

---

### The Input Array

From [02_indexing_manipulation.py](file:///Users/vijayparmar/code/Leaning/Data_Science/Data_Analysis/numpy/02_indexing_manipulation.py#L86-L91), `matrix` is a $3 \times 4$ array (3 rows, 4 columns):

```python
matrix = np.array([
    [10,  20,  30,  40],   # Row 0
    [50,  60,  70,  80],   # Row 1
    [90, 100, 110, 120]    # Row 2
])
#   Col0 Col1 Col2 Col3
```

---

### Structure of the Expression

$$\Large \mathbf{matrix}[\underbrace{0:2}_{\textbf{Part 1: Rows (Axis 0)}},\quad \underbrace{1:3}_{\textbf{Part 2: Columns (Axis 1)}}]$$

The comma `,` separates **Axis 0** (vertical dimension / rows) from **Axis 1** (horizontal dimension / columns).

---

### Part 1: Explaining `0:2` (Row Slice)

The slice follows the rule: **`start : stop`**

- **`start = 0`** $\rightarrow$ Start at index `0` (**inclusive**).
- **`stop = 2`** $\rightarrow$ Stop before index `2` (**exclusive**; does **not** include index 2).

#### Row Evaluation:

| Row Index | Condition: $0 \le \text{Index} < 2$ |    Status    | Data in that Row      |
| :-------: | :---------------------------------: | :----------: | :-------------------- |
| **Row 0** |      $0 \le 0 < 2$ is **True**      | **INCLUDED** | `[10, 20, 30, 40]`    |
| **Row 1** |      $0 \le 1 < 2$ is **True**      | **INCLUDED** | `[50, 60, 70, 80]`    |
| **Row 2** |     $0 \le 2 < 2$ is **False**      | **EXCLUDED** | `[90, 100, 110, 120]` |

> **Rows selected:** Only **Row 0** and **Row 1** (Total count: $2 - 0 = \mathbf{2}$ rows).

---

### Part 2: Explaining `1:3` (Column Slice)

The slice follows the same rule: **`start : stop`**

- **`start = 1`** $\rightarrow$ Start at index `1` (**inclusive**).
- **`stop = 3`** $\rightarrow$ Stop before index `3` (**exclusive**; does **not** include index 3).

#### Column Evaluation:

| Col Index | Condition: $1 \le \text{Index} < 3$ |    Status    |
| :-------: | :---------------------------------: | :----------: |
| **Col 0** |     $1 \le 0 < 3$ is **False**      | **EXCLUDED** |
| **Col 1** |      $1 \le 1 < 3$ is **True**      | **INCLUDED** |
| **Col 2** |      $1 \le 2 < 3$ is **True**      | **INCLUDED** |
| **Col 3** |     $1 \le 3 < 3$ is **False**      | **EXCLUDED** |

> **Columns selected:** Only **Col 1** and **Col 2** (Total count: $3 - 1 = \mathbf{2}$ columns).

---

### How Both Slices Combine (The Intersection)

NumPy takes the **intersection (Cartesian cross-product)** between the selected rows and columns:

$$\text{Selected Cells} = \{\text{Row 0}, \text{Row 1}\} \times \{\text{Col 1}, \text{Col 2}\}$$

| Coordinates `(row, col)` | Value in Matrix |
| :----------------------: | :-------------: |
|         `(0, 1)`         |     **20**      |
|         `(0, 2)`         |     **30**      |
|         `(1, 1)`         |     **60**      |
|         `(1, 2)`         |     **70**      |

- **Mathematical Point Expansion:**
  $$(0, 1) \times (1, 2) \implies [\,[(0,1), (0,2)],\; [(1,1), (1,2)]\,] \implies \begin{bmatrix} 20 & 30 \\ 60 & 70 \end{bmatrix}$$
  _(Logic: `(0, 1) _ (1, 2) => [[(0,1), (0,2)], [(1,1), (1,2)]] => [[20, 30], [60, 70]]`)\*

---

### Visual Grid Diagram

```text
                     Col 0       Col 1       Col 2       Col 3
                  (Excluded)  (INCLUDED)  (INCLUDED)  (Excluded)
                ┌───────────┬───────────┬───────────┬───────────┐
  Row 0         │    10     │  [ 20 ]   │  [ 30 ]   │    40     │
  (INCLUDED)    │           │           │           │           │
                ├───────────┼───────────┼───────────┼───────────┤
  Row 1         │    50     │  [ 60 ]   │  [ 70 ]   │    80     │
  (INCLUDED)    │           │           │           │           │
                ├───────────┼───────────┼───────────┼───────────┤
  Row 2         │    90     │   100     │   110     │   120     │
  (Excluded)    │           │           │           │           │
                └───────────┴───────────┴───────────┴───────────┘
```

Only the 4 boxed values inside the intersection are extracted.

---

### Final Output

```python
[[20, 30],
 [60, 70]]
```

- **Result Shape:** `(2, 2)` $\rightarrow$ **2 rows** by **2 columns**.
- **Formula for Output Shape:**
  - $\text{Number of rows} = \text{stop} - \text{start} = 2 - 0 = \mathbf{2}$
  - $\text{Number of columns} = \text{stop} - \text{start} = 3 - 1 = \mathbf{2}$

---

### Another Example: `matrix[0:3, 1:3]`

Here is how the slice **`matrix[0:3, 1:3]`** works step-by-step:

#### 1. Breakdown by Axis

- **Rows (`0:3`)**: Starts at index `0`, stops before index `3` ($0 \le \text{Index} < 3$).
  - **Selected rows:** `0, 1, 2` (All 3 rows)
  - _(Note: Since this selects all rows in a 3-row matrix, `matrix[:, 1:3]` produces the exact same result)._
- **Columns (`1:3`)**: Starts at index `1`, stops before index `3` ($1 \le \text{Index} < 3$).
  - **Selected columns:** `1, 2` (2 columns)

---

#### 2. Coordinate Mapping (Cartesian Product)

$$\{\text{Row 0}, \text{Row 1}, \text{Row 2}\} \times \{\text{Col 1}, \text{Col 2}\}$$

|    Row    |     Col 1 `matrix[row, 1]`     |     Col 2 `matrix[row, 2]`     |
| :-------: | :----------------------------: | :----------------------------: |
| **Row 0** | `(0, 1)` $\rightarrow$ **20**  | `(0, 2)` $\rightarrow$ **30**  |
| **Row 1** | `(1, 1)` $\rightarrow$ **60**  | `(1, 2)` $\rightarrow$ **70**  |
| **Row 2** | `(2, 1)` $\rightarrow$ **100** | `(2, 2)` $\rightarrow$ **110** |

- **Mathematical Point Expansion:**
  $$(0, 1, 2) \times (1, 2) \implies [\,[(0,1), (0,2)],\; [(1,1), (1,2)],\; [(2,1), (2,2)]\,] \implies \begin{bmatrix} 20 & 30 \\ 60 & 70 \\ 100 & 110 \end{bmatrix}$$
  _(Logic: `(0, 1, 2) _ (1, 2) => [[(0,1), (0,2)], [(1,1), (1,2)], [(2,1), (2,2)]] => [[20, 30], [60, 70], [100, 110]]`)\*

---

#### 3. Visual Grid Diagram

```text
                     Col 0       Col 1       Col 2       Col 3
                  (Excluded)  (INCLUDED)  (INCLUDED)  (Excluded)
                ┌───────────┬───────────┬───────────┬───────────┐
  Row 0         │    10     │  [ 20 ]   │  [ 30 ]   │    40     │
  (INCLUDED)    │           │           │           │           │
                ├───────────┼───────────┼───────────┼───────────┤
  Row 1         │    50     │  [ 60 ]   │  [ 70 ]   │    80     │
  (INCLUDED)    │           │           │           │           │
                ├───────────┼───────────┼───────────┼───────────┤
  Row 2         │    90     │  [ 100 ]  │  [ 110 ]  │   120     │
  (INCLUDED)    │           │           │           │           │
                └───────────┴───────────┴───────────┴───────────┘
```

---

#### 4. Final Output

```python
[[ 20,  30],
 [ 60,  70],
 [100, 110]]
```

- **Result Shape:** `(3, 2)` $\rightarrow$ **3 rows** by **2 columns**.
- **Formula for Output Shape:**
  - $\text{Number of rows} = \text{stop} - \text{start} = 3 - 0 = \mathbf{3}$
  - $\text{Number of columns} = \text{stop} - \text{start} = 3 - 1 = \mathbf{2}$
