# NumPy Descriptive Statistics: Salary Dataset

A practical guide to **mean, median, variance, standard deviation, percentiles, quartiles, and outlier detection** using NumPy.

> **VS Code compatibility:** This document uses plain-text mathematical notation and Markdown tables instead of LaTeX math blocks. It will display correctly in the built-in VS Code Markdown Preview without requiring a math-rendering extension.

## 1. Dataset

```python
import numpy as np

salaries = np.array([45000, 52000, 61000, 58000, 120000, 49000, 55000])
print(f"Salaries: {salaries}")
```

|  Employee |       Salary |
| --------: | -----------: |
|         1 |      $45,000 |
|         2 |      $52,000 |
|         3 |      $61,000 |
|         4 |      $58,000 |
|         5 |     $120,000 |
|         6 |      $49,000 |
|         7 |      $55,000 |
| **Total** | **$440,000** |

- Number of observations: `n = 7`
- Sorted salaries: `[45000, 49000, 52000, 55000, 58000, 61000, 120000]`

Sorting is useful when finding the median, quartiles, and percentiles.

## 2. Mean (Average)

The **mean** is the sum of all values divided by the number of values.

### Formula

- Population mean: `μ = Σxᵢ / N`
- Sample mean: `x̄ = Σxᵢ / n`

Where:

- `xᵢ` is an individual observation.
- `N` is the population size.
- `n` is the sample size.
- `Σ` means “sum of”.

### Calculation

`Mean = (45000 + 52000 + 61000 + 58000 + 120000 + 49000 + 55000) / 7`

`Mean = 440000 / 7 = 62857.14`

**Mean salary = $62,857.14**

```python
mean_salary = np.mean(salaries)
print(f"Mean: ${mean_salary:,.2f}")
```

The $120,000 salary pulls the mean upward. The mean can be sensitive to unusually high or low values.

## 3. Median (Middle Value)

The **median** is the middle value after sorting the data.

- If the number of values is **odd**, the median is the single middle value.
- If the number of values is **even**, the median is the average of the two middle values.

### Formula

Assume the values are sorted in ascending order.

- If `n` is odd: `Median = x[(n + 1) / 2]` using **1-based positions**.
- If `n` is even: `Median = (x[n / 2] + x[n / 2 + 1]) / 2` using **1-based positions**.

### Calculation

Sorted salaries:

`[45000, 49000, 52000, 55000, 58000, 61000, 120000]`

There are seven values, so the middle position is:

`(n + 1) / 2 = (7 + 1) / 2 = 4`

The fourth value is `$55,000`.

**Median salary = $55,000**

```python
median_salary = np.median(salaries)
print(f"Median: ${median_salary:,.2f}")
```

The median is less affected by extreme values than the mean. Here, the median ($55,000) is lower than the mean ($62,857.14), partly because of the $120,000 salary.

## 4. Variance

**Variance** measures the average squared distance of values from the mean. It describes how spread out the data is.

Variance is expressed in **squared units**. For salaries measured in dollars, variance is measured in dollars squared (`$²`).

### Formula

- Population variance: `σ² = Σ(xᵢ − μ)² / N`
- Sample variance: `s² = Σ(xᵢ − x̄)² / (n − 1)`

Use population variance when the data contains the entire population of interest. Use sample variance when the data is a sample used to estimate a larger population's variance.

The `n − 1` denominator is called **Bessel's correction**. Under standard random-sampling assumptions, it makes sample variance an unbiased estimator of population variance.

### Calculation table

The mean is approximately `$62,857.14`.

| Salary (`xᵢ`) | Deviation (`xᵢ − x̄`) | Squared deviation (`(xᵢ − x̄)²`) |
| ------------: | -------------------: | ------------------------------: |
|       $45,000 |          -$17,857.14 |                  318,877,551.02 |
|       $52,000 |          -$10,857.14 |                  117,877,551.02 |
|       $61,000 |           -$1,857.14 |                    3,448,979.59 |
|       $58,000 |           -$4,857.14 |                   23,591,836.73 |
|      $120,000 |           $57,142.86 |                3,265,306,122.45 |
|       $49,000 |          -$13,857.14 |                  191,020,408.16 |
|       $55,000 |           -$7,857.14 |                   61,734,693.88 |
|       **Sum** |            **$0.00** |            **3,982,857,142.86** |

Values are rounded for display; calculations should use full precision.

### Population variance

`σ² = 3,982,857,142.86 / 7`

`σ² ≈ 568,979,591.84`

### Sample variance

` s² = 3,982,857,142.86 / (7 − 1)`

` s² ≈ 663,809,523.81`

```python
population_variance = np.var(salaries, ddof=0)
sample_variance = np.var(salaries, ddof=1)

print(f"Population variance: {population_variance:,.2f}")
print(f"Sample variance:     {sample_variance:,.2f}")
```

In NumPy, `ddof` means **delta degrees of freedom**:

| `ddof` | Divisor | Meaning                                       |
| -----: | ------: | --------------------------------------------- |
|    `0` |     `n` | Population variance convention; NumPy default |
|    `1` | `n - 1` | Sample variance convention                    |

## 5. Standard Deviation

**Standard deviation** is the square root of variance. It expresses spread in the same unit as the original data, making it easier to interpret than variance.

### Formula

- Population standard deviation: `σ = √[Σ(xᵢ − μ)² / N]`
- Sample standard deviation: `s = √[Σ(xᵢ − x̄)² / (n − 1)]`

### Calculation

Population:

`σ = √568,979,591.84 ≈ 23,853.30`

Sample:

`s = √663,809,523.81 ≈ 25,764.50`

| Measure                       |     Divisor |     Result |
| ----------------------------- | ----------: | ---------: |
| Population standard deviation |     `n = 7` | $23,853.30 |
| Sample standard deviation     | `n - 1 = 6` | $25,764.50 |

```python
population_std = np.std(salaries, ddof=0)
sample_std = np.std(salaries, ddof=1)

print(f"Population standard deviation: ${population_std:,.2f}")
print(f"Sample standard deviation:     ${sample_std:,.2f}")
```

**Interpretation:** The population standard deviation is about `$23,853`. It summarizes the dataset's spread around its mean; it does not mean every salary is exactly this far from the mean.

> Sample variance with `ddof=1` is unbiased for population variance under standard sampling assumptions. The square root of sample variance (sample standard deviation) is not, in general, an exactly unbiased estimator of population standard deviation.

## 6. Percentiles and Quartiles

A **percentile** is a value at or below which a specified percentage of observations falls, according to the selected percentile convention.

| Statistic | Percentile | Meaning                 |
| --------- | ---------: | ----------------------- |
| `Q1`      |       25th | First quartile          |
| `Q2`      |       50th | Second quartile; median |
| `Q3`      |       75th | Third quartile          |

NumPy's default `np.percentile` method is `linear`. For `n` sorted observations and percentile fraction `p`, the interpolation position is:

`h = (n − 1) × p`

Here, `p` ranges from `0` to `1`; for example, the 25th percentile uses `p = 0.25`. If `h` falls between two positions, NumPy linearly interpolates between their values. Positions in this formula are **zero-based**.

### Q1: 25th percentile

`h = (7 − 1) × 0.25 = 1.5`

Zero-based position 1 is `$49,000`; position 2 is `$52,000`.

`Q1 = 49000 + 0.5 × (52000 − 49000) = 50500`

### Q2: 50th percentile

`h = (7 − 1) × 0.50 = 3`

Zero-based position 3 is `$55,000`.

`Q2 = 55000`

### Q3: 75th percentile

`h = (7 − 1) × 0.75 = 4.5`

Zero-based position 4 is `$58,000`; position 5 is `$61,000`.

`Q3 = 58000 + 0.5 × (61000 − 58000) = 59500`

| Percentile  | Calculation                             |  Result |
| ----------- | --------------------------------------- | ------: |
| 25th (`Q1`) | Interpolate between $49,000 and $52,000 | $50,500 |
| 50th (`Q2`) | Middle value                            | $55,000 |
| 75th (`Q3`) | Interpolate between $58,000 and $61,000 | $59,500 |

```python
p25, p50, p75 = np.percentile(salaries, [25, 50, 75])

print(f"25th percentile (Q1): ${p25:,.2f}")
print(f"50th percentile (Q2): ${p50:,.2f}")
print(f"75th percentile (Q3): ${p75:,.2f}")
```

Percentile values can differ between software or settings if a different percentile method is selected. For reproducible work, record the method used.

## 7. Interquartile Range (IQR)

The **interquartile range** measures the spread of the middle 50% of the data.

### Formula

`IQR = Q3 − Q1`

### Calculation

`IQR = 59500 − 50500 = 9000`

**IQR = $9,000**

```python
iqr = p75 - p25
print(f"IQR: ${iqr:,.2f}")
```

Unlike standard deviation, IQR is based on quartiles and is generally less sensitive to extreme values.

## 8. Potential Outlier Detection with the IQR Rule

A common exploratory method flags values outside these boundaries as **potential outliers**.

### Formulas

- Lower fence: `Q1 − 1.5 × IQR`
- Upper fence: `Q3 + 1.5 × IQR`

### Calculation

`Lower fence = 50500 − (1.5 × 9000) = 37000`

`Upper fence = 59500 + (1.5 × 9000) = 73000`

| Boundary    |   Value |
| ----------- | ------: |
| Lower fence | $37,000 |
| Upper fence | $73,000 |

Any value below `$37,000` or above `$73,000` is flagged by this rule. In this dataset, `$120,000` is above the upper fence, so it is a **potential outlier**.

```python
lower_fence = p25 - 1.5 * iqr
upper_fence = p75 + 1.5 * iqr

potential_outliers = salaries[
    (salaries < lower_fence) | (salaries > upper_fence)
]

print(f"Lower fence: ${lower_fence:,.2f}")
print(f"Upper fence: ${upper_fence:,.2f}")
print(f"Potential outliers: {potential_outliers}")
```

An outlier is not automatically an error. A `$120,000` salary may be valid—for example, it may belong to a more senior employee. Investigate the context before removing or changing it.

## 9. Complete NumPy Code

```python
import numpy as np

salaries = np.array([45000, 52000, 61000, 58000, 120000, 49000, 55000])

# Central tendency
mean_salary = np.mean(salaries)
median_salary = np.median(salaries)

# Variance and standard deviation
population_variance = np.var(salaries, ddof=0)
sample_variance = np.var(salaries, ddof=1)
population_std = np.std(salaries, ddof=0)
sample_std = np.std(salaries, ddof=1)

# Percentiles and IQR
p25, p50, p75 = np.percentile(salaries, [25, 50, 75])
iqr = p75 - p25

# IQR outlier fences
lower_fence = p25 - 1.5 * iqr
upper_fence = p75 + 1.5 * iqr
potential_outliers = salaries[
    (salaries < lower_fence) | (salaries > upper_fence)
]

print(f"Salaries: {salaries}")
print(f"Mean: ${mean_salary:,.2f}")
print(f"Median: ${median_salary:,.2f}")
print(f"Population variance: {population_variance:,.2f}")
print(f"Sample variance: {sample_variance:,.2f}")
print(f"Population std dev: ${population_std:,.2f}")
print(f"Sample std dev: ${sample_std:,.2f}")
print(f"Q1 (25th percentile): ${p25:,.2f}")
print(f"Q2 (50th percentile): ${p50:,.2f}")
print(f"Q3 (75th percentile): ${p75:,.2f}")
print(f"IQR: ${iqr:,.2f}")
print(f"Lower fence: ${lower_fence:,.2f}")
print(f"Upper fence: ${upper_fence:,.2f}")
print(f"Potential outliers: {potential_outliers}")
```

## 10. Final Results Summary

| Metric                        | NumPy expression              |         Result |
| ----------------------------- | ----------------------------- | -------------: |
| Count                         | `len(salaries)`               |              7 |
| Sum                           | `np.sum(salaries)`            |       $440,000 |
| Mean                          | `np.mean(salaries)`           |     $62,857.14 |
| Median                        | `np.median(salaries)`         |        $55,000 |
| Population variance           | `np.var(salaries, ddof=0)`    | 568,979,591.84 |
| Sample variance               | `np.var(salaries, ddof=1)`    | 663,809,523.81 |
| Population standard deviation | `np.std(salaries, ddof=0)`    |     $23,853.30 |
| Sample standard deviation     | `np.std(salaries, ddof=1)`    |     $25,764.50 |
| Q1                            | `np.percentile(salaries, 25)` |        $50,500 |
| Q2                            | `np.percentile(salaries, 50)` |        $55,000 |
| Q3                            | `np.percentile(salaries, 75)` |        $59,500 |
| IQR                           | `Q3 - Q1`                     |         $9,000 |
| Potential outlier             | IQR rule                      |       $120,000 |

## 11. When to Use Each Statistic

| Statistic          | What it tells you                         | Useful when                                                       |
| ------------------ | ----------------------------------------- | ----------------------------------------------------------------- |
| Mean               | Arithmetic average                        | Values are reasonably balanced and extreme values do not dominate |
| Median             | Middle of sorted values                   | Data is skewed or contains extreme values                         |
| Variance           | Spread in squared units                   | Mathematical modeling and statistical calculations                |
| Standard deviation | Spread in original units                  | Summarizing variation around the mean                             |
| Percentile         | Position within a distribution            | Thresholds, salary bands, performance analysis                    |
| IQR                | Spread of the middle half                 | Robustly summarizing spread                                       |
| IQR fences         | Values unusually far from the middle half | Flagging observations for investigation                           |

### Key takeaway

For this salary dataset, the mean is higher than the median because of the `$120,000` observation. The IQR rule flags that salary as a potential outlier, but domain knowledge is needed to decide whether it is unusual, valid, or erroneous. Choose population or sample formulas based on what your data represents.
