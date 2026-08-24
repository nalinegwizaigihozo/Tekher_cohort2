# Day 3 Pandas Report

## Dataset

The dataset used for this exercise is the classic **Iris dataset**, a small public tabular dataset containing measurements of iris flowers and their species. The raw CSV is stored in `data/raw/iris.csv`.

## Schema Report

The dataset contains **150 rows** and **5 columns**.

### Data Types

- `sepal_length`: float64
- `sepal_width`: float64
- `petal_length`: float64
- `petal_width`: float64
- `species`: object

### Missing Values

All columns contain **0 missing values**.

### Unique Counts

- `sepal_length`: 35
- `sepal_width`: 23
- `petal_length`: 43
- `petal_width`: 22
- `species`: 3

## Pandas Transformations

### 1. GroupBy and Aggregation

The dataset was grouped by `species`. For each species, the code calculated the average sepal length, average petal length, and number of observations.

### 2. Merge

The grouped summary was merged back into the original dataframe using `species` as the join key. This added species-level aggregate features to each individual row.

The shape before transformation was **(150, 5)**, and the processed dataframe remains **(150, 8)** because the merge is a many-to-one merge that preserves the original observations.

## What Surprised Me?

What surprised me was how easily Pandas can summarize many rows using `groupby()` without writing manual loops. I also noticed that keeping the join key consistent is important when using `merge()`. The exercise showed me how aggregation and merging can turn a raw table into a dataset that is more useful for analysis and later machine-learning work.