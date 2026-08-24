# Day 3 Pandas Assignment

## 1. Install the required libraries

```bash
pip install pandas scikit-learn
```

## 2. Run the assignment

From the project root:

```bash
python src/day03_pandas.py
```

## 3. Files

- `data/raw/iris.csv` — raw public dataset
- `data/processed/iris_processed.csv` — cleaned/processed dataset
- `src/day03_pandas.py` — Python solution
- `reports/day03-pandas.md` — required report

## 4. Main Pandas concepts

The solution demonstrates:

- `pd.read_csv()`
- `df.shape`
- `df.columns`
- `df.dtypes`
- `df.info()`
- `df.isna().sum()`
- `df.nunique()`
- `groupby()`
- `agg()`
- `merge()`
- `to_csv()`

## 5. Before and after

The original dataset has 150 rows and 5 columns. The processed dataset keeps the same 150 observations while adding aggregate feature columns from the species-level grouping.
