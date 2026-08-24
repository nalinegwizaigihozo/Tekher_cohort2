import pandas as pd

# 1. Load the public CSV dataset
df = pd.read_csv("data/raw/iris.csv")

# 2. Inspect the dataset before transforming it
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData types:")
print(df.dtypes)
print("\nDataset information:")
df.info()
print("\nMissing values:")
print(df.isna().sum())
print("\nUnique values:")
print(df.nunique())

# 3. Transform 1: group rows by species and calculate aggregates
# This gives one summary row for each species.
agg = df.groupby("species").agg(
    avg_sepal_length=("sepal_length", "mean"),
    avg_petal_length=("petal_length", "mean"),
    n=("sepal_length", "size")
).reset_index()

print("\nGrouped summary:")
print(agg)

# 4. Transform 2: merge the group summary back into the original table
# This adds species-level features to every observation.
featured = df.merge(
    agg,
    on="species",
    how="left"
)

# 5. Clean/standardize numeric values
numeric_cols = featured.select_dtypes(include="number").columns
featured[numeric_cols] = featured[numeric_cols].round(2)

# 6. Show before and after shapes
print("\nBefore transform shape:", df.shape)
print("After transform shape:", featured.shape)

# 7. Save the processed dataset
featured.to_csv("data/processed/iris_processed.csv", index=False)
print("\nSaved: data/processed/iris_processed.csv")
