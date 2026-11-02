
import pandas as pd
import numpy as np

# Step 1: Define the column names of the dataset
columns = [
    "id",
    "diagnosis",
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst"
]

# Step 2: Load the dataset
DATA_PATH = "dataset/wdbc.data"

df = pd.read_csv(
    DATA_PATH,
    header=None,
    names=columns
)

# Step 3: Display basic dataset information
print("\n========== DATASET OVERVIEW ==========")
print("First five rows:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nColumn data types:")
print(df.dtypes)

# Step 4: Check missing values
print("\n========== DATA QUALITY CHECK ==========")
print("Missing values in each column:")
print(df.isnull().sum())

print("\nTotal missing values:", df.isnull().sum().sum())

# Step 5: Check duplicate records
print("\nDuplicate records:", df.duplicated().sum())

# Step 6: Understand the target variable
print("\n========== TARGET VARIABLE ANALYSIS ==========")
print("Diagnosis counts:")
print(df["diagnosis"].value_counts())

print("\nDiagnosis percentages:")
print(df["diagnosis"].value_counts(normalize=True) * 100)

# Step 7: Remove the ID column
df = df.drop(columns=["id"])

# Step 8: Separate input features and target
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

print("\n========== FINAL DATASET ==========")
print("Number of records:", X.shape[0])
print("Number of input features:", X.shape[1])
print("Target variable:", y.name)
print("Target classes:", y.unique())

# Step 9: Statistical summary
print("\n========== FEATURE STATISTICS ==========")
print(X.describe())
