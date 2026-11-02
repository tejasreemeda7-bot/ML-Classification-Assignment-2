
import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Define the dataset column names
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

# Step 2: Load the original dataset
df = pd.read_csv(
    "dataset/wdbc.data",
    header=None,
    names=columns
)

# Step 3: Remove the ID column
df = df.drop(columns=["id"])

# Step 4: Separate features and target
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

# Step 5: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Step 6: Combine features and target for saving
train_data = X_train.copy()
train_data["diagnosis"] = y_train

test_data = X_test.copy()
test_data["diagnosis"] = y_test

# Step 7: Save the datasets
train_data.to_csv("dataset/train_data.csv", index=False)
test_data.to_csv("dataset/test_data.csv", index=False)

# Step 8: Display the results
print("\n========== DATA SPLITTING RESULTS ==========")

print("Original dataset records:", len(df))
print("Training records:", len(train_data))
print("Testing records:", len(test_data))

print("\nTraining dataset shape:", train_data.shape)
print("Testing dataset shape:", test_data.shape)

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

print("\nTraining and testing datasets saved successfully!")
