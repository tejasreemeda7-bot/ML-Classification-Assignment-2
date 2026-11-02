
import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)


# ==========================================================
# STEP 1: LOAD TRAINING AND TESTING DATA
# ==========================================================

train_data = pd.read_csv("dataset/train_data.csv")
test_data = pd.read_csv("dataset/test_data.csv")

X_train = train_data.drop(columns=["diagnosis"])
y_train = train_data["diagnosis"]

X_test = test_data.drop(columns=["diagnosis"])
y_test = test_data["diagnosis"]

print("\n========== DATA LOADED ==========")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Number of features:", X_train.shape[1])


# ==========================================================
# STEP 2: DEFINE THE FIVE CLASSIFICATION MODELS
# ==========================================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Gaussian Naive Bayes": GaussianNB(),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(random_state=42))
    ])

}


# ==========================================================
# STEP 3: CREATE FOLDERS
# ==========================================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ==========================================================
# STEP 4: TRAIN AND EVALUATE EACH MODEL
# ==========================================================

results = []

# Convert actual target values into binary values
# Malignant (M) = 1, Benign (B) = 0
y_test_binary = (y_test == "M").astype(int)

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    # Train the model
    model.fit(X_train, y_train)

    # Predict class labels
    y_pred = model.predict(X_test)

    # Obtain scores for the malignant class
    if model_name == "SVM":
        y_scores = model.decision_function(X_test)
    else:
        malignant_index = list(model.classes_).index("M")
        y_scores = model.predict_proba(X_test)[:, malignant_index]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    auc = roc_auc_score(
        y_test_binary,
        y_scores
    )

    precision = precision_score(
        y_test,
        y_pred,
        pos_label="M",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label="M",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label="M",
        zero_division=0
    )

    mcc = matthews_corrcoef(y_test, y_pred)

    # Store the results for this model
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc
    })

    # Save the trained model
    model_filename = (
        model_name.lower().replace(" ", "_") + ".joblib"
    )

    joblib.dump(
        model,
        os.path.join("models", model_filename)
    )

    print(f"{model_name} trained and saved successfully.")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"MCC: {mcc:.4f}")


# ==========================================================
# STEP 5: DISPLAY MODEL COMPARISON
# ==========================================================

results_df = pd.DataFrame(results)

print("\n========== MODEL PERFORMANCE COMPARISON ==========")

print(
    results_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)


# ==========================================================
# STEP 6: SAVE RESULTS
# ==========================================================

results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\nModel comparison saved to:")
print("results/model_comparison.csv")

print("\nAll five trained models saved successfully!")
