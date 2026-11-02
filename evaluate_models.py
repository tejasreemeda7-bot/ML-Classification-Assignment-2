
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# --------------------------------------------------
# 1. Load the testing dataset
# --------------------------------------------------

test_data = pd.read_csv("dataset/test_data.csv")

X_test = test_data.drop("diagnosis", axis=1)
y_test = test_data["diagnosis"]

# Class labels:
# B = Benign
# M = Malignant

class_labels = ["B", "M"]

# --------------------------------------------------
# 2. Define the saved models
# --------------------------------------------------

model_files = {
    "Logistic Regression": "models/logistic_regression.joblib",
    "Decision Tree": "models/decision_tree.joblib",
    "KNN": "models/knn.joblib",
    "Gaussian Naive Bayes": "models/gaussian_naive_bayes.joblib",
    "SVM": "models/svm.joblib"
}

# Create folders for evaluation outputs
os.makedirs("results/confusion_matrices", exist_ok=True)
os.makedirs("results/classification_reports", exist_ok=True)

# --------------------------------------------------
# 3. Load and evaluate each saved model
# --------------------------------------------------

for model_name, model_path in model_files.items():

    print("\n" + "=" * 55)
    print(f"EVALUATING: {model_name}")
    print("=" * 55)

    # Load the previously trained model
    model = joblib.load(model_path)

    # Predict the test data
    y_pred = model.predict(X_test)

    # --------------------------------------------------
    # 4. Generate confusion matrix
    # --------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=class_labels
    )

    print("\nConfusion Matrix:")
    print(cm)

    # Plot and save confusion matrix
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign (B)", "Malignant (M)"]
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    image_name = (
        model_name.lower().replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(
        os.path.join("results/confusion_matrices", image_name)
    )

    plt.close()

    # --------------------------------------------------
    # 5. Generate classification report
    # --------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        labels=class_labels,
        target_names=["Benign", "Malignant"],
        zero_division=0,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    print("\nClassification Report:")
    print(report_df.round(4))

    # Save classification report
    report_name = (
        model_name.lower().replace(" ", "_")
        + "_classification_report.csv"
    )

    report_df.to_csv(
        os.path.join("results/classification_reports", report_name)
    )

    print("\nEvaluation files saved successfully.")

print("\nAll five saved models evaluated successfully!")
