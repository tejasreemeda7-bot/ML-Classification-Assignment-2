
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve
)

# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Breast Cancer Classification Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# 2. CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.main-title {
    font-size: 2.2rem;
    font-weight: 750;
    color: #16324F;
    margin-bottom: 0.2rem;
}

.subtitle {
    color: #64748b;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}

.section-title {
    font-size: 1.35rem;
    font-weight: 650;
    color: #16324F;
    margin-top: 1rem;
    margin-bottom: 0.8rem;
}

.metric-card {
    background-color: #F4F7FB;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 17px;
    min-height: 112px;
}

.metric-label {
    color: #64748b;
    font-size: 0.88rem;
    font-weight: 600;
}

.metric-value {
    color: #16324F;
    font-size: 1.8rem;
    font-weight: 750;
    margin-top: 8px;
}

div[data-testid="stMetric"] {
    background-color: #F4F7FB;
    border: 1px solid #E2E8F0;
    padding: 14px;
    border-radius: 12px;
}

[data-testid="stSidebar"] {
    background-color: #F5F8FC;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# 3. HEADER
# ==========================================================

st.markdown(
    '<div class="main-title">Breast Cancer Classification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Model Evaluation Dashboard | '
    'Wisconsin Diagnostic Dataset'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================================
# 4. MODEL FILES
# ==========================================================

MODEL_FILES = {
    "Logistic Regression": "models/logistic_regression.joblib",
    "Decision Tree": "models/decision_tree.joblib",
    "K-Nearest Neighbors": "models/knn.joblib",
    "Gaussian Naive Bayes": "models/gaussian_naive_bayes.joblib",
    "Support Vector Machine": "models/svm.joblib"
}


# ==========================================================
# 5. LOAD SAVED MODELS
# ==========================================================

@st.cache_resource
def load_models():

    loaded_models = {}

    for model_name, model_path in MODEL_FILES.items():

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Missing model file: {model_path}"
            )

        loaded_models[model_name] = joblib.load(model_path)

    return loaded_models


try:
    models = load_models()

except Exception as error:
    st.error(f"Unable to load the saved models: {error}")
    st.stop()


# ==========================================================
# 6. SIDEBAR
# ==========================================================

st.sidebar.title("Model Controls")

st.sidebar.markdown(
    "Upload your test dataset and select a classifier."
)

st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"],
    help="Upload test_data.csv containing 30 features and diagnosis."
)

selected_model_name = st.sidebar.selectbox(
    "Select Classification Model",
    list(MODEL_FILES.keys())
)

st.sidebar.divider()

st.sidebar.markdown("### Available Classifiers")

for name in MODEL_FILES:
    st.sidebar.markdown(f"✓ {name}")

st.sidebar.divider()

st.sidebar.caption(
    "The application uses saved models. "
    "No model retraining is performed."
)


# ==========================================================
# 7. WELCOME SCREEN
# ==========================================================

if uploaded_file is None:

    st.info("Please upload your test CSV using the sidebar to begin.")

    

    st.write(
        "This application compares five supervised machine learning "
        "classification algorithms on the Breast Cancer Wisconsin "
        "Diagnostic dataset."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Classification Models", "5")
    col2.metric("Input Features", "30")
    col3.metric("Dataset Records", "569")

    st.subheader("Algorithms Included")

    algorithm_data = pd.DataFrame({
        "Algorithm": [
            "Logistic Regression",
            "Decision Tree",
            "K-Nearest Neighbors",
            "Gaussian Naive Bayes",
            "Support Vector Machine"
        ],
        "Category": [
            "Linear classifier",
            "Tree-based classifier",
            "Distance-based classifier",
            "Probabilistic classifier",
            "Kernel-based classifier"
        ]
    })

    st.dataframe(
        algorithm_data,
        hide_index=True,
        use_container_width=True
    )

    st.stop()


# ==========================================================
# 8. READ AND VALIDATE TEST DATA
# ==========================================================

try:

    test_data = pd.read_csv(uploaded_file)

    if "diagnosis" not in test_data.columns:
        st.error(
            "The CSV must contain a target column named 'diagnosis'."
        )
        st.stop()

    expected_features = list(
        models["Logistic Regression"].feature_names_in_
    )

    missing_features = [
        feature for feature in expected_features
        if feature not in test_data.columns
    ]

    extra_features = [
        column for column in test_data.columns
        if column not in expected_features
        and column != "diagnosis"
    ]

    if missing_features:
        st.error(f"Missing features: {missing_features}")
        st.stop()

    if extra_features:
        st.error(f"Unexpected columns: {extra_features}")
        st.stop()

    X_test = test_data[expected_features].copy()

    y_test = (
        test_data["diagnosis"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Ensure feature columns contain valid numeric values
    for column in X_test.columns:
        X_test[column] = pd.to_numeric(
            X_test[column],
            errors="coerce"
        )

    if X_test.isnull().any().any() or y_test.isnull().any():
        st.error(
            "The uploaded dataset contains missing or invalid values."
        )
        st.stop()

    if not np.isfinite(X_test.to_numpy(dtype=float)).all():
        st.error("The dataset contains infinite numeric values.")
        st.stop()

    if not set(y_test.unique()).issubset({"B", "M"}):
        st.error(
            "Diagnosis must contain only B (Benign) or M (Malignant)."
        )
        st.stop()

    if set(y_test.unique()) != {"B", "M"}:
        st.error(
            "Both Benign and Malignant cases are required "
            "for the evaluation metrics."
        )
        st.stop()

except Exception as error:

    st.error(f"Error reading the uploaded CSV: {error}")
    st.stop()


# ==========================================================
# 9. DATASET OVERVIEW
# ==========================================================

st.markdown(
    '<div class="section-title">Dataset Overview</div>',
    unsafe_allow_html=True
)

total_records = len(test_data)
total_features = len(expected_features)
benign_count = int((y_test == "B").sum())
malignant_count = int((y_test == "M").sum())

overview1, overview2, overview3, overview4 = st.columns(4)

overview1.metric("Test Records", total_records)
overview2.metric("Input Features", total_features)
overview3.metric("Benign Cases", benign_count)
overview4.metric("Malignant Cases", malignant_count)

st.caption(
    "The uploaded dataset has been validated and is ready for evaluation."
)

st.divider()


# ==========================================================
# 10. EVALUATE ALL SAVED MODELS
# ==========================================================

all_results = {}
all_predictions = {}
all_scores = {}

y_test_binary = (y_test == "M").astype(int)

for model_name, model in models.items():

    y_pred = model.predict(X_test)

    # SVM uses decision scores for ROC-AUC
    if model_name == "Support Vector Machine":

        y_scores = model.decision_function(X_test)

    else:

        malignant_index = list(model.classes_).index("M")

        y_scores = model.predict_proba(X_test)[
            :, malignant_index
        ]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test_binary, y_scores),
        "Precision": precision_score(
            y_test, y_pred,
            pos_label="M",
            zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred,
            pos_label="M",
            zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, y_pred,
            pos_label="M",
            zero_division=0
        ),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    all_results[model_name] = metrics
    all_predictions[model_name] = y_pred
    all_scores[model_name] = y_scores


comparison_df = pd.DataFrame(all_results).transpose()


# ==========================================================
# 11. SELECTED MODEL METRICS
# ==========================================================

st.markdown(
    '<div class="section-title">Selected Model Performance</div>',
    unsafe_allow_html=True
)

st.subheader(selected_model_name)

selected_metrics = all_results[selected_model_name]

metric_names = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1 Score",
    "MCC"
]

metric_columns = st.columns(3)

for index, metric_name in enumerate(metric_names):

    metric_columns[index % 3].metric(
        metric_name,
        f"{selected_metrics[metric_name]:.4f}"
    )

st.divider()


# ==========================================================
# 12. CONFUSION MATRIX
# ==========================================================

st.markdown(
    '<div class="section-title">Confusion Matrix</div>',
    unsafe_allow_html=True
)

selected_predictions = all_predictions[selected_model_name]

cm = confusion_matrix(
    y_test,
    selected_predictions,
    labels=["B", "M"]
)

fig_cm, ax_cm = plt.subplots(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"],
    linewidths=1,
    linecolor="white",
    cbar=False,
    ax=ax_cm
)

ax_cm.set_xlabel("Predicted Diagnosis")
ax_cm.set_ylabel("Actual Diagnosis")
ax_cm.set_title(f"{selected_model_name} - Confusion Matrix")

st.pyplot(fig_cm, use_container_width=False)
plt.close(fig_cm)


# ==========================================================
# 13. CLASSIFICATION REPORT
# ==========================================================

st.markdown(
    '<div class="section-title">Classification Report</div>',
    unsafe_allow_html=True
)

report = classification_report(
    y_test,
    selected_predictions,
    labels=["B", "M"],
    target_names=["Benign", "Malignant"],
    zero_division=0,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)

st.divider()


# ==========================================================
# 14. MODEL COMPARISON
# ==========================================================

st.markdown(
    '<div class="section-title">Comparison of All Five Models</div>',
    unsafe_allow_html=True
)

st.write(
    "The following table compares all five classifiers "
    "using the same uploaded test dataset."
)

st.dataframe(
    comparison_df.style.format("{:.4f}").highlight_max(
        axis=0,
        color="#D9F2E6"
    ),
    use_container_width=True
)


# ==========================================================
# 15. PERFORMANCE COMPARISON CHARTS
# ==========================================================

st.markdown(
    '<div class="section-title">Performance Comparison Charts</div>',
    unsafe_allow_html=True
)

comparison_metrics = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1 Score",
    "MCC"
]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

for ax, metric in zip(axes.flatten(), comparison_metrics):

    values = comparison_df[metric].sort_values(ascending=True)

    bars = ax.barh(
        values.index,
        values.values,
        color="#3977A8"
    )

    ax.set_title(metric, fontweight="bold")
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Score")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    for bar in bars:

        width = bar.get_width()

        ax.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            va="center",
            fontsize=8
        )

plt.tight_layout()

st.pyplot(fig, use_container_width=True)
plt.close(fig)


# ==========================================================
# 16. ROC CURVE COMPARISON
# ==========================================================

st.markdown(
    '<div class="section-title">ROC Curve Comparison</div>',
    unsafe_allow_html=True
)

fig_roc, ax_roc = plt.subplots(figsize=(8, 6))

for model_name in models:

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_test_binary,
        all_scores[model_name]
    )

    auc_value = all_results[model_name]["AUC"]

    ax_roc.plot(
        false_positive_rate,
        true_positive_rate,
        linewidth=2,
        label=f"{model_name} (AUC = {auc_value:.4f})"
    )

ax_roc.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="Random Classifier"
)

ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curves of Five Classifiers")
ax_roc.legend(loc="lower right", fontsize=8)
ax_roc.grid(alpha=0.3)

st.pyplot(fig_roc, use_container_width=True)
plt.close(fig_roc)


# ==========================================================
# 17. FINAL OBSERVATION
# ==========================================================

st.markdown(
    '<div class="section-title">Model Evaluation Summary</div>',
    unsafe_allow_html=True
)

best_accuracy_model = comparison_df["Accuracy"].idxmax()
best_auc_model = comparison_df["AUC"].idxmax()
best_f1_model = comparison_df["F1 Score"].idxmax()

summary1, summary2, summary3 = st.columns(3)

summary1.info(f"Highest Accuracy\n\n{best_accuracy_model}")
summary2.info(f"Highest AUC\n\n{best_auc_model}")
summary3.info(f"Highest F1 Score\n\n{best_f1_model}")

st.caption(
    "Note: These results describe performance on the uploaded test dataset. "
    
)

st.divider()


