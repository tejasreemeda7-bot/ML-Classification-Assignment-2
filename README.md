# Breast Cancer Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to develop and evaluate machine learning classification models to predict whether a breast tumour is **Benign (B)** or **Malignant (M)** using the Breast Cancer Wisconsin (Diagnostic) dataset.

Five supervised machine learning algorithms are implemented and compared:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The models are evaluated using Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

## 2. Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic)

**Source:** UCI Machine Learning Repository  
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

The dataset contains 569 instances and 30 numerical input features computed from digitized images of fine needle aspirates of breast masses.

The target variable is `diagnosis`, with two classes:

- B – Benign
- M – Malignant

### Dataset details

| Property | Description |
|---|---|
| Total instances | 569 |
| Input features | 30 |
| Target variable | diagnosis |
| Classification type | Binary classification |
| Benign instances | 357 |
| Malignant instances | 212 |
| Missing values | 0 |
| Duplicate records | 0 |

The ID column was removed because it is an identifier rather than a predictive feature.

### Data splitting

The dataset was divided into training and testing sets using an 80:20 split.

- Training instances: 455
- Testing instances: 114
- Random state: 42
- Stratification: Enabled

Stratification was used to preserve the class distribution in both subsets.

## 3. GitHub Repository Link

**GitHub Repository:** [Add your GitHub repository URL after publishing]

## 4. Machine Learning Models and Performance Comparison

All five classifiers were trained using the same training dataset and evaluated using the same independent test dataset.

Feature scaling was applied to Logistic Regression, KNN, and SVM using StandardScaler. Decision Tree and Gaussian Naive Bayes were trained without feature scaling.

Malignant (M) was treated as the positive class for calculating Precision, Recall, and F1 Score. AUC was calculated using predicted probabilities or decision scores.

### Performance comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| SVM | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

### Observations

**1. Logistic Regression**

Logistic Regression achieved an accuracy of 96.49% and the highest AUC score of 0.9960. It also achieved a recall of 92.86%, indicating that it correctly identified a high proportion of malignant cases. It provides strong overall classification performance.

**2. Decision Tree**

The Decision Tree achieved an accuracy of 92.98%. Its precision, recall, and F1 Score were all 0.9048. It performed reasonably well but had lower overall performance than the other models on this test dataset.

**3. K-Nearest Neighbors (KNN)**

KNN achieved an accuracy of 95.61%, with an AUC of 0.9823. Its precision was 0.9744, while its recall was 0.9048. The results indicate good classification performance, although it missed four malignant cases.

**4. Gaussian Naive Bayes**

Gaussian Naive Bayes achieved an accuracy of 93.86% and a precision of 1.0000. However, its recall was 0.8333, the lowest among the five models. It missed seven malignant cases, indicating that high precision alone does not guarantee strong performance in this classification problem.

**5. Support Vector Machine (SVM)**

SVM achieved the highest accuracy of 97.37%, an F1 Score of 0.9630, and an MCC of 0.9442. Its precision was 1.0000 and recall was 0.9286. It provided the strongest overall balance of classification metrics in this experiment.

### Overall conclusion

Based on the held-out test results, **Support Vector Machine (SVM)** was the overall best-performing model. It achieved the highest accuracy, F1 Score, and MCC, while also achieving perfect precision.

Logistic Regression achieved the highest AUC score of 0.9960 and the same recall as SVM.

Therefore, SVM is the strongest overall performer on this particular test split, while Logistic Regression is also a competitive model.

These results are based on one train-test split and should not be interpreted as clinical validation.

## 5. Project Structure

```text
ML_Assignment_2/
│
├── dataset/
│   ├── test_data.csv
│   ├── wdbc.data
│   └── wdbc.names
│
├── models/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── gaussian_naive_bayes.joblib
│   └── svm.joblib
│
├── results/
│   ├── model_comparison.csv
│   ├── confusion_matrices/
│   └── classification_reports/
│
├── data_exploration.py
├── prepare_data.py
├── train_models.py
├── evaluate_models.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 6. Streamlit Application

**Live Application:** [Add your Streamlit Community Cloud URL after deployment]

The Streamlit application provides an interactive interface for evaluating the five trained classification models.

### Application features

- Upload a test dataset in CSV format.
- Select one of the five saved classification models.
- Display Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
- Display the confusion matrix and classification report.
- Compare the performance of all five models.
- Visualize model performance through charts and ROC curves.

The application loads previously trained models from the repository. It evaluates the uploaded test data without retraining the models.

## 7. Installation and Execution

### Step 1: Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ML_Assignment_2
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Streamlit application

```bash
streamlit run app.py
```

### Step 4: Upload test data

Upload `dataset/test_data.csv` through the application and select the desired classification model.

## 8. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

## 9. Conclusion

This project demonstrates the implementation and comparison of five supervised machine learning classification algorithms using the Breast Cancer Wisconsin (Diagnostic) dataset.

The models were evaluated using six performance metrics, and their results were visualized through confusion matrices, classification reports, and comparison charts.

SVM achieved the strongest overall performance on the selected test dataset, while Logistic Regression achieved the highest AUC score.

The trained models were saved and integrated into a Streamlit application to provide an interactive model evaluation interface.