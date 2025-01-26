# SAMPLING

## Introduction
Handling imbalanced datasets is a significant challenge in machine learning. This project delves into **five effective sampling techniques** that address this issue, evaluating their influence on the performance of **five popular machine learning models**. By exploring and comparing the techniques, this project aims to provide a comprehensive understanding of the role of sampling in improving model accuracy.

## Dataset Overview
The project utilizes the **Credit Card Fraud Detection Dataset**, which is publicly available. The dataset helps simulate real-world scenarios where imbalanced data often hinders model performance.

### Dataset Details:
- **Features:** Various transaction attributes.
- **Target:** Fraudulent or non-fraudulent transaction.
- **Source:**
  [Creditcard_data.csv](Creditcard_data.csv)

---

## Methodology
This project is divided into the following steps to ensure systematic analysis:

### 1. Dataset Preprocessing
The dataset is downloaded and cleaned to remove noise, normalize features, and handle missing values.

### 2. Resampling Techniques
Five resampling techniques are applied to balance the dataset and create diverse samples:
- **Simple Random Sampling**
- **Stratified Sampling**
- **Systematic Sampling**
- **Cluster Sampling**
- **Bootstrap Sampling**

### 3. Model Training
The following machine learning models are trained on the sampled datasets:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Support Vector Machine (SVM)

### 4. Evaluation
Each model is evaluated using accuracy metrics to determine the most effective sampling technique for specific machine learning approaches.

---

## Results & Findings

### Accuracy Comparison Table:

| Sampling Technique  | Logistic Regression | Decision Tree | Random Forest | XGBoost | SVM |
|---------------------|---------------------|---------------|---------------|---------|-----|
| Simple Random      | 0.883               | 0.922         | 0.974         | 0.948   | 0.909 |
| Stratified         | 0.922               | 0.961         | 0.974         | 0.974   | 0.981 |
| Systematic         | 0.839               | 0.903         | **1.000**     | 0.903   | 0.839 |
| Cluster            | **0.978**           | **1.000**     | **1.000**     | **1.000**| **1.000** |
| Bootstrap          | 0.918               | 0.974         | 0.997         | 0.993   | 0.984 |

### Key Insights:
- **Cluster Sampling** emerged as the top-performing technique, consistently achieving high accuracy across multiple models.
- **Systematic Sampling** worked particularly well with **Random Forest**, delivering perfect accuracy.
- The choice of sampling technique significantly influences model performance, underscoring its critical role in machine learning workflows.

---


## How to Run the Code
### Prerequisites:
Install the following lobraries:
```bash
pip install numpy pandas scikit-learn xgboost
```

### Steps:
1. Clone the repository:
   ```bash
   git clone <your-github-repo-link>
   cd <repo-folder>
   ```
2. Run the Python script:
   ```bash
   python sampling_assignment.py
   ```
3. View results in the console and generated output files.

---

## Discussion

The results show that **Cluster Sampling** consistently performs well across multiple models. **Random Forest**, however, achieves perfect accuracy with both **Systematic Sampling** and **Cluster Sampling**. These findings suggest that the choice of sampling technique has a significant impact on model performance.

---
## License

This project is released under the MIT License.

---
## Contributors
- NANDINI SHEKHAR
---
