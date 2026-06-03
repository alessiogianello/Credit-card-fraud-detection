# Credit Card Fraud Detection

Binary classification project to detect fraudulent credit card transactions using supervised machine learning.

## Dataset

**Source:** Credit card transactions by European cardholders, 2023  
**Size:** 568,630 records — 30 features  
**Dataset file:** `creditcard_2023.csv` (not tracked in git)

| Column | Description |
|--------|-------------|
| V1–V28 | Anonymized features (PCA-transformed) |
| Amount | Transaction amount |
| Class | Target — `0` = legitimate, `1` = fraud |

The dataset is **balanced** (50% fraud / 50% legit), with no missing values and 1 duplicate row removed.

## Project structure

```
.
├── main.ipynb       # Full analysis and model training
├── .gitignore
└── README.md
```

## Methodology

### EDA
- Descriptive statistics and data quality checks (nulls, duplicates)
- Distribution analysis per feature and by class
- Class balance verification

### Preprocessing
- Train/test split (80/20, `random_state=23`)
- Standard scaling (`StandardScaler`) applied where required

### Models

| Model | Precision | Recall | F1-Score | ROC AUC |
|-------|-----------|--------|----------|---------|
| Gaussian Naive Bayes | 0.9819 | 0.8495 | 0.9109 | 0.9776 |
| SVC linear (scaled) | 0.9819 | 0.9503 | 0.9658 | 0.9932 |
| SVC rbf (scaled) | 0.9927 | 0.9854 | 0.9890 | 0.9993 |
| SVC best (RandomizedSearchCV) | 0.9913 | 0.9697 | 0.9804 | 0.9975 |
| Decision Tree | 0.9975 | 0.9989 | 0.9982 | 0.9982 |
| Decision Tree (tuned) | 0.9940 | 0.9974 | 0.9957 | 0.9992 |
| **Random Forest** | **0.9997** | **1.0000** | **0.9999** | **0.9999** |
| Random Forest (tuned) | 0.9979 | 0.9909 | 0.9944 | 0.9998 |
| Histogram Gradient Boosting | 0.9994 | 0.9999 | 0.9997 | 0.9999 |
| Hist. GB (tuned) | 0.9981 | 0.9998 | 0.9989 | 0.9999 |
| XGBoost | 0.9995 | 1.0000 | 0.9998 | 0.9999 |
| XGBoost (tuned) | 0.9955 | 0.9982 | 0.9968 | 0.9999 |
| Neural Network | 0.9980 | 0.9989 | 0.9985 | 0.9999 |

Best overall: **Random Forest** (Precision 0.9997, Recall 1.0, F1 0.9999, ROC AUC 0.9999)

Hyperparameter tuning was performed with `RandomizedSearchCV` using PR AUC as scoring metric (more appropriate than accuracy on a fraud detection task).

### Evaluation metrics

Since accuracy is misleading on fraud detection tasks, models are evaluated on:
- **Precision** — of predicted frauds, how many are actually fraud
- **Recall** — of actual frauds, how many are correctly detected
- **F1-Score** — harmonic mean of Precision and Recall
- **ROC AUC** — area under the ROC curve
- **PR AUC** — area under the Precision-Recall curve (primary metric)

## Dependencies

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
keras / tensorflow
```
