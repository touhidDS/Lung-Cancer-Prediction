# Lung Cancer Prediction (97% Accuracy)

## Overview
This repository contains a Jupyter Notebook for predicting lung cancer using a dataset from Kaggle. The notebook explores data, preprocesses features, trains multiple machine learning models, and evaluates them, achieving up to 97% accuracy with XGBoost. It's designed for binary classification (YES/NO for lung cancer).

## Dataset
- **Source**: `/kaggle/input/lung-cancer-dataset/dataset.csv` (3,000+ entries)
- **Features** (15 total, binary/numerical/categorical):
  - `GENDER` (M/F), `AGE` (numerical), `SMOKING` (1/2), `YELLOW_FINGERS` (1/2), `ANXIETY` (1/2), `PEER_PRESSURE` (1/2), `CHRONIC_DISEASE` (1/2), `FATIGUE` (1/2), `ALLERGY` (1/2), `WHEEZING` (1/2), `ALCOHOL_CONSUMING` (1/2), `COUGHING` (1/2), `SHORTNESS_OF_BREATH` (1/2), `SWALLOWING_DIFFICULTY` (1/2), `CHEST_PAIN` (1/2)
- **Target**: `LUNG_CANCER` (YES/NO)
- **Preprocessing**: Map GENDER (M=1, F=0) and LUNG_CANCER (YES=1, NO=0); no missing values.

## Approach
1. **Data Loading & Exploration**:
   - Load with Pandas; view head, info, describe.
   - Check for nulls; value counts for categoricals.
   - Visualizations: Correlation heatmap, pie chart for target balance, countplots for features vs. target.

2. **Preprocessing**:
   - Encode categorical columns.
   - Split into features (X) and target (y).
   - Train-test split (80/20).

3. **Modeling & Evaluation**:
   - Models:
     - `LogisticRegression`: Standard logistic regression.
     - `RandomForestClassifier`: 100 estimators, random_state=42.
     - `XGBClassifier`: With `use_label_encoder=False`, `eval_metric='logloss'`.
     - `KNeighborsClassifier`: k=10 neighbors.
     - `DecisionTreeClassifier`: random_state=42.
     - `GradientBoostingClassifier`: random_state=42.
     - `AdaBoostClassifier`: random_state=42.
   - Metrics: Accuracy, Precision, Recall, F1-Score.
   - Best: XGBoost (visualized with confusion matrix heatmap).

## Requirements
- Python 3.10+
- Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`

Install via:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```


## Results
- Highest Accuracy: 97% (XGBoost).
- Other metrics (e.g., Precision: 0.97, Recall: 0.98, F1: 0.97).
- Visuals: Heatmaps, pie charts, countplots, confusion matrix.

## Notes
- Optimized for Kaggle environment; adjust paths for local use.
- Imbalanced dataset (more YES cases); potential for oversampling.
- Improvements: Hyperparameter tuning, cross-validation.
