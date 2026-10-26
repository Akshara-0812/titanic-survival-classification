# titanic-survival-classification
# Titanic Survival Prediction with Classical Machine Learning

Predicting passenger survival on the Titanic using Logistic Regression and
Random Forest, with exploratory analysis, feature engineering, and
cross-validation.

## Overview
The goal is to classify whether a passenger survived based on attributes such
as age, sex, ticket class, and fare, using the Titanic dataset from seaborn.

## Pipeline
1. **Data loading:** Loaded the `titanic` dataset from `seaborn`.
2. **Cleaning:** Dropped `deck` (~77% missing); imputed missing `age`,
   `embarked`, and `embark_town`.
3. **EDA:** Visualized survival by sex and ticket class.
4. **Feature engineering:** Encoded `sex` numerically, created `family_size`
   (`sibsp` + `parch` + 1), and one-hot encoded `embarked`.
5. **Modeling:** Trained Logistic Regression and Random Forest on a 70/30
   train-test split, then validated with 5-fold cross-validation.
6. **Serialization:** Saved the final model with `joblib`.

## Results

**Test set (70/30 split), initial features**

| Model | Accuracy | Precision | Recall | F1-Score |
|:---|:---:|:---:|:---:|:---:|
| Logistic Regression | 81.3% | 81.4% | 71.2% | 76.0% |
| Random Forest | 78.4% | 75.2% | 71.2% | 73.1% |

**Effect of added features and cross-validation (accuracy)**

| Model | Initial features | + Embarked & family size | 5-fold CV average |
|:---|:---:|:---:|:---:|
| Logistic Regression | 81.3% | 80.6% | 79.0% |
| Random Forest | 78.4% | 77.6% | 80.6% |

## Key Findings
- Logistic Regression scored higher on the single split, but Random Forest
  performed slightly better in cross-validation (80.6% vs 79.0%). Since the
  test split is small, cross-validation is the more reliable comparison.
- Adding `embarked` and `family_size` did not improve test accuracy for
  either model.
- Top Random Forest feature importances: Fare (27.2%), Sex (26.2%),
  Age (25.9%), consistent with the known priority given to women and
  higher-class passengers.

## Limitations
- Small dataset (891 passengers), so results vary between splits.
- Missing ages were imputed, which can introduce bias.
- No hyperparameter tuning beyond defaults [edit if you did tune].
- Feature importance in Random Forest can favor continuous features like
  Fare and Age.

## How to Run
Open in Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Akshara-0812/titanic-survival-classification/blob/main/titanic_survival_classification.ipynb)

Or locally:

    git clone https://github.com/Akshara-0812/titanic-survival-classification.git
    cd titanic-survival-classification
    pip install -r requirements.txt
    jupyter notebook

## Future Work
- Tune hyperparameters with GridSearchCV
- Try gradient boosting (e.g., XGBoost)
- Add a confusion matrix and error analysis of misclassified passengers

## Tech Stack
Python, pandas, scikit-learn, seaborn, matplotlib, joblib
