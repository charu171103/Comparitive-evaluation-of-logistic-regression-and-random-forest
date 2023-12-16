# Comparitive-evaluation-of-logistic-regression-and-random-forest

## Description
This project aims to perform a comparative evaluation of machine learning models—Logistic Regression and Random Forest—on the Wincosin Breast Cancer dataset available on Kaggle. The primary goal is to assess and compare the performance of these models in predicting breast cancer diagnosis based on various features provided in the dataset.

## Dataset
- *Source:* Kaggle
- *Dataset Name:* Wincosin Breast Cancer Dataset
- *Description:* The dataset contains information about breast cancer cases, including various features used for prediction and diagnosis.
- *Access:* You can access the dataset on Kaggle [here]([link_to_kaggle_dataset_if_public](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)) or find it in the data directory of this repository.

## Models Used
### Logistic Regression
- *Description:* Logistic Regression is a linear classification algorithm used for binary classification tasks.
- *Usage:* It's widely used in medical diagnostics, including cancer prediction, due to its simplicity and interpretability.
- *Advantages:* Offers insights into the impact of features and provides probabilities for predictions.
- *Applicability:* Effective for problems with linear decision boundaries.

### Random Forest
- *Description:* Random Forest is an ensemble learning method based on decision tree classifiers.
- *Usage:* Known for handling complex relationships in data and reducing overfitting compared to individual decision trees.
- *Advantages:* Works well with non-linear data, handles missing values, and provides feature importance.
- *Applicability:* Suitable for problems where multiple decision trees can contribute to better predictive performance.

## Dependencies
- NumPy (numpy)
- Pandas (pandas)
- Seaborn (seaborn)
- Scikit-learn (scikit-learn)
- Matplotlib (matplotlib)

## Results
### Key Findings
- *Model Performance:* 
  - Logistic Regression achieved an accuracy of 97.37%, slightly outperforming Random Forest with 96.49% accuracy.
  - Both models exhibited strong precision and recall for identifying breast cancer cases, with Logistic Regression having a slightly higher recall for class 1.
- *Evaluation Metrics:*
  - Logistic Regression:
    - Precision: Class 0 - 0.97, Class 1 - 0.98
    - Recall: Class 0 - 0.99, Class 1 - 0.95
    - F1-score: Class 0 - 0.98, Class 1 - 0.96
    - ROC-AUC Score: 0.9697
  - Random Forest:
    - Precision: Class 0 - 0.96, Class 1 - 0.98
    - Recall: Class 0 - 0.99, Class 1 - 0.93
    - F1-score: Class 0 - 0.97, Class 1 - 0.95
    - ROC-AUC Score: 0.9581
- *Insights:*
  - Logistic Regression showed a slightly better performance in overall accuracy and class 1 recall compared to Random Forest.
  - Both models demonstrated robust performance in diagnosing breast cancer, with high precision and recall rates.
