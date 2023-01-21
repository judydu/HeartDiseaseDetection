# Heart Disease Detection

**Dataset:**

This project uses Cleveland's heart disease dataset from the UCI Machine Learning Repository. The dataset and its attribute information can be found at: https://archive.ics.uci.edu/ml/datasets/heart+disease

<br>

**Project Objectives:**

(1) Gain a better understanding of the factors that drive heart disease

(2) Detect heart disease presence by training and evaluating boosting algorithms XGBoost, LightGBM, and Catboost

(3) Demonstrate end-to-end ML workflow, from data preparation and exploratory data analysis to model training and serving

<br>

**Results:**

(1) It does not come as a surprise that EDA highlights age, cholesterol, and blood pressure are some leading risk factors

(2) The dataset is very generalized in a sense that heart disease includes many types of heart conditions, such as heart valve disease, coronary artery disease, and more

(3) Because heart disease encompasses many conditions, XGBoost, LightGBM, and Catboost were trained using all the features in the dataset. CatBoost outperformed XGBoost and LightGBM when it came to evaluating accuracy, precision, recall, and f1 score. It achieved an accuracy of 86% and predicted each class well