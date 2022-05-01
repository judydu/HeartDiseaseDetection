# Heart-Disease

**Dataset:**

This project uses Cleveland's heart disease data from the UCI Machine Learning Repository. The dataset and its attribute information can be found at: https://archive.ics.uci.edu/ml/datasets/heart+disease


**Project goals:**

(1) Detect heart disease presence by training and evaluating boosting algorithms XGBoost, LightGBM, and Catboost

(2) Demonstrate ML workflow, from data preparation and exploratory data analysis to model training and serving. Leverage Azure Machine Learning, a cloud platform for managing ML workloads, to showcase workflow by developing re-usable and production-ready Python scripts

(3) Gain a better understanding of the factors that drive heart disease


**Comments and Takeaways:**

(1) Dataset is very generalized: heart disease includes many types of heart conditions, such as heart valve disease, coronary artery disease, and more

(2) It does not come as a surprise that EDA highlights age, cholesterol, and blood pressure are some leading risk factors

(3) Because heart disease encompasses many conditions, XGBoost, LightGBM, and Catboost were trained using all the attributes in the dataset. The algorithms did not overfit on the small dataset. CatBoost outperformed XGBoost and LightGBM when it came to evaluating accuracy, precision, recall, and f1 score


