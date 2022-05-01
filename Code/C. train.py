# train.py

### Load packages
import os
import argparse
from azureml.core import Run
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from joblib import dump


### Set seed for reproducibility
rng = np.random.RandomState(123)


### Get experiment run context
run = Run.get_context()


### Get arguments
parser = argparse.ArgumentParser()
parser.add_argument("--prepped_data", type = str, dest = "prepped_data_id")
args = parser.parse_args()
prepped_data = args.prepped_data_id


### Load prepped data
file_path = os.path.join(prepped_data, "prepped_data.csv")
hd_df = pd.read_csv(file_path)

cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
for c in cat_cols:
    hd_df[c] = hd_df[c].astype("str")
    hd_df[c] = hd_df[c].astype("category")


### Split DataFrame into train and test sets
X = hd_df.iloc[:, 0:-1]
Y = hd_df.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.70, stratify = hd_df["heart_disease"], random_state = rng)


### Fit Catboost model and score it
cb = CatBoostClassifier(loss_function = "Logloss", learning_rate = 0.10, n_estimators = 200, colsample_bylevel = 1, max_depth = 8, l2_leaf_reg = 0, random_state = 123)
cb.fit(X_train, Y_train, cat_features = [1, 2, 5, 6, 8, 10, 11, 12])
acc = cb.score(X_test, Y_test)

# Save model to local location
dump(cb, "/mnt/batch/tasks/shared/LS_root/mounts/clusters/ml-ci/code/Users/judyxdu/Heart-Disease/Model/catboost.pkl")

# Log metrics and upload model
run.log("Accuracy", acc)
run.upload_file(name = "outputs/catboost.pkl", 
    path_or_stream = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/ml-ci/code/Users/judyxdu/Heart-Disease/Model/catboost.pkl")

# Register model
cb_model = run.register_model(model_name = "cb", model_path = "outputs/catboost.pkl")

### Complete run
run.complete()


