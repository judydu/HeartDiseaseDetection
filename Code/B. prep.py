# prep.py

### Load packages
import os
import numpy as np
import argparse
from azureml.core import Run


### Get the experiment run context
run = Run.get_context()


### Get arguments
parser = argparse.ArgumentParser()
parser.add_argument("--hs_ds", type = str, dest = "hs_ds_id")
parser.add_argument("--out_folder", type = str, dest = "folder")
args = parser.parse_args()
output_folder = args.folder

def transform(df):
    """Function cleans and transform the input DataFrame"""
    # Rename columns
    col_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "heart_disease"]
    df.columns = col_names

    # Change data types
    cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "heart_disease"]
    for c in cat_cols:
        df[c] = df[c].astype("str")
        df[c] = df[c].astype("category")

    # Change to binary variable
    df["heart_disease"] = np.where(df["heart_disease"] == "0", 0, 1)

    # Drop records
    ind_ = df[(df["ca"] == "?") | (df["thal"] == "?")].index
    df.drop(index = ind_, inplace = True)

    df["ca"].cat.remove_categories(["?"])
    df["thal"] = df["thal"].cat.remove_categories(["?"])

    return df


### Load and transform data
dataset = run.input_datasets["heart_disease_data"]
hd_df = dataset.to_pandas_dataframe()
hd_df = transform(df = hd_df)


### Save DataFrame

# Save to local location
# hd_df.to_csv("/mnt/batch/tasks/shared/LS_root/mounts/clusters/ml-ci/code/Users/judyxdu/Heart-Disease/Code/prepped.csv",  index = False, header = True)

# Save to blob store
os.makedirs(output_folder, exist_ok = True)
output_path = os.path.join(output_folder, "prepped_data.csv")
hd_df.to_csv(output_path, index = False, header = True)


