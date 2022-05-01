# score.py

### Load packages
import json
import joblib
import numpy as np
import os


### Load model
def init():
    """Function that loads CatBoost model"""
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "catboost.pkl")
    model = joblib.load(model_path)


### Score model
def run(raw_data):
    """Function that applies model scoring"""
    data = np.array(json.loads(raw_data)["data"])
    predictions = model.predict(data)
    return predictions.tolist()


