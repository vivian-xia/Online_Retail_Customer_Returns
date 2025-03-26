import json
import numpy as np
import joblib
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path("model_9.pkl")
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)["data"]
        data = np.array(data)
        preds = model.predict(data)
        return preds.tolist()
    except Exception as e:
        return str(e)
