import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "loan_model.joblib"
INFO_PATH = BASE_DIR / "model_info.json"

try:
    pipeline = joblib.load(MODEL_PATH)
    logging.info("Model Loaded")
except FileNotFoundError:
    logging.warning("Load Failed: File not found")
    pipeline = None

def get_model_info():
    """Return the model saved details as a dict, or None if the file is missing or invalid"""
    try:
        with open(INFO_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as err:
        logging.warning("Could not load model info: %s", err)
        return None
    
info = get_model_info()

def model_ready():
    """Return True if both the model file and the info file loaded correctly"""
    return pipeline is not None and info is not None

def predict_loan(applicant):
    """Return the applicant liability for a loan using .predict"""
    temp_df = pd.DataFrame([applicant])
    temp_df = temp_df.astype(object).where(temp_df.notna(), np.nan)
    prediction = pipeline.predict(temp_df)
    return prediction[0]

def get_metrics():
    """Return metrics of the model with the saved info"""
    return {
        "Accuracy": info["accuracy"],
        "Precision": info["precision"],
        "Recall": info["recall"],
        "F1 Score": info["f1 score"], 
        "Confusion Matrix":info["confusion matrix"]
    }

def get_features():
    """Return features of the model using the saved info"""
    return {
        "Number Features" : info["number features"],
        "Text Features" : info["text features"]
    }

def get_samples():
    """Return the samples we saved in the saved info"""
    return {"Model Samples" : info['model samples']}

def get_margin(applicant):
    """
    Return the position of the applicant relative to the boundary,
    Positive side > 0, Negative < 0, +-1 is on the margin edge.
    """
    temp_df = pd.DataFrame([applicant])
    temp_df = temp_df.astype(object).where(temp_df.notna(), np.nan)
    ap_margin = pipeline.decision_function(temp_df)
    return float(ap_margin[0])

if __name__ == "__main__":
    if model_ready():
        row = dict(info["model samples"][0])
        row.pop("Loan_Status")
        print(predict_loan(row))
        print(get_margin(row))
        print(get_metrics())
        print(get_features())
        print(get_samples())
    else:
        print("Model is unready.")