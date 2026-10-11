from flask import Flask, jsonify, request
import logging
from model_service import (
    get_model_info, get_metrics, get_features, get_samples,
    predict_loan, get_margin, model_ready,
)

app = Flask(__name__)

@app.get("/")
def index():
    """Serve the dashboard page from static/index.html"""
    return app.send_static_file("index.html")

@app.get("/api/check")
def ready_check():
    """Report whether the model and info files loaded"""
    return jsonify({"ready": model_ready()})

@app.get("/api/info")
def get_info():
    """Return the model info"""
    if model_ready():
        return jsonify(get_model_info())
    else:
        logging.warning("Could not find model")
        return jsonify({"error": "Model not found, please train a model first"}), 503

@app.get("/api/metrics")
def metrics_print():
    """Return the metrics of the model"""
    if model_ready():
        return jsonify(get_metrics())
    else:
        logging.warning("Could not find model")
        return jsonify({"error": "Model not found, please train a model first"}), 503

@app.get("/api/features")
def features_print():
    """Return the features of the model"""
    if model_ready():
        return jsonify(get_features())
    else:
        logging.warning("Could not find model")
        return jsonify({"error": "Model not found, please train a model first"}), 503

@app.get("/api/samples")
def samples_print():
    """Return the saved samples"""
    if model_ready():
        return jsonify(get_samples())
    else:
        logging.warning("Could not find model")
        return jsonify({"error": "Model not found, please train a model first"}), 503

@app.post("/api/predict")
def prediction_print():
    """Return the loan prediction for an applicant and their distance from the decision boundary"""
    if model_ready():
        applicant = request.get_json(silent=True)
        if applicant is None:
            logging.warning("Bad input")
            return jsonify({"error": "Bad input can't be predicted on"}), 400
        features = get_features()
        required = features["Number Features"] + features["Text Features"]
        missing = [name for name in required if name not in applicant]
        if missing:
            logging.warning("Predict rejected, missing fields: %s", missing)
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400
        try:
            loan_pred = predict_loan(applicant)
            ap_margin =  get_margin(applicant)
            logging.info("Prediction: %s, margin: %.2f", loan_pred, ap_margin)
            return jsonify({
                "prediction": loan_pred,
                "margin": ap_margin
                })
        except ValueError as err:
            logging.warning("Predict rejected, wrong value: %s", err)
            return jsonify({"error": "Wrong value, make sure the input data is in the right format"}), 400
    else:
        logging.warning("Could not find model")
        return jsonify({"error": "Model not found, please train a model first"}), 503

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0", port=5000)

