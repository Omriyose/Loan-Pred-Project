from flask import Flask, jsonify, request
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
    return jsonify({"ready": model_ready()})

@app.get("/api/info")
def get_info():
    if model_ready():
        return jsonify(get_model_info())
    else:
        return jsonify({"error": "Model not found, please train a model first"}), 503

@app.get("/api/metrics")
def metrics_print():
    if model_ready():
        return jsonify(get_metrics())
    else:
        return jsonify({"error": "Model not found, please train a model first"}), 503

@app.get("/api/features")
def features_print():
    if model_ready():
        return jsonify(get_features())
    else:
        return jsonify({"error": "Model not found, please train a model first"}), 503

@app.get("/api/samples")
def samples_print():
    if model_ready():
        return jsonify(get_samples())
    else:
        return jsonify({"error": "Model not found, please train a model first"}), 503

@app.post("/api/predict")
def prediction_print():
    if model_ready():
        applicant = request.get_json(silent=True)
        if applicant is None:
            return jsonify({"error": "Bad input can't be predicted on"}), 400
        features = get_features()
        required = features["Number Features"] + features["Text Features"]
        missing = [name for name in required if name not in applicant]
        if missing:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400
        try:
            return jsonify({
                "prediction": predict_loan(applicant),
                "margin": get_margin(applicant)
                })
        except ValueError:
            return jsonify({"error": "Wrong value, make sure the input data is in the right format"}), 400
    else:
        return jsonify({"error": "Model not found, please train a model first"}), 503

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0", port=5000)

