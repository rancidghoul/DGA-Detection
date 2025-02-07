from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import dns.resolver
import logging
from math import log2

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow frontend to call backend

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Models
try:
    rf_model = joblib.load("rf_dga_model.pkl")
    lstm_model = load_model("lstm_dga_model.h5")
    tokenizer = Tokenizer(char_level=True)
    logger.info("Models loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise

# Feature Extraction (Same as during training)
def entropy(domain):
    prob = [domain.count(c) / len(domain) for c in set(domain)]
    return -sum(p * log2(p) for p in prob)

def extract_features(domain):
    return [
        len(domain),
        len(set(domain)),
        sum(c.isdigit() for c in domain),
        sum(1 for c in domain if c in "aeiou") / len(domain),
        entropy(domain),
    ]

# DNS Resolution Check
def check_dns_resolution(domain):
    try:
        dns.resolver.resolve(domain, "A")  # Try resolving the domain
        return "Resolvable"
    except dns.resolver.NXDOMAIN:
        return "Unresolvable"
    except Exception as e:
        return f"DNS Error: {str(e)}"

# API Documentation
@app.route("/", methods=["GET"])
def home():
    return """
    <h1>DGA Detection API</h1>
    <p>Send a POST request to /predict with a JSON body containing a "domain" field.</p>
    <p>Example: {"domain": "example.com"}</p>
    """

# Prediction Endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get domain from request
        data = request.json
        domain = data.get("domain")
        
        if not domain:
            return jsonify({"error": "Domain field is required."}), 400
        
        logger.info(f"Received domain: {domain}")

        # Random Forest Prediction
        rf_features = np.array([extract_features(domain)])
        rf_prediction = rf_model.predict(rf_features)[0]
        rf_result = "DGA" if rf_prediction == 1 else "Legitimate"

        # LSTM Prediction (if available)
        if lstm_model:
            tokenizer.fit_on_texts([domain])
            lstm_sequence = pad_sequences(tokenizer.texts_to_sequences([domain]), maxlen=20)
            lstm_prediction = lstm_model.predict(lstm_sequence)[0][0]
            lstm_result = "DGA" if lstm_prediction > 0.5 else "Legitimate"
        else:
            lstm_result = "LSTM model unavailable"

        # DNS Check
        dns_check = check_dns_resolution(domain)

        # Combine predictions: if both models agree, use that result
        if rf_result == lstm_result:
            final_result = rf_result
        else:
            final_result = "Uncertain"  # Or apply any other logic like prioritizing one model

        # Return results
        return jsonify({
            "final_prediction": final_result,
            "rf_prediction": rf_result,
            "lstm_prediction": lstm_result,
            "dns_check": dns_check
        })
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500


# Run the app
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)