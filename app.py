import pickle
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# LOAD MODEL ON STARTUP
# ─────────────────────────────────────────────
MODEL_PATH      = "model/model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"
ACCURACY_PATH   = "model/accuracy.txt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found. Run train.py first.")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# Read saved accuracy
model_accuracy = "N/A"
if os.path.exists(ACCURACY_PATH):
    with open(ACCURACY_PATH, "r") as f:
        model_accuracy = f.read().strip() + "%"

print(f"Model loaded. Accuracy: {model_accuracy}")


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", accuracy=model_accuracy)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON: { "text": "news article content here" }
    Returns JSON: { "prediction": "Fake" | "Real", "confidence": 0.95, "label": 0 | 1 }
    """
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"].strip()

    if len(text) < 20:
        return jsonify({"error": "Text too short. Please enter a proper news article."}), 400

    # Preprocess + vectorize
    text_lower = text.lower()
    text_tfidf = vectorizer.transform([text_lower])

    # Predict
    label     = int(model.predict(text_tfidf)[0])
    proba     = model.predict_proba(text_tfidf)[0]
    confidence = float(max(proba))

    result = {
        "prediction": "Real" if label == 1 else "Fake",
        "label":      label,
        "confidence": round(confidence * 100, 2),
        "fake_prob":  round(float(proba[0]) * 100, 2),
        "real_prob":  round(float(proba[1]) * 100, 2),
    }

    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_accuracy": model_accuracy})


# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
