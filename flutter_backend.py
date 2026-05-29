from pathlib import Path
import tempfile

from flask import Flask, jsonify, request
from flask_cors import CORS

from backend import DEFAULT_CONF, predict_and_analyze


app = Flask(__name__)
CORS(app)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "missing image"}), 400

    uploaded_file = request.files["image"]
    suffix = Path(uploaded_file.filename or "").suffix or ".jpg"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        uploaded_file.save(tmp.name)
        image_path = Path(tmp.name)

    try:
        detections, analysis = predict_and_analyze(image_path, conf=DEFAULT_CONF)
    finally:
        image_path.unlink(missing_ok=True)

    return jsonify({
        **analysis,
        "detections": detections,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
