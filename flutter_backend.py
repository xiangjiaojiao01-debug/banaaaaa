from pathlib import Path
import tempfile
import traceback

from flask import Flask, jsonify, request
from flask_cors import CORS

from backend import DEFAULT_CONF, predict_and_analyze


app = Flask(__name__)
CORS(app)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


def _request_conf():
    raw_conf = request.form.get("conf") or request.args.get("conf")
    if raw_conf in (None, ""):
        return DEFAULT_CONF

    try:
        conf = float(raw_conf)
    except ValueError:
        raise ValueError("conf must be a number")

    if not 0.01 <= conf <= 0.99:
        raise ValueError("conf must be between 0.01 and 0.99")

    return conf


def _build_response(detections, analysis):
    black_spot_pct = float(analysis["black_spot_pct"])
    spot_count = int(analysis["spot_count"])

    return {
        "ok": True,
        **analysis,
        "black_spot_pct": black_spot_pct,
        "spot_count": spot_count,
        "detections": detections,
        # Backward-compatible aliases for older Flutter code.
        "spot_ratio": black_spot_pct,
        "black_spot_ratio": black_spot_pct,
        "brix_estimate": round(black_spot_pct * 0.5 + 14, 1),
    }


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "missing image"}), 400

    uploaded_file = request.files["image"]
    if uploaded_file.filename == "":
        return jsonify({"error": "empty filename"}), 400

    suffix = Path(uploaded_file.filename or "").suffix or ".jpg"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        uploaded_file.save(tmp.name)
        image_path = Path(tmp.name)

    try:
        detections, analysis = predict_and_analyze(image_path, conf=_request_conf())
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": "prediction failed", "detail": str(exc)}), 500
    finally:
        image_path.unlink(missing_ok=True)

    return jsonify(_build_response(detections, analysis))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
