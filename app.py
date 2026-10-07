"""
app.py — Flask application for Student Performance Predictor.

Routes
------
GET  /                  → serves index.html
POST /api/predict       → JSON prediction endpoint
GET  /api/metadata      → model names + metrics (for UI info panel)
GET  /api/plots/<name>  → serves PNG plots from models/ and models/plots/
GET  /api/health        → health check

Run:
    python app.py
"""

import os
import json
from flask import Flask, request, jsonify, render_template, send_file, abort

from src.predict import predict, validate_inputs, get_metadata

# --------------------------------------------------------------------------- #
# App setup
# --------------------------------------------------------------------------- #
_ROOT = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(_ROOT, "templates"),
    static_folder=os.path.join(_ROOT, "static"),
)
app.config["JSON_SORT_KEYS"] = False


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html")


@app.route("/api/health")
def health():
    """Simple health check."""
    return jsonify({"status": "ok"})


@app.route("/api/metadata")
def metadata():
    """
    Return model names, metrics for all models, and feature list.
    Used by the UI to populate the 'Model Info' panel.
    """
    try:
        meta = get_metadata()
        return jsonify({"success": True, "data": meta})
    except FileNotFoundError as e:
        return jsonify({"success": False, "error": str(e)}), 503


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    POST body (JSON):
    {
        "studytime":      1-4,
        "attendance_pct": 0-100,
        "G1":             0-20,
        "failures":       0-4,
        "absences":       0-93
    }

    Response (JSON):
    {
        "success": true,
        "data": { ...prediction fields... }
    }
    or on error:
    {
        "success": false,
        "errors": ["..."]
    }
    """
    # Accept JSON body or form data
    if request.is_json:
        raw = request.get_json(silent=True) or {}
    else:
        raw = request.form.to_dict()

    if not raw:
        return jsonify({
            "success": False,
            "errors": ["No input data received. Send a JSON body."]
        }), 400

    # Validate
    clean, errors = validate_inputs(raw)
    if errors:
        return jsonify({"success": False, "errors": errors}), 422

    # Predict
    try:
        result = predict(clean)
        return jsonify({"success": True, "data": result})
    except FileNotFoundError as e:
        return jsonify({
            "success": False,
            "errors": [str(e)]
        }), 503
    except Exception as e:
        app.logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "errors": ["An unexpected error occurred. Please try again."]
        }), 500


@app.route("/api/plots/<path:filename>")
def serve_plot(filename):
    """
    Serve PNG plots stored in models/ or models/plots/.
    E.g. /api/plots/confusion_matrix.png
         /api/plots/plots/01_target_distribution.png
    """
    # Security: only allow .png files, no path traversal
    if not filename.endswith(".png"):
        abort(400, "Only .png files are served.")
    if ".." in filename or filename.startswith("/"):
        abort(400, "Invalid path.")

    models_dir = os.path.join(_ROOT, "models")
    full_path  = os.path.join(models_dir, filename)

    if not os.path.exists(full_path):
        abort(404, f"Plot not found: {filename}")

    return send_file(full_path, mimetype="image/png")


# --------------------------------------------------------------------------- #
# Error handlers
# --------------------------------------------------------------------------- #
@app.errorhandler(404)
def not_found(e):
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "errors": [str(e)]}), 404
    return render_template("index.html"), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"success": False, "errors": ["Method not allowed."]}), 405


@app.errorhandler(500)
def server_error(e):
    return jsonify({"success": False, "errors": ["Internal server error."]}), 500


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import os as _os
    port  = int(_os.environ.get("PORT", 5000))
    debug = _os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    print("=" * 50)
    print("  Student Performance Predictor")
    print(f"  http://127.0.0.1:{port}")
    print("=" * 50)
    app.run(debug=debug, host="0.0.0.0", port=port)
