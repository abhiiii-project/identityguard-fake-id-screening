"""Web UI + REST API.

    python app.py            # http://localhost:5000
    POST /api/screen         multipart: fields + optional `document` image
    GET  /api/stats
"""
import os

from flask import Flask, jsonify, render_template, request

from screening import ScreeningPipeline
from screening.store import HistoryStore

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB
store = HistoryStore(os.environ.get("IDSCREEN_DB", "idscreen.db"))
pipeline = ScreeningPipeline(store=store)

FIELDS = ["name", "dob", "aadhaar", "pan", "phone", "email", "pincode", "issue_date", "expiry_date"]


@app.route("/")
def index():
    return render_template("index.html", ml=pipeline.model.available)


@app.post("/api/screen")
def screen():
    declared = {k: (request.form.get(k) or "").strip() for k in FIELDS}
    f = request.files.get("document")
    data = f.read() if f and f.filename else None
    m1, m2 = request.form.get("mrz1", "").strip(), request.form.get("mrz2", "").strip()
    mrz = (m1, m2) if m1 and m2 else None
    record = request.form.get("record", "1") != "0"
    report = pipeline.screen(declared, data, mrz=mrz, record=record)
    return jsonify(report.to_dict())


@app.get("/api/stats")
def stats():
    return jsonify(store.stats())


@app.errorhandler(413)
def too_big(_):
    return jsonify({"error": "File too large (max 10 MB)"}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
