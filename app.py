from flask import Flask, request, jsonify, render_template
from database import Base, engine
from pipeline import register_user, recognize_user
import logging
import os
from pathlib import Path

app = Flask(__name__)
if not os.path.exists("temp"):
    os.makedirs("temp")

Base.metadata.create_all(bind=engine)

# Logging for anomaly/alert events
LOG_PATH = Path("logs/anomalies.log")
LOG_PATH.parent.mkdir(exist_ok=True)
logger = logging.getLogger("aihuman.anomaly")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_PATH)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)


def build_alert_message(name, distance):
    """Create a short Vietnamese warning for low-confidence or failed recognition."""
    if name == "Không thấy mặt":
        return "Không phát hiện khuôn mặt. Ảnh mờ hoặc lệch góc, thử chụp gần và đủ sáng."
    if name == "Người lạ":
        return "Không khớp dữ liệu. Có thể là người mới hoặc ảnh chưa rõ; thử chụp lại hoặc đăng ký mới."

    try:
        dist_val = float(distance)
    except (TypeError, ValueError):
        return None

    if dist_val >= 0.6:
        return f"Độ tin cậy thấp (khoảng cách {dist_val:.2f}). Giữ mặt thẳng, ánh sáng đều và đứng gần camera hơn."
    return None


def log_alert(name, distance, alert):
    """Persist anomaly alerts to file for later review."""
    if not alert:
        return
    try:
        dist_val = float(distance)
    except (TypeError, ValueError):
        dist_val = -1
    logger.warning("Alert: %s | name=%s | distance=%.2f", alert, name, dist_val)

@app.route("/")
def index():
    return render_template("index.html")


# ------------------------ ĐĂNG KÝ ------------------------
@app.route("/register", methods=["POST"])
def register():
    file = request.files["file"]
    name = request.form["name"]

    user, msg = register_user(file, name)
    if user is None:
        return jsonify({"message": msg}), 400

    return jsonify({"message": msg})


# ------------------------ NHẬN DIỆN ------------------------
@app.route("/recognize", methods=["POST"])
def recognize():
    file = request.files["file"]

    name, attributes, dist = recognize_user(file)
    alert = build_alert_message(name, dist)
    log_alert(name, dist, alert)

    return jsonify({
        "name": name,
        "distance": f"{dist:.2f}",
        "info": attributes,
        "alert": alert
    })


# ------------------------ PHÂN TÍCH ------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files["file"]
    from attributes import analyze_attributes
    # Pass FileStorage directly to analyze_attributes (it handles file-like objects)
    info = analyze_attributes(file)

    return jsonify({"info": info})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
