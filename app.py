from flask import Flask, request, jsonify, render_template
from database import Base, engine
from pipeline import register_user, recognize_user
import os

app = Flask(__name__)
if not os.path.exists("temp"):
    os.makedirs("temp")

Base.metadata.create_all(bind=engine)

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

    return jsonify({
        "name": name,
        "distance": f"{dist:.2f}",
        "info": attributes
    })


# ------------------------ PHÂN TÍCH ------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files["file"]
    temp_path = "temp/analyze.jpg"
    file.save(temp_path)

    from attributes import analyze_attributes
    info = analyze_attributes(temp_path)

    return jsonify({"info": info})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
