from flask import Flask, request, jsonify, render_template
from database import Base, engine
from pipeline_ensemble import register_user,  recognize_user_with_details
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

    name, attributes, dist =  recognize_user_with_details(file)

    return jsonify({
        "name": name,
        "distance": f"{dist:.2f}",
        "info": attributes
    })


# ------------------------ PHÂN TÍCH ------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files["file"]
    from attributes import analyze_attributes
    # Pass the FileStorage directly; analyze_attributes handles file-like objects
    info = analyze_attributes(file)

    return jsonify({"info": info})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
