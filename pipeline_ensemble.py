"""
Updated pipeline with ensemble model support
This file shows how to integrate ensemble models into your existing pipeline
"""

import pickle
from PIL import Image
from models import User
from database import SessionLocal
from face_recognition import get_embedding
from attributes import analyze_attributes

# NEW: Import ensemble-based emotion prediction
from models_service.emotion_model import predict_emotion_ensemble
from io import BytesIO


def _load_image(image_or_path):
    """Load an image from a path, file-like object, Flask FileStorage, or bytes."""
    if isinstance(image_or_path, str):
        return Image.open(image_or_path).convert("RGB")
    if isinstance(image_or_path, (bytes, bytearray)):
        return Image.open(BytesIO(image_or_path)).convert("RGB")
    try:
        return Image.open(image_or_path).convert("RGB")
    except Exception:
        stream = getattr(image_or_path, "stream", None)
        if stream is not None:
            stream.seek(0)
            return Image.open(stream).convert("RGB")
        raise


def register_user(image_file, name):
    """Pipeline ĐĂNG KÝ người dùng."""
    image = _load_image(image_file)
    embedding = get_embedding(image)

    if embedding is None:
        return None, "Không tìm thấy khuôn mặt."

    db = SessionLocal()
    user = User(name=name, encoding=pickle.dumps(embedding))
    db.add(user)
    db.commit()
    db.close()

    return user, "Đăng ký thành công."


def recognize_user(image_file, use_ensemble=True):
    """
    Pipeline nhận diện + phân tích cảm xúc
    
    Args:
        image_file: uploaded image file
        use_ensemble: if True, use ensemble emotion model; if False, use ViT only
    
    Returns:
        user_name, attributes, distance
    """
    image = _load_image(image_file)
    target_embedding = get_embedding(image)

    if target_embedding is None:
        return "Không thấy mặt", {}, 100

    # --- Phân tích cảm xúc (with ensemble) ---
    attributes = analyze_attributes(image)

    # --- So sánh embedding ---
    db = SessionLocal()
    users = db.query(User).all()
    db.close()

    min_dist = 100
    result_name = "Người lạ"

    for u in users:
        db_embedding = pickle.loads(u.encoding)
        dist = (target_embedding - db_embedding).norm().item()
        if dist < 0.8 and dist < min_dist:
            min_dist = dist
            result_name = u.name

    return result_name, attributes, min_dist


def recognize_user_with_details(image_file, use_ensemble=True):
    """
    Enhanced recognize function with detailed emotion scores
    
    Returns:
        user_name, attributes_with_details, distance
    """
    image = _load_image(image_file)
    target_embedding = get_embedding(image)

    if target_embedding is None:
        return "Không thấy mặt", {}, 100

    # --- Phân tích cảm xúc với chi tiết ---
    if use_ensemble:
        # Get detailed scores from ensemble
        from models_service.emotion_model import predict_emotion_ensemble
        emotion_en, emotion_conf, details = predict_emotion_ensemble(image, return_details=True)
        
        attributes = {
            "emotion": emotion_en,
            "emotion_confidence": round(emotion_conf, 2),
            "emotion_details": {
                "vit_confidence": round(details['vit_confidence'], 2),
                "transunet_confidence": round(details['transunet_confidence'], 2)
            }
        }
    else:
        # Use standard analyze_attributes
        attributes = analyze_attributes(image)

    # --- So sánh embedding ---
    db = SessionLocal()
    users = db.query(User).all()
    db.close()

    min_dist = 100
    result_name = "Người lạ"

    for u in users:
        db_embedding = pickle.loads(u.encoding)
        dist = (target_embedding - db_embedding).norm().item()
        if dist < 0.8 and dist < min_dist:
            min_dist = dist
            result_name = u.name

    return result_name, attributes, min_dist


# Example of how to use in Flask app:
"""
# In your app.py:

@app.route("/recognize", methods=["POST"])
def recognize():
    file = request.files["file"]
    use_ensemble = request.args.get("ensemble", "true").lower() == "true"
    
    name, attributes, dist = recognize_user(file, use_ensemble=use_ensemble)

    return jsonify({
        "name": name,
        "distance": f"{dist:.2f}",
        "info": attributes
    })

@app.route("/recognize-detailed", methods=["POST"])
def recognize_detailed():
    '''Endpoint that returns detailed emotion scores from ensemble'''
    file = request.files["file"]
    
    name, attributes, dist = recognize_user_with_details(file, use_ensemble=True)

    return jsonify({
        "name": name,
        "distance": f"{dist:.2f}",
        "info": attributes
    })
"""
