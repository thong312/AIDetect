import pickle
from PIL import Image
from models import User
from database import SessionLocal
from face_recognition import get_embedding
from attributes import analyze_attributes

def register_user(image_file, name):
    """Pipeline ĐĂNG KÝ người dùng."""
    image = Image.open(image_file).convert("RGB")
    embedding = get_embedding(image)

    if embedding is None:
        return None, "Không tìm thấy khuôn mặt."

    db = SessionLocal()
    user = User(name=name, encoding=pickle.dumps(embedding))
    db.add(user)
    db.commit()
    db.close()

    return user, "Đăng ký thành công."


def recognize_user(image_file):
    """Pipeline nhận diện + phân tích cảm xúc."""
    temp_path = "temp/query.jpg"
    image_file.save(temp_path)

    image = Image.open(temp_path).convert("RGB")
    target_embedding = get_embedding(image)

    if target_embedding is None:
        return "Không thấy mặt", {}, 100

    # --- Phân tích cảm xúc ---
    attributes = analyze_attributes(temp_path)

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
