import pickle
from PIL import Image
from models import User
from database import SessionLocal
from face_recognition import get_embedding
from attributes import analyze_attributes
from io import BytesIO


def _load_image(image_or_path):
    """Load an image from a path, file-like object, Flask FileStorage, or bytes.

    Returns a PIL Image in RGB mode.
    """
    # path string
    if isinstance(image_or_path, str):
        return Image.open(image_or_path).convert("RGB")

    # bytes
    if isinstance(image_or_path, (bytes, bytearray)):
        return Image.open(BytesIO(image_or_path)).convert("RGB")

    # file-like / FileStorage
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


def recognize_user(image_file):
    """Pipeline nhận diện + phân tích cảm xúc."""
    # Load image directly (supports path, FileStorage, bytes, or PIL Image)
    image = _load_image(image_file)
    target_embedding = get_embedding(image)

    if target_embedding is None:
        return "Không thấy mặt", {}, 100

    # --- Phân tích cảm xúc ---
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
