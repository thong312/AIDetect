from PIL import Image
from models_service.emotion_model import predict_emotion
from models_service.gender_model import predict_gender
from models_service.age_model import predict_age_group
from io import BytesIO

emotion_map = {
    "anger": "Tức giận 😡",
    "disgust": "Ghê tởm 🤢",
    "fear": "Sợ hãi 😨",
    "happiness": "Vui vẻ 😄",
    "sadness": "Buồn 😢",
    "surprise": "Ngạc nhiên 😲",
    "neutral": "Bình thường 😐"
}

def _load_image(image_or_path):
    """Load an image from a path, file-like object, Flask FileStorage, or bytes."""
    # Path string
    if isinstance(image_or_path, str):
        return Image.open(image_or_path).convert("RGB")

    # Bytes
    if isinstance(image_or_path, (bytes, bytearray)):
        return Image.open(BytesIO(image_or_path)).convert("RGB")

    # File-like / FileStorage: PIL can open file-like objects directly
    try:
        return Image.open(image_or_path).convert("RGB")
    except Exception:
        # As a last resort, try reading .stream if present
        stream = getattr(image_or_path, "stream", None)
        if stream is not None:
            stream.seek(0)
            return Image.open(stream).convert("RGB")
        raise


def analyze_attributes(image_or_path):
    """Analyze attributes from a PIL Image or a path/file-like object.

    Returns a dict with age/gender/emotion and confidences. This is robust
    to being passed a path string ("temp/query.jpg"), a Flask FileStorage,
    raw bytes, or a PIL Image.
    """
    try:
        if isinstance(image_or_path, Image.Image):
            image = image_or_path
        else:
            image = _load_image(image_or_path)

        # Emotion with confidence
        emotion_en, emotion_conf = predict_emotion(image)
        emotion_vi = emotion_map.get(emotion_en, "Không rõ")

        # Gender with confidence
        gender_vi, gender_conf = predict_gender(image)

        # Age with confidence
        age_vi, age_conf = predict_age_group(image)

        return {
            "age": age_vi,
            "age_confidence": round(age_conf, 2),
            "gender": gender_vi,
            "gender_confidence": round(gender_conf, 2),
            "emotion": emotion_vi,
            "emotion_confidence": round(emotion_conf, 2)
        }

    except Exception as e:
        print("Error analyzing attributes:", e)
        return {
            "age": "?",
            "age_confidence": 0,
            "gender": "?",
            "gender_confidence": 0,
            "emotion": "?",
            "emotion_confidence": 0
        }

