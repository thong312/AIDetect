from PIL import Image
from models_service.emotion_model import predict_emotion
from models_service.gender_model import predict_gender
from models_service.age_model import predict_age_group

emotion_map = {
    "anger": "Tức giận 😡",
    "disgust": "Ghê tởm 🤢",
    "fear": "Sợ hãi 😨",
    "happiness": "Vui vẻ 😄",
    "sadness": "Buồn 😢",
    "surprise": "Ngạc nhiên 😲",
    "neutral": "Bình thường 😐"
}

def analyze_attributes(img_path):
    try:
        image = Image.open(img_path).convert("RGB")

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

