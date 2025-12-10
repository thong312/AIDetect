from PIL import Image
from emotion_model import predict_emotion
from gender_model import predict_gender

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

        # Emotion
        emotion_en = predict_emotion(image)
        emotion_vi = emotion_map.get(emotion_en, "Không rõ")

        # Gender
        gender_vi = predict_gender(image)

        return {
            "age": "Không hỗ trợ",
            "gender": gender_vi,
            "emotion": emotion_vi
        }

    except Exception as e:
        print("Emotion/Gender Error:", e)
        print("DEBUG emotion_en:", emotion_en)

        return {"age": "?", "gender": "?", "emotion": "?"}

