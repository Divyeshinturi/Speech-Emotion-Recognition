import joblib
import numpy as np
import tensorflow as tf

from pathlib import Path
from src.feature_extraction import FeatureExtractor

MODELS = Path("models")


class EmotionPredictor:

    def __init__(self):

        self.model = tf.keras.models.load_model(
            MODELS / "emotion_model.keras"
        )

        self.scaler = joblib.load(
            MODELS / "scaler.pkl"
        )

        self.encoder = joblib.load(
            MODELS / "label_encoder.pkl"
        )

        self.extractor = FeatureExtractor()

    def predict(self, audio_path):

        features = self.extractor.extract(audio_path)

        features = features.reshape(1, -1)

        features = self.scaler.transform(features)

        probabilities = self.model.predict(
            features,
            verbose=0
        )

        predicted_index = np.argmax(probabilities)

        predicted_emotion = self.encoder.inverse_transform(
            [predicted_index]
        )[0]

        confidence = float(np.max(probabilities))

        return {
            "emotion": predicted_emotion,
            "confidence": confidence,
            "probabilities": probabilities[0]
        }


if __name__ == "__main__":

    test_audio = input("Enter audio file path: ").strip()

    predictor = EmotionPredictor()

    result = predictor.predict(test_audio)

    print("\n" + "=" * 60)
    print("EMOTION PREDICTION")
    print("=" * 60)

    print(f"Predicted Emotion : {result['emotion']}")
    print(f"Confidence        : {result['confidence'] * 100:.2f}%")

    print("\nProbability Distribution")
    print("-" * 40)

    for emotion, probability in zip(
        predictor.encoder.classes_,
        result["probabilities"]
    ):
        print(f"{emotion:<12} : {probability * 100:.2f}%")