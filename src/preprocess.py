import librosa
import pandas as pd

from src.dataset_loader import DatasetLoader
from src.audio_validator import AudioValidator
from src.augmentation import AudioAugmentor
from src.feature_extraction import FeatureExtractor
from src.config import OUTPUT_DIR

def preprocess():

    print("=" * 70)
    print("STARTING PREPROCESSING PIPELINE")
    print("=" * 70)

    loader = DatasetLoader()
    samples = loader.load()

    validator = AudioValidator()
    samples = validator.validate(samples)

    augmentor = AudioAugmentor()
    extractor = FeatureExtractor()

    dataset = []
    emotion_count = {}

    ravdess_count = 0
    cremad_count = 0

    total_files = len(samples)

    for index, sample in enumerate(samples, start=1):

        if index % 100 == 0:
            print(f"Processed {index}/{total_files} files")

        if sample["dataset"] == "RAVDESS":
            ravdess_count += 1
        else:
            cremad_count += 1

        emotion = sample["emotion"]

        emotion_count[emotion] = (
            emotion_count.get(emotion, 0) + 1
        )

        audio, sr = librosa.load(
            sample["path"],
            sr=None
        )

        augmented_audio = augmentor.augment(
            audio,
            sr
        )

        for aug_name, audio_sample in augmented_audio:

            features = extractor.extract_from_audio(
                audio_sample,
                sr
            )

            row = {
                "dataset": sample["dataset"],
                "emotion": emotion,
                "augmentation": aug_name
            }

            for i in range(40):
                row[f"mfcc_{i+1}"] = features[i]

            row["chroma_mean"] = features[40]
            row["mel_mean"] = features[41]
            row["zcr"] = features[42]
            row["rms"] = features[43]

            dataset.append(row)
            df = pd.DataFrame(dataset)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_DIR / "emotion_features.csv",
        index=False
    )

    print("\nemotion_features.csv saved")

    summary = pd.DataFrame({

        "Dataset": [
            "RAVDESS",
            "CREMA-D"
        ],

        "Validated Files": [
            ravdess_count,
            cremad_count
        ],

        "Generated Samples": [
            ravdess_count * 4,
            cremad_count * 4
        ]

    })

    summary.to_csv(
        OUTPUT_DIR / "dataset_summary.csv",
        index=False
    )

    print("dataset_summary.csv saved")

    emotion_df = pd.DataFrame({

        "Emotion": list(emotion_count.keys()),
        "Count": list(emotion_count.values())

    })

    emotion_df = emotion_df.sort_values(
        by="Emotion"
    ).reset_index(drop=True)

    emotion_df.to_csv(
        OUTPUT_DIR / "emotion_distribution.csv",
        index=False
    )

    print("emotion_distribution.csv saved")
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETED")
    print("=" * 70)

    print(f"Validated Audio Files : {len(samples)}")
    print(f"RAVDESS Files         : {ravdess_count}")
    print(f"CREMA-D Files         : {cremad_count}")
    print(f"Generated Samples     : {len(dataset)}")

    print("\nGenerated Files")
    print("---------------------------")
    print("✓ emotion_features.csv")
    print("✓ dataset_summary.csv")
    print("✓ emotion_distribution.csv")


if __name__ == "__main__":
    preprocess()