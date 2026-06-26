"""
Dataset Loader

Loads audio files from:
1. RAVDESS
2. CREMA-D

Returns a unified dataset.
"""

from pathlib import Path

from config import RAVDESS_PATH, CREMAD_PATH
from emotion_mapper import map_ravdess, map_cremad


class DatasetLoader:

    def __init__(self):
        self.samples = []

    # ---------------------------------
    # Load RAVDESS
    # ---------------------------------
    def load_ravdess(self):

        print("\nLoading RAVDESS...")

        total = 0

        for actor in RAVDESS_PATH.iterdir():

            if not actor.is_dir():
                continue

            for audio_file in actor.glob("*.wav"):

                emotion_code = audio_file.stem.split("-")[2]

                emotion = map_ravdess(emotion_code)

                if emotion is None:
                    continue

                self.samples.append({
                    "dataset": "RAVDESS",
                    "path": str(audio_file),
                    "emotion": emotion
                })

                total += 1

        print(f"Loaded {total} RAVDESS samples")

    # ---------------------------------
    # Load CREMA-D
    # ---------------------------------
    def load_cremad(self):

        print("\nLoading CREMA-D...")

        total = 0

        for audio_file in CREMAD_PATH.glob("*.wav"):

            parts = audio_file.stem.split("_")

            emotion_code = parts[2]

            emotion = map_cremad(emotion_code)

            if emotion is None:
                continue

            self.samples.append({
                "dataset": "CREMA-D",
                "path": str(audio_file),
                "emotion": emotion
            })

            total += 1

        print(f"Loaded {total} CREMA-D samples")

    # ---------------------------------
    # Load Everything
    # ---------------------------------
    def load(self):

        self.load_ravdess()
        self.load_cremad()

        return self.samples


if __name__ == "__main__":

    loader = DatasetLoader()

    data = loader.load()

    print("\n" + "=" * 50)
    print("Dataset Loaded Successfully")
    print("=" * 50)

    print(f"\nTotal Samples : {len(data)}")

    print("\nFirst 5 Samples:\n")

    for sample in data[:5]:
        print(sample)