"""
Audio Augmentation Module

Returns augmented audio along with
its augmentation name.
"""

import librosa
import numpy as np

from config import (
    ADD_NOISE,
    PITCH_SHIFT,
    TIME_STRETCH
)


class AudioAugmentor:

    def __init__(self):
        pass

    # -------------------------------------------------
    # Noise Injection
    # -------------------------------------------------

    def add_noise(self, audio):

        noise = np.random.randn(len(audio))

        return audio + 0.005 * noise

    # -------------------------------------------------
    # Pitch Shift
    # -------------------------------------------------

    def pitch_shift(self, audio, sr):

        return librosa.effects.pitch_shift(
            y=audio,
            sr=sr,
            n_steps=2
        )

    # -------------------------------------------------
    # Time Stretch
    # -------------------------------------------------

    def time_stretch(self, audio):

        return librosa.effects.time_stretch(
            y=audio,
            rate=1.1
        )

    # -------------------------------------------------
    # Apply Augmentation
    # -------------------------------------------------

    def augment(self, audio, sr):

        augmented = []

        # Original
        augmented.append(("original", audio))

        if ADD_NOISE:
            augmented.append(
                ("noise", self.add_noise(audio))
            )

        if PITCH_SHIFT:
            augmented.append(
                ("pitch_shift", self.pitch_shift(audio, sr))
            )

        if TIME_STRETCH:
            augmented.append(
                ("time_stretch", self.time_stretch(audio))
            )

        return augmented


# -------------------------------------------------
# Test
# -------------------------------------------------

if __name__ == "__main__":

    from dataset_loader import DatasetLoader

    loader = DatasetLoader()

    samples = loader.load()

    audio, sr = librosa.load(
        samples[0]["path"],
        sr=None
    )

    augmentor = AudioAugmentor()

    augmented = augmentor.augment(audio, sr)

    print("=" * 60)
    print("AUGMENTATION TEST")
    print("=" * 60)

    print(f"Generated {len(augmented)} samples\n")

    for name, _ in augmented:
        print(name)