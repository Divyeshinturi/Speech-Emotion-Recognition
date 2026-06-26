"""
Feature Extraction Module

Extracts audio features from a single audio file or audio array.
"""

import librosa
import numpy as np

from src.config import  SAMPLE_RATE, N_MFCC


class FeatureExtractor:

    def __init__(self):
        self.feature_count = N_MFCC + 4

    # ---------------------------------------
    # Extract features from an audio file
    # ---------------------------------------
    def extract(self, file_path):

        audio, sr = librosa.load(
            file_path,
            sr=SAMPLE_RATE
        )

        return self.extract_from_audio(audio, sr)

    # ---------------------------------------
    # Extract features from an audio array
    # ---------------------------------------
    def extract_from_audio(self, audio, sr):

        # MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=N_MFCC
        )
        mfcc = np.mean(mfcc.T, axis=0)

        # Chroma
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=sr
        )
        chroma = np.mean(chroma)

        # Mel Spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr
        )
        mel = np.mean(mel)

        # Zero Crossing Rate
        zcr = np.mean(
            librosa.feature.zero_crossing_rate(audio)
        )

        # RMS Energy
        rms = np.mean(
            librosa.feature.rms(y=audio)
        )

        features = np.concatenate([
            mfcc,
            [chroma],
            [mel],
            [zcr],
            [rms]
        ])

        return features


# -------------------------------------------------
# Test
# -------------------------------------------------

if __name__ == "__main__":

    from dataset_loader import DatasetLoader

    print("=" * 60)
    print("FEATURE EXTRACTION TEST")
    print("=" * 60)

    loader = DatasetLoader()

    samples = loader.load()

    extractor = FeatureExtractor()

    features = extractor.extract(samples[0]["path"])

    print(f"\nFeature Count : {len(features)}")

    print("\nFirst 10 Features:\n")

    print(features[:10])