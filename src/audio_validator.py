"""
Audio Validator

Checks whether audio files are valid
before preprocessing.
"""

import librosa


class AudioValidator:

    def __init__(self):
        self.valid = 0
        self.invalid = 0

    def validate(self, samples):

        valid_samples = []

        for sample in samples:

            try:

                audio, sr = librosa.load(
                    sample["path"],
                    sr=None
                )

                if len(audio) == 0:
                    raise Exception("Empty Audio")

                sample["sample_rate"] = sr
                sample["duration"] = len(audio) / sr

                valid_samples.append(sample)

                self.valid += 1

            except Exception as e:

                self.invalid += 1

                print(f"Skipped: {sample['path']}")
                print(e)

        print("\n" + "=" * 50)
        print("Audio Validation Summary")
        print("=" * 50)

        print(f"Valid Files   : {self.valid}")
        print(f"Invalid Files : {self.invalid}")

        return valid_samples


# --------------------------------------------------
# Test the module
# --------------------------------------------------

if __name__ == "__main__":

    from dataset_loader import DatasetLoader

    loader = DatasetLoader()

    samples = loader.load()

    validator = AudioValidator()

    validated = validator.validate(samples)

    print(f"\nValidated Samples : {len(validated)}")