"""
Emotion Mapping Module

Converts dataset-specific emotion labels
into one common label format.
"""

# --------------------------
# RAVDESS Emotion Mapping
# --------------------------

RAVDESS_EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# --------------------------
# CREMA-D Emotion Mapping
# --------------------------

CREMAD_EMOTIONS = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}


def map_ravdess(emotion_code):
    """
    Convert RAVDESS emotion code
    into common emotion label.
    """
    return RAVDESS_EMOTIONS.get(emotion_code)


def map_cremad(emotion_code):
    """
    Convert CREMA-D emotion code
    into common emotion label.
    """
    return CREMAD_EMOTIONS.get(emotion_code)


if __name__ == "__main__":

    print("=" * 40)
    print("Emotion Mapper Test")
    print("=" * 40)

    print("\nRAVDESS")
    print("03 ->", map_ravdess("03"))
    print("06 ->", map_ravdess("06"))

    print("\nCREMA-D")
    print("HAP ->", map_cremad("HAP"))
    print("ANG ->", map_cremad("ANG"))