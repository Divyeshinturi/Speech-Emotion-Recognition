import pandas as pd

df = pd.read_csv("outputs/emotion_features.csv")

print("=" * 50)
print("Dataset Shape")
print(df.shape)

print("\nEmotion Classes")
print(df["emotion"].unique())

print("\nNumber of Classes")
print(df["emotion"].nunique())