import joblib
import pandas as pd
import tensorflow as tf

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

OUTPUTS = Path("outputs")
MODELS = Path("models")

MODELS.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

df = pd.read_csv(
    OUTPUTS / "emotion_features.csv"
)

print("\nDataset Shape:")
print(df.shape)

X = df.drop(
    columns=[
        "dataset",
        "emotion",
        "augmentation"
    ]
)

y = df["emotion"]

print("\nEncoding Labels...")

encoder = LabelEncoder()
y = encoder.fit_transform(y)

print(encoder.classes_)

print("\nScaling Features...")

scaler = StandardScaler()
X = scaler.fit_transform(X)

print("\nSplitting Dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

joblib.dump(
    scaler,
    MODELS / "scaler.pkl"
)

joblib.dump(
    encoder,
    MODELS / "label_encoder.pkl"
)
print("\nBuilding TensorFlow Model...")

model = Sequential([

    tf.keras.Input(
        shape=(X_train.shape[1],)
    ),

    Dense(
        256,
        activation="relu"
    ),

    BatchNormalization(),

    Dropout(0.30),

    Dense(
        128,
        activation="relu"
    ),

    BatchNormalization(),

    Dropout(0.30),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.20),

    Dense(
        len(encoder.classes_),
        activation="softmax"
    )

])

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

print("\n")
model.summary()

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=8,

    restore_best_weights=True,

    verbose=1

)

reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=4,

    verbose=1,

    min_lr=1e-6

)

checkpoint = ModelCheckpoint(

    filepath=MODELS / "emotion_model.keras",

    monitor="val_accuracy",

    save_best_only=True,

    mode="max",

    verbose=1

)

print("\nStarting Training...\n")

history = model.fit(

    X_train,

    y_train,

    validation_split=0.20,

    epochs=50,

    batch_size=128,

    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ],

    verbose=1

)
print("\nLoading Best Saved Model...\n")

best_model = tf.keras.models.load_model(
    MODELS / "emotion_model.keras"
)

print("Evaluating Model...\n")

test_loss, test_accuracy = best_model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(f"Test Accuracy : {test_accuracy:.4f}")
print(f"Test Loss     : {test_loss:.4f}")

joblib.dump(
    history.history,
    MODELS / "history.pkl"
)

print("\nTraining History Saved")

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")
print(f"Emotion Classes  : {len(encoder.classes_)}")

print("\nEmotion Labels")
print("-" * 30)

for index, emotion in enumerate(encoder.classes_):
    print(f"{index} -> {emotion}")

print("\nSaved Files")
print("-" * 30)
print("✓ emotion_model.keras")
print("✓ scaler.pkl")
print("✓ label_encoder.pkl")
print("✓ history.pkl")

print("\nProject Status")
print("-" * 30)
print("✓ Dataset Loaded")
print("✓ Labels Encoded")
print("✓ Features Scaled")
print("✓ TensorFlow Model Trained")
print("✓ Best Model Saved")
print("✓ Training History Saved")