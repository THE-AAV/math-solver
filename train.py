"""Step 1: train the CNN.   Run:  python train.py"""
import os
import random
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from utils import CLASSES, IMG_SIZE, MODEL_PATH, normalize

DATA_DIR = "extracted_images"   # folder from the Kaggle dataset
MAX_PER_CLASS = 3000            # keeps training fast on a laptop
EPOCHS = 15


def load_dataset():
    rng = random.Random(42)
    X, y = [], []
    for label, name in enumerate(CLASSES):
        folder = os.path.join(DATA_DIR, name)
        if not os.path.isdir(folder):
            sys.exit(f"Folder not found: {folder}  (check DATA_DIR at the top of train.py)")
        files = os.listdir(folder)
        rng.shuffle(files)
        files = files[:MAX_PER_CLASS]
        for f in files:
            img = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # dataset = black ink on white -> make it white ink on black
            _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
            X.append(normalize(binary))
            y.append(label)
        print(f"  {name:>6}: {len(files)} images")
    X = np.array(X)[..., np.newaxis]
    y = np.array(y)
    idx = np.random.default_rng(42).permutation(len(X))
    return X[idx], y[idx]


def build_model():
    return models.Sequential([
        layers.Input((IMG_SIZE, IMG_SIZE, 1)),
        # data augmentation (only active while training)
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.05, 0.05),
        # feature extractor
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        # classifier
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(len(CLASSES), activation="softmax"),
    ])


def save_plots(history):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history.history["accuracy"], label="train")
    ax[0].plot(history.history["val_accuracy"], label="validation")
    ax[0].set_title("Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(history.history["loss"], label="train")
    ax[1].plot(history.history["val_loss"], label="validation")
    ax[1].set_title("Loss"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=120)


if __name__ == "__main__":
    print("Loading dataset...")
    X, y = load_dataset()
    split = int(0.85 * len(X))
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    model = build_model()
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=64,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc * 100:.2f}%")

    # per-symbol accuracy (handy for the presentation)
    preds = model.predict(X_test, verbose=0).argmax(axis=1)
    print("\nPer-symbol accuracy:")
    for i, name in enumerate(CLASSES):
        mask = y_test == i
        if mask.any():
            print(f"  {name:>6}: {(preds[mask] == i).mean() * 100:.1f}%")

    model.save(MODEL_PATH)
    save_plots(history)
    print(f"\nSaved model to {MODEL_PATH} and graphs to training_curves.png")
