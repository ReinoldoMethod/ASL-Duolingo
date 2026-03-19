"""
Train a RandomForest classifier on collected ASL hand landmark data.

Reads normalized landmark features from data/landmarks.csv, trains an
80/20 stratified split, and saves the resulting model to
models/asl_classifier.pkl.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import joblib

DATA_PATH = os.path.join("data", "landmarks.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "asl_classifier.pkl")

FEATURE_COLUMNS = [f"f{i}" for i in range(63)]


def main():
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Collect landmark data first.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} samples from {DATA_PATH}")

    # ------------------------------------------------------------------
    # 2. Inspect class distribution and warn if too few samples
    # ------------------------------------------------------------------
    class_counts = df["label"].value_counts()
    print("\nSamples per class:")
    for label, count in class_counts.sort_index().items():
        print(f"  {label}: {count}")

    if (class_counts < 10).any():
        print(
            "\nWARNING: Some classes have fewer than 10 samples. "
            "Consider collecting more data for reliable training."
        )

    # ------------------------------------------------------------------
    # 3. Prepare features and labels
    # ------------------------------------------------------------------
    X = df[FEATURE_COLUMNS].values
    y = df["label"].values

    # ------------------------------------------------------------------
    # 4. Train/test split (80/20, stratified)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    # ------------------------------------------------------------------
    # 5. Train RandomForest classifier
    # ------------------------------------------------------------------
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 6. Evaluate
    # ------------------------------------------------------------------
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{'=' * 40}")
    print(f"  TEST ACCURACY: {acc:.4f}")
    print(f"{'=' * 40}\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # ------------------------------------------------------------------
    # 7. Save model
    # ------------------------------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
