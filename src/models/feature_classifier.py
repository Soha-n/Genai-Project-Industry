"""
Feature-based baseline classifier using time-domain features from the CWRU dataset.
Uses Random Forest and/or XGBoost for comparison with the CNN model.
"""

import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run(config_path="configs/config.yaml"):
    cfg = load_config(config_path)
    csv_path = cfg["paths"]["feature_csv"]
    mcfg = cfg["model"]

    print(f"Loading features from {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  Shape: {df.shape}")
    print(f"  Fault classes: {df['fault'].nunique()} — {df['fault'].unique().tolist()}")

    # Features and labels
    feature_cols = [c for c in df.columns if c != "fault"]
    X = df[feature_cols].values
    y = df["fault"].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=mcfg["test_split"],
        random_state=42,
        stratify=y_encoded,
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # Random Forest
    print("\n--- Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"Accuracy: {rf_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, rf_preds, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, rf_preds))

    # Feature importance
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nFeature Importance (top 5):")
    for i in sorted_idx[:5]:
        print(f"  {feature_cols[i]}: {importances[i]:.4f}")


if __name__ == "__main__":
    run()
