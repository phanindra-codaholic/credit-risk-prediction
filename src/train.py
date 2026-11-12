import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    build_preprocessor,
    clean_data,
)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "credit_risk_dataset.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PIPELINE_PATH = os.path.join(MODELS_DIR, "credit_risk_pipeline.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")


def load_dataset() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        from data.download_data import download_dataset
        download_dataset()

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def evaluate_model(name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

    return {
        "model": name,
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "confusion_matrix": cm,
    }


def train_and_compare_models():
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = clean_data(load_dataset())
    print(f"Cleaned dataset: {df.shape[0]} rows.")

    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    }

    all_metrics = []
    trained_pipelines = {}

    for name, model in candidate_models.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", model),
        ])
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline
        all_metrics.append(evaluate_model(name, pipeline, X_test, y_test))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
    ])
    cv_scores = cross_val_score(rf_pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"\nRandom Forest 5-Fold CV ROC-AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    best_model_name = "Random Forest"
    joblib.dump(trained_pipelines[best_model_name], PIPELINE_PATH)

    summary = {
        "best_model": best_model_name,
        "features": {
            "numerical": NUMERICAL_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
        "target": TARGET_COLUMN,
        "all_model_evaluations": all_metrics,
        "rf_cross_validation_roc_auc_mean": round(float(np.mean(cv_scores)), 4),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved pipeline to {PIPELINE_PATH}")


if __name__ == "__main__":
    train_and_compare_models()
