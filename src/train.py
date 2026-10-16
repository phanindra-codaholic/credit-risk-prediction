import os
import json
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import (
    clean_data,
    build_preprocessor,
    TARGET_COLUMN,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "credit_risk_dataset.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PIPELINE_PATH = os.path.join(MODELS_DIR, "credit_risk_pipeline.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.json")


def load_dataset() -> pd.DataFrame:
    """Loads dataset from local CSV file or downloads it if not present."""
    if not os.path.exists(DATA_PATH):
        print("Data file not found locally. Downloading...")
        from data.download_data import download_dataset
        download_dataset()
    
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def evaluate_model(name: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Computes comprehensive evaluation metrics for a model pipeline."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"\n==================== {name} ====================")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(f"  [TN={cm[0][0]}, FP={cm[0][1]}]")
    print(f"  [FN={cm[1][0]}, TP={cm[1][1]}]")

    return {
        "model": name,
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "confusion_matrix": cm
    }


def train_and_compare_models():
    """Executes the complete training workflow, model comparison, and serialization."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Load and clean data
    raw_df = load_dataset()
    df = clean_data(raw_df)
    print(f"Cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns.")

    # 2. Separate features & target
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN]

    # 3. Stratified Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples | Testing set: {X_test.shape[0]} samples")

    # 4. Define candidate models
    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    }

    all_metrics = []
    trained_pipelines = {}

    for name, model in candidate_models.items():
        print(f"\nTraining {name} with ColumnTransformer...")
        preprocessor = build_preprocessor()
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])
        
        # Fit on training data
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        # Evaluate on test data
        metrics = evaluate_model(name, pipeline, X_test, y_test)
        all_metrics.append(metrics)

    # 5. Cross-Validation on the best candidate (Random Forest)
    print("\n--- 5-Fold Stratified Cross Validation (Random Forest ROC-AUC) ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
    ])
    cv_scores = cross_val_score(rf_pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"CV ROC-AUC Scores: {[round(s, 4) for s in cv_scores]}")
    print(f"Mean CV ROC-AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    # 6. Select the best model (Random Forest) and save the full pipeline
    best_model_name = "Random Forest"
    best_pipeline = trained_pipelines[best_model_name]

    print(f"\nSaving best pipeline ({best_model_name}) to: {PIPELINE_PATH}")
    joblib.dump(best_pipeline, PIPELINE_PATH)

    # Save summary metrics
    summary = {
        "best_model": best_model_name,
        "features": {
            "numerical": NUMERICAL_FEATURES,
            "categorical": CATEGORICAL_FEATURES
        },
        "target": TARGET_COLUMN,
        "all_model_evaluations": all_metrics,
        "rf_cross_validation_roc_auc_mean": round(float(np.mean(cv_scores)), 4)
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Model metrics and metadata saved to: {METRICS_PATH}")
    print("Training phase completed successfully!")


if __name__ == "__main__":
    train_and_compare_models()
