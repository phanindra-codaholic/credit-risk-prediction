import os
import joblib
import pandas as pd
from typing import Dict, Any
from app.config import settings

class CreditRiskPredictor:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or settings.MODEL_PATH
        self.pipeline = None
        self.load_model()

    def load_model(self):
        """Loads the serialized Scikit-learn pipeline from disk."""
        if not os.path.isabs(self.model_path):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.model_path = os.path.join(project_root, self.model_path)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Trained model not found at {self.model_path}. "
                "Please run `python src/train.py` before starting the server."
            )

        print(f"Loading ML pipeline from: {self.model_path}")
        self.pipeline = joblib.load(self.model_path)
        print("ML pipeline loaded successfully.")

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Receives raw applicant features as a dictionary, transforms them into a DataFrame,
        and generates risk prediction and default probability.
        """
        if self.pipeline is None:
            self.load_model()

        # Build single-row DataFrame
        df = pd.DataFrame([input_data])

        # Generate prediction and probability of default (Class 1)
        raw_pred = self.pipeline.predict(df)[0]
        raw_proba = self.pipeline.predict_proba(df)[0][1]

        is_default = bool(raw_pred == 1)
        prediction_label = "High Risk" if is_default else "Low Risk"
        risk_description = "Likely Default" if is_default else "Likely Non-Default"

        return {
            "prediction": prediction_label,
            "probability": round(float(raw_proba), 4),
            "default_risk": risk_description
        }

# Global predictor instance
predictor = CreditRiskPredictor()
