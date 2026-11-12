import os
from typing import Any, Dict

import joblib
import pandas as pd

from app.config import settings


class CreditRiskPredictor:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or settings.MODEL_PATH
        self.pipeline = None
        self.load_model()

    def load_model(self):
        if not os.path.isabs(self.model_path):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.model_path = os.path.join(project_root, self.model_path)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Trained model not found at {self.model_path}. Run `python src/train.py` first."
            )

        self.pipeline = joblib.load(self.model_path)

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if self.pipeline is None:
            self.load_model()

        df = pd.DataFrame([input_data])
        raw_pred = self.pipeline.predict(df)[0]
        raw_proba = self.pipeline.predict_proba(df)[0][1]

        is_default = bool(raw_pred == 1)
        return {
            "prediction": "High Risk" if is_default else "Low Risk",
            "probability": round(float(raw_proba), 4),
            "default_risk": "Likely Default" if is_default else "Likely Non-Default",
        }


predictor = CreditRiskPredictor()
