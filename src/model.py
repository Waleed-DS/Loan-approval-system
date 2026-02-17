import joblib
import os
import pandas as pd

class RiskModel:
    def __init__(self):
        # Dynamically find the models folder
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_dir)
        models_dir = os.path.join(root_dir, "models")

        # Load artifacts
        self.model = joblib.load(os.path.join(models_dir, "loan_model.pkl"))
        self.model_columns = joblib.load(os.path.join(models_dir, "model_columns.pkl"))
        self.threshold = joblib.load(os.path.join(models_dir, "approval_threshold.pkl"))

    def predict(self, input_df):
        # Probability of default (class 1)
        probability = self.model.predict_proba(input_df)[0][1]
        
        # Decision based on the optimized threshold
        decision = "Reject" if probability >= self.threshold else "Approve"
        
        return {
            "default_probability": float(probability),
            "decision": decision
        }
