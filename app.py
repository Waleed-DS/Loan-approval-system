import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.model import RiskModel
from src.preprocessing import clean_input_data
from src.features import engineer_features, align_model_columns

# 1. Initialize API
app = FastAPI(
    title="Loan Approval AI",
    description="XGBoost Model for Credit Risk Assessment",
    version="2.0"
)

# 2. Load the Brain (Model) Once at Startup
try:
    brain = RiskModel()
    print("--- [INFO] RiskModel Loaded Successfully ---")
except Exception as e:
    print(f"--- [CRITICAL] Failed to load model: {e} ---")
    brain = None

# 3. Define the Input Format (Strong Typing)
# This automatically validates data before it reaches your model!
class LoanApplication(BaseModel):
    income: float = Field(..., gt=0, description="Annual Income")
    loan_amount: float = Field(..., gt=0, description="Requested Loan Amount")
    credit_score: float = Field(..., ge=300, le=850, description="Credit Score (300-850)")
    employment_length_years: float = Field(..., ge=0, description="Years of Employment")
    home_ownership: str = Field(..., description="RENT, MORTGAGE, OWN, OTHER")
    loan_intent: str = Field(..., description="PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT")
    loan_grade: str = Field(..., description="Loan Grade (A-G)")
    historical_default: str = Field(..., description="Previous Default History (Y/N)")

    class Config:
        json_schema_extra = {
            "example": {
                "income": 55000,
                "loan_amount": 12000,
                "credit_score": 710,
                "employment_length_years": 6,
                "home_ownership": "RENT",
                "loan_intent": "PERSONAL",
                "loan_grade": "B",
                "historical_default": "N"
            }
        }

@app.get("/")
def root():
    return {"message": "Loan Approval AI is active. Go to /docs to test prediction."}

@app.get("/health")
def health_check():
    if brain is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "threshold": brain.threshold}

@app.post("/predict")
def predict_loan(application: LoanApplication):
    if brain is None:
        raise HTTPException(status_code=503, detail="Model is offline")

    try:
        # 1. Convert incoming JSON to DataFrame
        # .dict() converts the Pydantic object to a standard dictionary
        data_dict = application.dict()
        raw_data = pd.DataFrame([data_dict])

        # 2. Run the Engineering Pipeline
        cleaned = clean_input_data(raw_data)
        feats = engineer_features(cleaned)
        
        # 3. Align Columns (The Critical Fix we worked on)
        final_input = align_model_columns(feats, brain.model_columns)

        # 4. Predict
        result = brain.predict(final_input)

        return {
            "status": "success",
            "input_summary": {
                "income": application.income,
                "credit_score": application.credit_score
            },
            "prediction": result
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")

# Run with: python app.py
if __name__ == "__main__":
    # 1. Print a clickable link in the terminal
    print("---  API IS LIVE! ---")
    print("--- Go to: http://127.0.0.1:5000/docs to test predictions ---")