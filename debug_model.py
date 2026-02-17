import sys
import os

print("LOG: 1. Script started")
try:
    import joblib
    import pandas as pd
    import sklearn
    print(f"LOG: 2. Libraries loaded (sklearn version: {sklearn.__version__})")
except Exception as e:
    print(f"LOG: ERROR during library import: {e}")
    sys.exit()

model_path = "models/loan_model.pkl"
print(f"LOG: 3. Checking for model file at {model_path}...")

if os.path.exists(model_path):
    print("LOG: 4. File found. Attempting to load (This is where it usually hangs)...")
    try:
        model = joblib.load(model_path)
        print("LOG: 5. Model loaded successfully!")
    except Exception as e:
        print(f"LOG: ERROR during model loading: {e}")
else:
    print("LOG: ERROR - Model file not found!")