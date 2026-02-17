import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# --- PRODUCTION CONSTANTS ---
# These are the EXACT values from your EDA notebook.
# This guarantees the API treats outliers exactly like the model expects.
INCOME_CAP = 97146.91475     # 95th Percentile
LOAN_CAP = 25000.0           # 99th Percentile
MEDIAN_INCOME = 36271.225    # Median Income
MEDIAN_LOAN = 15000.0        # Median Loan
MEDIAN_CREDIT = 652.0        # Median Credit Score

def clean_input_data(data) -> pd.DataFrame:
    """
    Applies the EXACT cleaning logic from '01_eda_and_cleaning.ipynb'.
    Ensures Zero Training-Serving Skew.
    """
    # 1. Standardize Input to DataFrame
    # (Handles both API dictionary and Notebook DataFrame)
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Input must be a Dictionary or DataFrame")

    # 2. Fix Negatives (Matches notebook .abs())
    cols_to_abs = ['income', 'loan_amount']
    for col in cols_to_abs:
        if col in df.columns:
            if (df[col] < 0).any():
                logger.warning(f"Negative values detected in {col}. Converting to absolute.")
                df[col] = df[col].abs()

    # 3. Handle Missing Values (Matches notebook .fillna(median))
    if 'income' in df.columns:
        df['income'] = df['income'].fillna(MEDIAN_INCOME)
    
    if 'loan_amount' in df.columns:
        df['loan_amount'] = df['loan_amount'].fillna(MEDIAN_LOAN)
        
    if 'credit_score' in df.columns:
        df['credit_score'] = df['credit_score'].fillna(MEDIAN_CREDIT)

    # 4. Handle Employment Length (Matches notebook logic)
    if 'employment_length_years' in df.columns:
        # Fill NaN with 0
        df['employment_length_years'] = df['employment_length_years'].fillna(0)
        # Clip max value to 50 years (Matches notebook clip)
        df['employment_length_years'] = df['employment_length_years'].clip(0, 50)

    # 5. Outlier Capping (Winsorization)
    # Matches notebook: np.where(income > cap, cap, income)
    if 'income' in df.columns:
        # Using numpy.where ensures it works efficiently on full columns
        df['income'] = np.where(df['income'] > INCOME_CAP, INCOME_CAP, df['income'])

    if 'loan_amount' in df.columns:
        df['loan_amount'] = np.where(df['loan_amount'] > LOAN_CAP, LOAN_CAP, df['loan_amount'])

    # 6. Credit Score Clipping (Matches notebook clip(300, 850))
    if 'credit_score' in df.columns:
        df['credit_score'] = df['credit_score'].clip(300, 850)

    return df