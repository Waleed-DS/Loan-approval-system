import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Ratios (Safely handle inputs as numeric)
    df['income'] = pd.to_numeric(df['income'], errors='coerce')
    df['loan_amount'] = pd.to_numeric(df['loan_amount'], errors='coerce')
    
    df['loan_to_income'] = df['loan_amount'] / df['income']
    
    # 2. Stress Flags
    df['high_loan_stress'] = (df['loan_to_income'] > 0.5).astype(int)
    
    # 3. Stress Index
    df['credit_score'] = pd.to_numeric(df['credit_score'], errors='coerce')
    df['stress_index'] = df['loan_to_income'] / (df['credit_score'] / 850.0)

    # 4. Credit Tier Binning
    bins = [300, 580, 670, 740, 800, 850]
    labels = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
    df['credit_tier'] = pd.cut(df['credit_score'], bins=bins, labels=labels, include_lowest=True)
    
    # 5. Employment Category
    df['employment_length_years'] = pd.to_numeric(df['employment_length_years'], errors='coerce')
    
    def cat_emp(years):
        if pd.isna(years): return 'New'
        if years < 1: return 'New'
        elif 1 <= years <= 5: return 'Stable'
        else: return 'Long-term'

    df['emp_category'] = df['employment_length_years'].apply(cat_emp)
    
    return df

def align_model_columns(df: pd.DataFrame, model_columns: list) -> pd.DataFrame:
    # List of ALL categorical columns from notebook
    categorical_cols = [
        'credit_tier', 'emp_category', 
        'home_ownership', 'loan_intent', 
        'loan_grade', 'historical_default'
    ]
    
    present_cats = [col for col in categorical_cols if col in df.columns]
    
    # 1. Generate Dummies
    df_processed = pd.get_dummies(df, columns=present_cats)
    
    # 2. Create empty frame
    df_final = pd.DataFrame(columns=model_columns)
    
    # 3. Map processed data
    for col in df_processed.columns:
        if col in df_final.columns:
            df_final.loc[0, col] = df_processed[col].iloc[0]
            
    # 4. Fill missing with 0
    df_final = df_final.fillna(0)

    # --- CRITICAL FIX: FORCE FLOAT ---
    # Coerce everything to numeric, then float
    for col in df_final.columns:
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce').fillna(0)
    
    return df_final.astype(float)
