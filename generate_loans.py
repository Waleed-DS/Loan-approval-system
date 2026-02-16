import pandas as pd
import numpy as np

def generate_loan_data(n_samples=10000):
    print("Generating synthetic loan data...")
    np.random.seed(42)
    
    # Generate Features
    incomes = np.random.lognormal(mean=10.5, sigma=0.6, size=n_samples) 
    credit_scores = np.random.normal(650, 100, n_samples)
    credit_scores = np.clip(credit_scores, 300, 850)
    
    loan_amounts = np.random.choice([5000, 10000, 15000, 20000, 25000], size=n_samples)
    employment_length = np.random.randint(0, 40, size=n_samples)
    
    # Risk Logic
    risk_score = (
        (850 - credit_scores) / 850 * 2 + 
        (loan_amounts / incomes) * 5 - 
        (employment_length * 0.05)
    )
    risk_score += np.random.normal(0, 0.2, n_samples)
    prob_default = 1 / (1 + np.exp(-risk_score + 1)) 
    defaults = np.random.binomial(1, prob_default)
    
    df = pd.DataFrame({
        'income': np.round(incomes, 2),
        'credit_score': np.round(credit_scores).astype(int),
        'loan_amount': loan_amounts,
        'employment_length_years': employment_length,
        'loan_status': defaults 
    })
    

    df.loc[np.random.choice(df.index, 50), 'income'] = np.nan
    df.loc[np.random.choice(df.index, 5), 'income'] = -50000
    
    output_path = 'historical_loans.csv'
    df.to_csv(output_path, index=False)
    print(f"Success! Data saved to {output_path}")

if __name__ == "__main__":
    generate_loan_data()