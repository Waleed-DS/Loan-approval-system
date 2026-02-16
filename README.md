Executive Summary:

This project builds a machine learning system to predict loan defaults for a high-risk consumer. Using XGBoost, the model identifies risky borrowers with an ROC-AUC of 0.83, significantly outperforming a Logistic Regression baseline.

Unlike standard models that rely on raw data, this engine features custom financial engineering (e.g., Stress Index, Loan-to-Income Ratios) to capture the nuance of applicant debt burden. The final system includes a business-optimized decision threshold to balance risk (recall) with profitability (precision).

The Business Problem:

Context: A financial institution faces a high default rate (60.8%) in its subprime lending portfolio. Manual review is slow, inconsistent, and fails to detect complex risk interactions.

Goal: Build an automated scoring engine that:

Flags High-Risk Applicants: Identifies users likely to default.

Reduces False Rejections: Approves reliable borrowers even in a high-risk pool.

Explains Decisions: Uses interpretable risk drivers (e.g., Debt-to-Income) rather than "black box" logic.

Dataset Overview
High-Risk Consumer Lending Data
Total Records: 10,000 Loan Applicants
Target Variable: loan_status (Binary: 0 = Repaid, 1 = Default)

🚨 Class Imbalance & Risk Profile
This dataset represents a subprime / high-risk portfolio. Unlike traditional banking datasets where defaults are rare (1-5%), this population has a 60.8% Default Rate.

Default (Class 1): ~6,080 applicants (Majority)

Repaid (Class 0): ~3,920 applicants (Minority)

Note: This specific imbalance required Stratified Splitting and Class Weighting strategies during modeling to prevent the algorithm from ignoring the minority "Good" borrowers.

| Feature Name | Description | Type |
| :--- | :--- | :--- |
| income | Annual income of the applicant. | Numerical |
| loan amount | The total principal amount requested. | Numerical |
| credit score | FICO/Bureau score (300–850 range). | Numerical |
| loan to income | derived Ratio of loan size to annual income. Key indicator of repayment stress. | Ratio |
| employment length | Number of years in current job. Used to gauge income stability. | Numerical |


Data Cleaning & Integrity Report:

Ensuring High-Quality Input for Risk Modeling

Objective:
In credit risk modeling, "Garbage In, Garbage Out" is a critical failure mode. This module focuses on sanitizing the raw dataset to ensure the downstream XGBoost model receives robust, logically consistent, and bias-free data.

1. Missing Value Imputation
The Issue: The income variable contained 50 missing entries (NaNs). In a financial context, missing income usually implies unemployment or data entry error.

The Strategy: Applied Median Imputation.

Why Median? Income data is highly right-skewed (a few millionaires pull the average up). Using the Mean would artificially inflate the income of typical borrowers. The Median provides a more robust "typical" value.

Result: 0% missing values in the final training set.

2. Outlier Management (Winsorization)
The Issue: detected extreme outliers in income (e.g., applicants claiming >$1M annual income). While possible, these extreme values can distort the gradients in tree-based models and cause overfitting.

The Strategy: Applied 99th Percentile Winsorization (Capping).

Logic: Any income above the 99th percentile was capped at that 99th percentile value.

Benefit: This preserves the "high income" signal without letting extreme outliers skew the split points in the decision trees.

3. Logical Integrity Checks
Financial data must obey the laws of physics and banking rules. We enforced the following constraints:

Credit Score Validation: Removed or corrected any scores outside the FICO range (300–850).

Negative Values: Corrected negative integers found in employment_length_years and loan_amount (likely data entry typos) by converting them to absolute values.

Duplicate Removal: Scanned for and removed 0 exact duplicate rows to prevent data leakage between train and test sets.

4. Class Distribution Analysis
Observation: The dataset is heavily imbalanced with a 60.8% Default Rate (Target = 1).

Action: This verified the need for Stratified Splitting and Scale_Pos_Weight during the modeling phase to prevent the model from becoming biased toward the majority class.

🛠️ Technologies Used
Pandas: For manipulation and boolean indexing.

Seaborn/Matplotlib: For visualizing distributions (Boxplots) before and after cleaning.

Numpy: For statistical calculations (Median/Percentiles).

📉 Impact on Model
By rigorously cleaning the data before feature engineering, we ensured that our derived features (like loan_to_income) were calculated on valid numbers, directly contributing to the 0.83 AUC score in the final model.

Feature Engineering & Selection
Transforming Raw Data into Financial Risk Signals
Objective:
Raw financial data (e.g., "Income: 50000") is often insufficient for predicting complex behaviors like default. This module derives 14 new features designed to capture Financial Stress, Repayment Capacity, and Creditworthiness more effectively than raw variables alone.

1. Derived Financial Ratios (The "Heavy Lifters")
We engineered specific ratios to expose hidden risk factors that linear models might miss:

Loan-to-Income Ratio (LTI):

Formula: Loan Amount / Income

Business Logic: Applicants asking for >50% of their annual income are statistically more likely to default due to cash flow strain. This feature had a high correlation (0.42) with default.

Stress Index:

Formula: (Loan-to-Income) / (Credit Score / 850)

Business Logic: A high loan burden is dangerous, but it is critical when combined with a low credit score. This interaction feature captures "Double Jeopardy" risk—borrowers who are both over-leveraged and have a history of poor repayment.

High Loan Stress Flag:

Formula: 1 if (LTI > 0.5) else 0

Business Logic: Converts the continuous LTI ratio into a binary "Danger Flag" for the model. This simplifies the decision boundary for the algorithm and proved to be the single strongest predictor (0.43 correlation) in the entire dataset.

2. Categorical Bucketing (Non-Linearity)
Risk is rarely linear. A credit score drop from 700 to 600 is bad, but a drop from 600 to 500 is catastrophic. We captured this by binning continuous variables:

Credit Score Tiers:

Mapped raw FICO scores (300-850) into standard banking tiers: Poor, Fair, Good, Very Good, Excellent.

Impact: Allowed the model to treat "Poor" credit as a distinct, high-risk category rather than just a lower number.

Employment Stability:

Binned employment_length_years into three distinct groups:

New (<1 yr): High risk of income instability.

Stable (1-5 yrs): Standard risk profile.

Long-term (>5 yrs): Lower risk due to high tenure.

3. Encoding & Transformation
To make these categorical insights usable for the XGBoost algorithm, we applied One-Hot Encoding:

Converted text labels (e.g., credit_tier_Excellent) into binary vectors (0/1).

Result: The final dataset expanded to 14 features, giving the model granular control over exactly which specific traits drive risk.

4. Feature Selection Analysis
Before modeling, we validated the new features using a Correlation Matrix:

Retained: stress_index and loan_to_income showed strong positive correlation with loan_status (Default), confirming their predictive power.

Dropped: Removed raw IDs and redundant intermediate calculations to prevent multicollinearity and overfitting.

Impact on Model Performance
The inclusion of these engineered features was the primary driver for achieving an ROC-AUC of 0.83. The "Stress Index" consistently ranked as the #1 most important feature in the final XGBoost model, proving that context (debt relative to ability) matters more than raw numbers.


Model Selection & Training
Benchmarking & Champion Model Selection
Objective:
To determine the best algorithm for separating high-risk borrowers from safe ones, we benchmarked a simple linear baseline against a state-of-the-art gradient boosting machine.

 Model Comparison Results
We evaluated models using ROC-AUC (rank ordering ability) and PR-AUC (precision-recall balance), prioritizing the model's ability to minimize False Negatives (missed defaults).

Model,Type,ROC-AUC Score,Verdict
Logistic Regression,Baseline,0.81,"Good interpretability, but missed non-linear risk interactions."
XGBoost,Advanced (Champion),0.83,"Winner. Superior at capturing complex relationships (e.g., Stress Index)."

Training Methodology
We adhered to strict data science protocols to prevent data leakage and overfitting:

Class Imbalance Strategy:

The dataset has a 60.8% Default Rate.

Action: We applied scale_pos_weight in XGBoost and class_weight='balanced' in Logistic Regression. This penalizes the model more for missing the minority class, ensuring it learns to identify "Good" borrowers effectively despite the skew.

Cross-Validation:

Used 5-Fold Stratified Cross-Validation to ensure the model's performance was consistent across different subsets of the data, not just a lucky train/test split.

Conservative Hyperparameter Tuning:

Instead of chasing the highest possible score (which leads to overfitting), we prioritized stability.

Settings: max_depth=4 (prevents memorization), learning_rate=0.05 (slow, robust learning), and subsample=0.8 (reduces variance).

 Final Decision
XGBoost was selected as the production engine. The +0.02 AUC gain represents a significant improvement in risk separation, allowing the business to approve more loans with the same level of risk exposure compared to the baseline.

Evaluation
Assessing Model Reliability & Business Impact
Objective:
Beyond simple accuracy, we evaluated the model's ability to rank risk (ROC-AUC) and minimize financial loss (Precision-Recall), ensuring it works in a real-world lending environment.

1. ROC-AUC Comparison (Rank Ordering)
The primary metric for credit scoring is AUC (Area Under the Curve), which measures how well the model separates "Good" borrowers from "Bad" ones.

Logistic Regression: 0.81

XGBoost: 0.83

Result: XGBoost demonstrated superior separation, meaning it is less likely to confuse a risky borrower with a safe one.

2. Precision-Recall Analysis
Given the high default rate (60.8%), accuracy is misleading. We focused on the Precision-Recall trade-off:

Precision: Of the applicants we flagged as "High Risk," how many actually defaulted? (Minimizes lost revenue from false rejections).

Recall: Of the actual defaulters, how many did we catch? (Minimizes direct loss from bad loans).

Outcome: The model maintains high recall, catching the majority of defaults, while optimized thresholds ensure we don't aggressively reject viable customers.

3. Probability Calibration Check
A risk score of "80%" must mean an 80% chance of default.

Observation: The XGBoost model was slightly conservative, predicting slightly higher risk probabilities than the baseline.

Business Impact: In subprime lending, being slightly "risk-averse" is preferable to underestimating risk. The calibration curve showed reliable monotonicity (higher scores = consistently higher risk).

4. Feature Importance (Why the Model Decides)
We used Feature Importance plots to ensure the model wasn't relying on noise. The top drivers aligned perfectly with financial theory:

stress_index: (Highest Impact) - The interaction between debt burden and credit history.

loan_to_income: The sheer size of the loan relative to earnings.

high_loan_stress: The binary flag for extreme leverage.

Validation: Raw income and credit score were less important than these engineered "context" features, proving the value of our feature engineering step.

 Final Verdict
XGBoost was selected as the final production model.
It provided the best combination of predictive power (0.83 AUC), stability, and explainable risk drivers, making it the safest choice for deployment.

Key Findings:
Top Risk Driver: The engineered stress_index was the #1 predictor of default, validating our feature engineering strategy.

High-Risk Pool: The 60.8% default rate required a custom Probability Threshold (optimized via F1-Score) rather than the standard 0.50 cutoff.

Stability: The model uses conservative hyperparameters (max_depth=4, learning_rate=0.05) to ensure it generalizes well to new, unseen applicants.