1 Simple XAI Risk (Credit Risk) Documentation
This process documents the creation of a compliant and transparent risk scoring system.

Phase 1: Data Preparation and Feature Selection
Data Simulation: Generate or load a simulated dataset containing key credit risk features (Credit_Score, Debt_to_Income, Num_Delinquencies, Loan_Term_Months) and a binary target (Default_Target).

Feature Definition: Verify features are clean (no complex transformations needed, given the simplicity goal).

Phase 2: Model Training and Explainability
Model Selection: Choose Logistic Regression for its inherent transparency and ease of interpreting coefficients.

Training: Fit the model using the prepared feature set.

Coefficient Extraction: Extract the learned model coefficients for each feature. These coefficients serve as the primary explanation (XAI).

Phase 3: Actionable Output and Streamlit Deployment
Prediction Logic: Define the final prediction function to calculate the probability of default for a new applicant.

Thresholding: Establish the clear Accept/Reject threshold (e.g., 50% default probability).

Adverse Action Logic: Implement conditional logic in Streamlit to trigger the display of rejection reasons based on the applicant's input values and the direction of the model's coefficients (e.g., if DTI > 0.5 and the DTI coefficient is positive).

Documentation Output: Document the exact threshold and the coefficient-based rules used for compliance.
