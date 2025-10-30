import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

np.random.seed(42)

def train_simple_model():
    """Trains a simple Logistic Regression model for credit risk."""
    n_samples = 1000
    df = pd.DataFrame({
        'Credit_Score': np.random.randint(450, 850, n_samples),
        'Debt_to_Income': np.random.uniform(0.1, 0.7, n_samples),
        'Num_Delinquencies': np.random.randint(0, 5, n_samples),
        'Loan_Term_Months': np.random.choice([12, 36, 60], n_samples)
    })
    
    # Simple risk formula: Low Score, High DTI, High Delinquency -> Higher Risk
    risk = 0.01 * df['Credit_Score'] + 5 * df['Debt_to_Income'] + 0.5 * df['Num_Delinquencies']
    df['Default_Target'] = (risk > 7.5).astype(int) 

    X = df[['Credit_Score', 'Debt_to_Income', 'Num_Delinquencies', 'Loan_Term_Months']]
    y = df['Default_Target']

    # Train a simple model
    model = LogisticRegression(solver='liblinear')
    model.fit(X, y)
    
    return model, X.columns

@st.cache_resource
def load_data_and_model_p1():
    return train_simple_model()

def run_project1():
    st.header("Project 1: 💡 Simple Explainability for Adverse Action (Credit Risk)")
    st.markdown("---")
    
    model, features = load_data_and_model_p1()

    # Simplified UI inputs
    st.sidebar.subheader("Inputs for Project 1")
    cs = st.sidebar.slider('Credit Score', 450, 850, 600)
    dti = st.sidebar.slider('DTI Ratio', 0.10, 0.70, 0.55, 0.01)
    delinq = st.sidebar.slider('Num Delinquencies', 0, 5, 3)
    term = st.sidebar.selectbox('Loan Term (Months)', [12, 36, 60], index=2)
    
    input_data = pd.DataFrame({
        'Credit_Score': [cs], 'Debt_to_Income': [dti], 
        'Num_Delinquencies': [delinq], 'Loan_Term_Months': [term]
    })
    
    risk_prob = model.predict_proba(input_data)[:, 1][0]
    ACCEPT_THRESHOLD = 0.50

    st.subheader(f"Decision: {'🔴 REJECTED' if risk_prob >= ACCEPT_THRESHOLD else '🟢 ACCEPTED'}")
    st.metric("Predicted Default Probability", f"{risk_prob*100:.2f}%")
    
    if risk_prob >= ACCEPT_THRESHOLD:
        st.subheader("Adverse Action Reasons (Based on Model Coefficients)")
        st.warning("⚠️ Factors that increased the risk:")
        
        # Interpret coefficients (simpler than SHAP)
        coefficients = pd.Series(model.coef_[0], index=features)
        
        # Logic: If feature value pushes risk higher (e.g., high DTI, high Delinquency)
        reasons = []
        if dti > 0.5 and coefficients['Debt_to_Income'] > 0:
            reasons.append("High Debt-to-Income Ratio.")
        if cs < 620 and coefficients['Credit_Score'] < 0:
            reasons.append("Low Credit Score History.")
        if delinq >= 3 and coefficients['Num_Delinquencies'] > 0:
            reasons.append("Excessive Number of Recent Delinquencies.")
            
        for i, reason in enumerate(reasons[:3]):
            st.markdown(f"**{i+1}.** {reason}")
    else:
        st.success("Applicant meets criteria.")