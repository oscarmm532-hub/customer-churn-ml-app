import os
import sys
import joblib
import pandas as pd
import streamlit as st

# Permitir importar el script de entrenamiento desde src/
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from train import train_churn_model

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="centered")

MODEL_PATH = "models/churn_model.joblib"

# Si el modelo aún no existe en el servidor, lo entrena automáticamente
if not os.path.exists(MODEL_PATH):
    with st.spinner("Entrenando modelo predictivo por primera vez..."):
        train_churn_model()

model = joblib.load(MODEL_PATH)

st.title("📊 Customer Churn Prediction App")
st.markdown(
    "Interactive Machine Learning application built with **Scikit-Learn** and **Streamlit** "
    "to estimate the probability of a telecom customer canceling their service."
)

st.subheader("Customer Profile")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (Months with company)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=70.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=float(tenure * monthly_charges))
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

with col2:
    internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])

if st.button("Predict Churn Risk", type="primary"):
    input_data = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract,
        "InternetService": internet_service,
        "PaymentMethod": payment_method,
        "TechSupport": tech_support,
        "OnlineSecurity": online_security
    }])

    churn_prob = model.predict_proba(input_data)[0][1]
    risk_pct = churn_prob * 100

    st.divider()
    st.metric(label="Churn Probability", value=f"{risk_pct:.1f}%")

    if churn_prob >= 0.50:
        st.error("⚠️ High Risk of Churn: Consider offering a loyalty discount or annual contract upgrade.")
    else:
        st.success("✅ Low Risk of Churn: Customer shows stable retention indicators.")
