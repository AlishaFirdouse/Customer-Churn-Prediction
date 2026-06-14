from fpdf import FPDF
import os
import csv
from datetime import datetime
import streamlit as st
import pandas as pd
import pickle

# ================= LOAD MODEL ================= #
model = pickle.load(open("churn_model.pkl", "rb"))

# ================= PAGE CONFIG ================= #
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ================= TITLE ================= #
st.markdown("<h1>📊 Customer Churn Prediction System</h1>", unsafe_allow_html=True)
st.info("AI-powered prediction: Will your customer stay or leave?")

# ================= INPUT SECTION ================= #
st.markdown("## 🧾 Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])

with col2:
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])

with col3:
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

col4, col5 = st.columns(2)

with col4:
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    )

with col5:
    monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.5)
    total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

# ================= CONVERSION ================= #
gender = 0 if gender == "Female" else 1
senior = 0 if senior == "No" else 1
partner = 0 if partner == "No" else 1
dependents = 0 if dependents == "No" else 1
phone_service = 0 if phone_service == "No" else 1
multiple_lines = 0 if multiple_lines == "No" else 1
internet_service = 0 if internet_service == "No" else (1 if internet_service == "DSL" else 2)
contract = 0 if contract == "Month-to-month" else (1 if contract == "One year" else 2)
paperless_billing = 0 if paperless_billing == "No" else 1

payment_method = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer": 2,
    "Credit card": 3
}[payment_method]

# ================= PREDICTION ================= #
st.markdown("---")

if st.button("🚀 Predict Churn"):

    # Create input
    new_customer = pd.DataFrame([[ 
        gender, senior, partner, dependents,
        tenure,
        phone_service, multiple_lines, internet_service,
        0, 0, 0, 0, 0, 0,
        contract, paperless_billing, payment_method,
        monthly_charges, total_charges
    ]])

    # Predict
    with st.spinner("Analyzing customer behavior..."):
        prediction = model.predict(new_customer)
        probability = model.predict_proba(new_customer)

    churn_prob = probability[0][1] * 100

    # ================= RESULT ================= #
    st.markdown("## 📊 Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to STAY")

    # ================= RISK METER ================= #
    st.markdown("### 🎯 Risk Meter")
    st.progress(int(churn_prob))

    st.markdown(f"""
    <div style="
        text-align:center;
        font-size:22px;
        font-weight:bold;
        padding:15px;
        border-radius:12px;
        background: {'#ffebee' if churn_prob > 70 else '#fff3e0' if churn_prob > 30 else '#e8f5e9'};
        color: {'#c62828' if churn_prob > 70 else '#ef6c00' if churn_prob > 30 else '#2e7d32'};
    ">
        {churn_prob:.2f}% Risk Score
    </div>
    """, unsafe_allow_html=True)

    st.metric("Churn Probability", f"{churn_prob:.2f}%")

    # ================= AI INSIGHTS ================= #
    st.markdown("### 🧠 AI Insight Engine")

    insights = []

    if tenure < 12:
        insights.append("⚠️ Short tenure increases churn risk")

    if monthly_charges > 80:
        insights.append("⚠️ High monthly charges increase churn risk")

    if contract == 0:
        insights.append("⚠️ Month-to-month contract increases churn risk")

    if internet_service == 2:
        insights.append("⚠️ Fiber optic users have higher churn tendency")

    if len(insights) == 0:
        insights.append("✅ Customer profile looks stable")

    for i in insights:
        st.write("•", i)

    # ================= SAVE TO CSV ================= #
    data = [
        datetime.now(),
        float(churn_prob),
        int(prediction[0])
    ]

    with open("predictions.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(data)

    st.success("📁 Saved to history")

# ================= PDF CREATION ================= #
def create_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title
    pdf.cell(200, 10, txt="Customer Churn Prediction Report", ln=True, align="C")
    pdf.ln(10)

    # Bold headers
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(60, 10, "Time", 1)
    pdf.cell(60, 10, "Churn Probability", 1)
    pdf.cell(60, 10, "Prediction", 1)
    pdf.ln()

    # Normal rows
    pdf.set_font("Arial", '', 12)
    for _, row in df.iterrows():
        pdf.cell(60, 10, str(row["Time"]), 1)
        pdf.cell(60, 10, f"{row['Churn Probability']:.2f}", 1)
        pdf.cell(60, 10, str(row["Prediction"]), 1)
        pdf.ln()

    return pdf.output(dest="S").encode("latin-1")


# ================= HISTORY ================= #
st.markdown("### 📊 Prediction History")

if os.path.exists("predictions.csv"):
    df = pd.read_csv("predictions.csv", names=["Time", "Churn Probability", "Prediction"])
    st.dataframe(df.tail(10))

    # Prepare files
    csv_file = df.to_csv(index=False).encode('utf-8')
    pdf_file = create_pdf(df)

    # Show buttons side by side
    colA, colB = st.columns(2)

    with colA:
        st.download_button(
            label="📥 Download Full Report (CSV)",
            data=csv_file,
            file_name="churn_history.csv",
            mime="text/csv"
        )

    with colB:
        st.download_button(
            label="📥 Download Full Report (PDF)",
            data=pdf_file,
            file_name="churn_history.pdf",
            mime="application/pdf"
        )
