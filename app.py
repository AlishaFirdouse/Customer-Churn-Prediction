from fpdf import FPDF

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
