import streamlit as st
import pickle
import pandas as pd

# Load trained model
model = pickle.load(open("churn_model.pkl", "rb"))

# Page Title
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊"
)

st.title("🏦 Credit Card Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn or stay.")

# User Inputs

Customer_Age = st.number_input(
    "Customer Age",
    min_value=18,
    max_value=100,
    value=40
)

Months_Inactive_12_mon = st.number_input(
    "Months Inactive (Last 12 Months)",
    min_value=0,
    max_value=12,
    value=2
)

Credit_Limit = st.number_input(
    "Credit Limit",
    min_value=1000,
    max_value=50000,
    value=10000
)

Total_Trans_Amt = st.number_input(
    "Total Transaction Amount",
    min_value=0,
    max_value=20000,
    value=5000
)

Total_Trans_Ct = st.number_input(
    "Total Transaction Count",
    min_value=0,
    max_value=200,
    value=50
)

# Prediction Button

if st.button("Predict"):

    sample = pd.DataFrame({
        "Customer_Age": [Customer_Age],
        "Months_Inactive_12_mon": [Months_Inactive_12_mon],
        "Credit_Limit": [Credit_Limit],
        "Total_Trans_Amt": [Total_Trans_Amt],
        "Total_Trans_Ct": [Total_Trans_Ct]
    })

    prediction = model.predict(sample)

    # Probability
    probability = model.predict_proba(sample)[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:

        st.error("⚠️ High Churn Risk Customer")

    else:

        st.success("✅ Customer Likely To Stay")

    st.write(
        f"**Churn Probability:** {probability * 100:.2f}%"
    )

    st.progress(float(probability))

    # Business Insight

    if probability > 0.70:

        st.warning(
            "Recommended Action: Offer retention benefits, loyalty rewards, or personalized engagement."
        )

    elif probability > 0.40:

        st.info(
            "Recommended Action: Monitor customer activity and engagement."
        )

    else:

        st.success(
            "Customer appears healthy and actively engaged."
        )