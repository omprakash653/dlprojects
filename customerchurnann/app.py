import streamlit as st
import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ==================================================
# CUSTOM CSS
# ==================================================
st.markdown("""
<style>
.main-title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#1E88E5;
}
.sub-title{
    text-align:center;
    color:gray;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<p class="main-title">📊 Customer Churn Prediction System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">Deep Learning ANN Model</p>',
    unsafe_allow_html=True
)

st.divider()

# ==================================================
# LOAD MODEL
# ==================================================
model = load_model("telco_churn_ann.keras")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox(
    "Gender",
    ["Female","Male"]
)

senior = st.sidebar.selectbox(
    "Senior Citizen",
    [0,1]
)

partner = st.sidebar.selectbox(
    "Partner",
    ["No","Yes"]
)

dependents = st.sidebar.selectbox(
    "Dependents",
    ["No","Yes"]
)

tenure = st.sidebar.slider(
    "Tenure",
    0,
    72,
    24
)

phone = st.sidebar.selectbox(
    "Phone Service",
    ["No","Yes"]
)

multiple = st.sidebar.selectbox(
    "Multiple Lines",
    ["No","Yes","No phone service"]
)

internet = st.sidebar.selectbox(
    "Internet Service",
    ["DSL","Fiber optic","No"]
)

online_security = st.sidebar.selectbox(
    "Online Security",
    ["No","Yes","No internet service"]
)

online_backup = st.sidebar.selectbox(
    "Online Backup",
    ["No","Yes","No internet service"]
)

device_protection = st.sidebar.selectbox(
    "Device Protection",
    ["No","Yes","No internet service"]
)

tech_support = st.sidebar.selectbox(
    "Tech Support",
    ["No","Yes","No internet service"]
)

stream_tv = st.sidebar.selectbox(
    "Streaming TV",
    ["No","Yes","No internet service"]
)

stream_movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["No","Yes","No internet service"]
)

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month","One year","Two year"]
)

paperless = st.sidebar.selectbox(
    "Paperless Billing",
    ["No","Yes"]
)

payment = st.sidebar.selectbox(
    "Payment Method",
    [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check"
    ]
)

monthlycharges = st.sidebar.slider(
    "Monthly Charges",
    0.0,
    150.0,
    75.0
)

totalcharges = st.sidebar.number_input(
    "Total Charges",
    0.0,
    10000.0,
    2500.0
)

# ==================================================
# CREATE INPUT DATAFRAME
# ==================================================
input_df = pd.DataFrame(
    np.zeros((1, len(columns))),
    columns=columns
)

# ==================================================
# NUMERIC FEATURES
# ==================================================
input_df["SeniorCitizen"] = senior
input_df["tenure"] = tenure
input_df["MonthlyCharges"] = monthlycharges
input_df["TotalCharges"] = totalcharges

# ==================================================
# ENCODE CATEGORICAL VARIABLES
# ==================================================

if gender == "Male":
    input_df["gender_Male"] = 1

if partner == "Yes":
    input_df["Partner_Yes"] = 1

if dependents == "Yes":
    input_df["Dependents_Yes"] = 1

if phone == "Yes":
    input_df["PhoneService_Yes"] = 1

if multiple == "Yes":
    input_df["MultipleLines_Yes"] = 1
elif multiple == "No phone service":
    input_df["MultipleLines_No phone service"] = 1

if internet == "Fiber optic":
    input_df["InternetService_Fiber optic"] = 1
elif internet == "No":
    input_df["InternetService_No"] = 1

if online_security == "Yes":
    input_df["OnlineSecurity_Yes"] = 1
elif online_security == "No internet service":
    input_df["OnlineSecurity_No internet service"] = 1

if online_backup == "Yes":
    input_df["OnlineBackup_Yes"] = 1
elif online_backup == "No internet service":
    input_df["OnlineBackup_No internet service"] = 1

if device_protection == "Yes":
    input_df["DeviceProtection_Yes"] = 1
elif device_protection == "No internet service":
    input_df["DeviceProtection_No internet service"] = 1

if tech_support == "Yes":
    input_df["TechSupport_Yes"] = 1
elif tech_support == "No internet service":
    input_df["TechSupport_No internet service"] = 1

if stream_tv == "Yes":
    input_df["StreamingTV_Yes"] = 1
elif stream_tv == "No internet service":
    input_df["StreamingTV_No internet service"] = 1

if stream_movies == "Yes":
    input_df["StreamingMovies_Yes"] = 1
elif stream_movies == "No internet service":
    input_df["StreamingMovies_No internet service"] = 1

if contract == "One year":
    input_df["Contract_One year"] = 1
elif contract == "Two year":
    input_df["Contract_Two year"] = 1

if paperless == "Yes":
    input_df["PaperlessBilling_Yes"] = 1

if payment == "Credit card (automatic)":
    input_df["PaymentMethod_Credit card (automatic)"] = 1

elif payment == "Electronic check":
    input_df["PaymentMethod_Electronic check"] = 1

elif payment == "Mailed check":
    input_df["PaymentMethod_Mailed check"] = 1

# ==================================================
# SCALE NUMERIC FEATURES
# ==================================================
num_cols = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

input_df[num_cols] = scaler.transform(
    input_df[num_cols]
)

# ==================================================
# PREDICTION
# ==================================================
if st.button("🔍 Predict Churn"):

    prob = model.predict(
        input_df,
        verbose=0
    )[0][0]

    st.subheader("Prediction Result")

    st.progress(float(prob))

    st.metric(
        "Churn Probability",
        f"{prob:.2%}"
    )

    if prob >= 0.5:

        st.error(
            f"⚠ Customer likely to CHURN\n\nConfidence: {prob:.2%}"
        )

    else:

        st.success(
            f"✅ Customer likely to STAY\n\nConfidence: {(1-prob):.2%}"
        )

# ==================================================
# DEBUG
# ==================================================
with st.expander("Debug Info"):

    st.write("Model Input Shape:", model.input_shape)

    st.write("Input Shape:", input_df.shape)

    st.dataframe(input_df)