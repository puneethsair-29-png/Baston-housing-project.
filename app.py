
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("house_price_model.pkl")

# Title
st.title("🏠 Boston House Price Prediction")

st.write("Enter the house details below to predict the house price.")

# Input fields
CRIM = st.number_input("CRIM", value=0.0)
ZN = st.number_input("ZN", value=0.0)
INDUS = st.number_input("INDUS", value=0.0)
CHAS = st.number_input("CHAS", value=0.0)
NOX = st.number_input("NOX", value=0.0)
RM = st.number_input("RM", value=0.0)
AGE = st.number_input("AGE", value=0.0)
DIS = st.number_input("DIS", value=0.0)
RAD = st.number_input("RAD", value=0.0)
TAX = st.number_input("TAX", value=0.0)
PTRATIO = st.number_input("PTRATIO", value=0.0)
B = st.number_input("B", value=0.0)
LSTAT = st.number_input("LSTAT", value=0.0)

# Prediction button
if st.button("Predict House Price"):

    # Create input DataFrame
    input_data = pd.DataFrame([[
        CRIM,
        ZN,
        INDUS,
        CHAS,
        NOX,
        RM,
        AGE,
        DIS,
        RAD,
        TAX,
        PTRATIO,
        B,
        LSTAT
    ]], columns=[
        "CRIM",
        "ZN",
        "INDUS",
        "CHAS",
        "NOX",
        "RM",
        "AGE",
        "DIS",
        "RAD",
        "TAX",
        "PTRATIO",
        "B",
        "LSTAT"
    ])

    # Predict
    prediction = model.predict(input_data)

    # Display result
    st.success(
        f"Predicted House Price: ${prediction[0]:.2f}"
    )
