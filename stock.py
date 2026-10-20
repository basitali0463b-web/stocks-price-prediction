import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="centered")

st.title("📈 Stock Price Prediction App")
st.write("Random Forest Model ka use karke stock ki Close price predict karein!")

# Load Trained Model
@st.cache_resource
def load_model():
    return joblib.load('stochs.pkl')

model = load_model()

st.subheader("📊 Stock Indicators Input Karein")

# User se inputs lene ke liye clean layout
col1, col2 = st.columns(2)

with col1:
    open_price = st.number_input("Open Price ($)", value=50.0, step=0.1)
    high_price = st.number_input("High Price ($)", value=52.0, step=0.1)
    low_price = st.number_input("Low Price ($)", value=49.0, step=0.1)

with col2:
    volume = st.number_input("Volume", value=1000000, step=1000)
    ma_5 = st.number_input("5-Day Moving Avg (MA_5)", value=50.5, step=0.1)
    close_lag1 = st.number_input("Previous Close (Close_Lag1)", value=49.8, step=0.1)

# Predict Button
if st.button("🚀 Predict Close Price", type="primary"):
    # Input data ko DataFrame mein convert karein
    input_data = pd.DataFrame(
        [[open_price, high_price, low_price, volume, ma_5, close_lag1]],
        columns=['Open', 'High', 'Low', 'Volume', 'MA_5', 'Close_Lag1']
    )
    
    # Prediction nikalein
    prediction = model.predict(input_data)
    
    # Result show karein
    st.success(f"💰 **Predicted Close Price:** ${prediction[0]:.2f}")
