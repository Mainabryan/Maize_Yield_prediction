
import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Set page config
st.set_page_config(page_title="Kenya Maize Yield Predictor", layout="centered")

# App title
st.title("🌾 Kenya Maize Yield Predictor")
st.markdown("Predict maize yield per hectare based on key agricultural inputs in Kenya 🇰🇪")

# Load trained model and scaler
model = joblib.load("ridge_model.pkl")
scaler = joblib.load("scaler.pkl")

# Sidebar input form
st.sidebar.header("Input Farmer Data 🌱")
rainfall = st.sidebar.slider("🌧️ Average Rainfall (mm)", 100, 1000, 450)
temperature = st.sidebar.slider("🌡️ Average Temperature (°C)", 10, 40, 25)
soil_quality = st.sidebar.slider("🧪 Soil Quality (0=Poor, 1=Good)", 0, 1, 1)
fertilizer = st.sidebar.slider("🧴 Fertilizer Usage (kg/acre)", 0, 200, 100)
prev_yield = st.sidebar.number_input("📊 Previous Season Yield (t/ha)", 0.0, 10.0, 3.0)
altitude = st.sidebar.slider("⛰️ Altitude (meters)", 500, 3000, 1500)
seed_type = st.sidebar.selectbox("🌱 Seed Type", ["Local", "Hybrid"])
method = st.sidebar.selectbox("🚜 Farming Method", ["Manual", "Mechanized"])
season = st.sidebar.selectbox("☁️ Planting Season", ["Long Rains", "Short Rains"])

# One-hot encoding for categorical features
seed_type_hybrid = 1 if seed_type == "Hybrid" else 0
farming_method_mechanized = 1 if method == "Mechanized" else 0
season_short = 1 if season == "Short Rains" else 0

# Prepare input
X_input = np.array([[rainfall, temperature, soil_quality, fertilizer, prev_yield, altitude,
                     seed_type_hybrid, farming_method_mechanized, season_short]])

X_scaled = scaler.transform(X_input)

# Prediction
if st.sidebar.button("Predict Yield"):
    predicted_yield = model.predict(X_scaled)[0]
    st.subheader("🔍 Predicted Yield:")
    st.success(f"{predicted_yield:.2f} tonnes per hectare")

    # Recommendation logic
    if predicted_yield < 2:
        st.warning("⚠️ Yield is low. Consider improving soil, using hybrid seeds, or mechanizing.")
    elif 2 <= predicted_yield <= 5:
        st.info("📈 Moderate yield. Good! Consider optimizing fertilizer or seed type.")
    else:
        st.success("🌟 High yield! You're doing great. Keep up the sustainable practices.")

# Footer
st.markdown("---")
st.markdown("Made by **Bryan Waweru** | Data Science x Agriculture")
