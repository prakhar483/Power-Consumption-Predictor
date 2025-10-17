import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Setting the Page Configuration first : 
st.set_page_config(page_title="⚡ Household Power Consumption Predictor", page_icon="🔌", layout="centered")


# --- Load model and scaler ---
@st.cache_resource
def load_artifacts():
    scaler = joblib.load('scaler_energy.pkl')
    model = load_model('ann_energy_model.h5',compile=False)
    return scaler, model

scaler, model = load_artifacts()

# --- App Title and Description ---
st.title("⚡ Household Power Consumption Predictor")
st.markdown("""
Welcome to the **Household Power Consumption Predictor**!  
This app predicts the **Global Active Power** (in kilowatts) using your input features.  
*Dataset: [UCI Household Power Consumption](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)*  
""")

with st.sidebar:
    st.header("📝 About the Dataset")
    st.write("""
    - **Global_active_power**: Household global minute-averaged active power (kilowatt)
    - **Global_reactive_power**: Household global minute-averaged reactive power (kilowatt)
    - **Voltage**: Minute-averaged voltage (volt)
    - **Global_intensity**: Minute-averaged current intensity (ampere)
    - **Sub_metering_3**: Energy sub-metering No. 3 (watt-hour of active energy)
    - **hour**: Hour of the day (0-23)
    """)

    st.markdown("---")
    st.info("👈 Enter the input features on the main page to get a prediction!")

# --- Input Form ---
st.subheader("🔢 Enter Input Features")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        global_intensity = st.number_input("Global Intensity (A)", min_value=0.0, max_value=50.0, value=18.4, step=0.1)
        voltage = st.number_input("Voltage (V)", min_value=200.0, max_value=260.0, value=234.84, step=0.1)
        hour = st.slider("Hour of Day", min_value=0, max_value=23, value=17)
    with col2:
        global_reactive_power = st.number_input("Global Reactive Power (kW)", min_value=0.0, max_value=2.0, value=0.418, step=0.001)
        sub_metering_3 = st.number_input("Sub Metering 3 (Wh)", min_value=0.0, max_value=50.0, value=17.0, step=0.1)

    submitted = st.form_submit_button("🔮 Predict")

# --- Prediction ---
if submitted:
    input_data = np.array([[global_intensity, global_reactive_power, voltage, sub_metering_3, hour]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success(f"🌟 **Predicted Global Active Power:** `{prediction[0][0]:.3f} kW`")

    st.balloons()
    st.markdown("##### 📊 Model Input Summary")
    st.dataframe(pd.DataFrame(input_data, columns=['Global_intensity', 'Global_reactive_power', 'Voltage', 'Sub_metering_3', 'hour']))

st.markdown("---")
st.caption("Made with ❤️ by Prakhar Dwivedi using Streamlit | [UCI Dataset](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)")