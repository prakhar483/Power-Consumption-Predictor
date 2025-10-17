# 🔌 Energy Consumption Forecasting Using ANN & Streamlit

> Predict household electricity usage for a given time range using historical energy metrics and a trained Artificial Neural Network (ANN) model.

---

## 📘 Project Overview

This project demonstrates how machine learning and deep learning can be used to forecast **household energy consumption**. Using the **UCI Household Power Consumption Dataset**, we built and deployed an **Artificial Neural Network (ANN)** model capable of predicting power usage during specific hours of the day based on electrical measurements like:

- Global intensity (A)
- Voltage (V)
- Reactive power (kW)
- Sub-metering (kWh)
- Hour range (e.g., 1 PM to 5 PM)

A simple and intuitive **Streamlit web app** allows users to interact with the model by inputting these features and getting instant predictions.

---

## 🧾 Files in the Repository

- ├── Ann_Project_3_Energy_Consumption.ipynb # Jupyter notebook with preprocessing, training, evaluation
- ├── Energy_app.py # Streamlit app script
- ├── ann_energy_model.h5 # Trained ANN model
- ├── scaler_energy.pkl # MinMaxScaler used for input normalization


> ⚠️ **Note:** Due to GitHub file size limitations, the original dataset is not included. You can download it manually from the [UCI Repository](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption).

---

## 📊 Dataset Info

- 📈 **Rows:** 2,075,259  
- 📉 **Columns:** 9  
- 🗂️ **Size:** ~120 MB (CSV)

### Key Columns:
| Column | Description |
|--------|-------------|
| Global_active_power | Target variable (kW) |
| Global_reactive_power | Reactive power (kW) |
| Voltage | Voltage (V) |
| Global_intensity | Current intensity (A) |
| Sub_metering_3 | Sub-metered energy usage (kWh) |
| Date, Time | Used to extract `hour` feature |

---

## 🚀 How the App Works

Users input the following:
- Global intensity (A)
- Global reactive power (kW)
- Voltage (V)
- Sub_metering_3 (kWh)
- Start and end hour (e.g., 13 to 17)

✅ The model predicts **total power consumption** for the given hour range.  
✅ Powered by **Streamlit** for a clean, responsive web UI.  
✅ Supports real-time predictions using a trained ANN model.
