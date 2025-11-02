import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
import datetime

# Try importing TensorFlow
use_lstm = True
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except Exception:
    use_lstm = False
    from sklearn.neural_network import MLPRegressor

st.set_page_config(page_title="Stock Price Prediction", layout="wide")

st.title("📈 Stock Price Prediction App")
st.write("Predict next-day and 30-day forecast using pre-trained model and metadata.")

# Load metadata
meta_path = "symbols_valid_meta.csv"
if not os.path.exists(meta_path):
    st.error("symbols_valid_meta.csv not found.")
    st.stop()

meta = pd.read_csv(meta_path)
symbol_col = None
for candidate in ['symbol', 'Symbol', 'SYMBOL']:
    if candidate in meta.columns:
        symbol_col = candidate
        break

symbols = sorted(meta[symbol_col].dropna().unique().tolist())
default_symbol = 'AAPL' if 'AAPL' in symbols else symbols[0]

selected_symbol = st.sidebar.selectbox("Select a stock symbol", symbols, index=symbols.index(default_symbol))

# Company info
company_info = meta[meta[symbol_col] == selected_symbol]
if not company_info.empty:
    st.subheader("Company Information")
    st.dataframe(company_info)

# Load CSV for selected stock
csv_path = f"{selected_symbol}.csv"
if not os.path.exists(csv_path):
    st.warning(f"No CSV found for {selected_symbol}. Please upload {selected_symbol}.csv to continue.")
    uploaded_file = st.file_uploader("Upload stock CSV file", type="csv")
    if uploaded_file is not None:
        with open(csv_path, "wb") as f:
            f.write(uploaded_file.read())
    else:
        st.stop()

df = pd.read_csv(csv_path)
if 'Date' not in df.columns or 'Close' not in df.columns:
    st.error("CSV file must contain 'Date' and 'Close' columns.")
    st.stop()

# Preprocess
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df = df.dropna(subset=['Close'])
st.subheader(f"{selected_symbol} Closing Price Data")
st.line_chart(df.set_index('Date')['Close'])

# Load scaler
scaler_path = "models/close_scaler.save"
if not os.path.exists(scaler_path):
    st.error("Scaler file not found. Please train model first.")
    st.stop()
scaler = joblib.load(scaler_path)

# Model paths
lstm_path = "models/lstm_aapl_model.h5"
mlp_path = "models/mlp_aapl_model.joblib"

# Load appropriate model
if use_lstm and os.path.exists(lstm_path):
    model = load_model(lstm_path)
    model_type = "LSTM"
else:
    if not os.path.exists(mlp_path):
        st.error("No trained model found. Please run notebook training first.")
        st.stop()
    model = joblib.load(mlp_path)
    use_lstm = False
    model_type = "MLPRegressor"

st.sidebar.write(f"Loaded model type: {model_type}")

# Prepare data
close_values = df['Close'].values.reshape(-1, 1)
scaled_close = scaler.transform(close_values)

window = 60

# Prediction functions
def inv_scale(arr):
    arr = np.array(arr).reshape(-1, 1)
    return scaler.inverse_transform(arr).reshape(-1)

def iterative_forecast(model_obj, start_scaled, n_days, use_lstm_flag):
    preds = []
    buffer = start_scaled.copy()
    for _ in range(n_days):
        if use_lstm_flag:
            x = buffer.reshape(1, window, 1)
            y_s = model_obj.predict(x, verbose=0)
        else:
            x = buffer.reshape(1, window)
            y_s = model_obj.predict(x).reshape(-1,1)
        preds.append(y_s[0,0])
        buffer = np.vstack([buffer[1:], [[y_s[0,0]]]])
    return np.array(preds)

# Forecast
last_window = scaled_close[-window:].reshape(window, 1)
next_scaled = model.predict(last_window.reshape(1, window, 1)) if use_lstm else model.predict(last_window.reshape(1, window))
next_price = inv_scale(next_scaled)[0]
st.metric("Next-Day Predicted Close", f"${next_price:.2f}")

# 30-day forecast
future_scaled = iterative_forecast(model, last_window, 30, use_lstm)
future_prices = inv_scale(future_scaled)
future_dates = pd.date_range(df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=30, freq='B')

# Plot
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(df['Date'], df['Close'], label='Historical Close')
ax.plot(future_dates, future_prices, '--', label='30-Day Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Close Price (USD)')
ax.set_title(f'{selected_symbol} Forecast ({model_type})')
ax.legend()
st.pyplot(fig)

st.write("Forecasted Prices (first 5 days):")
st.dataframe(pd.DataFrame({
    "Date": future_dates[:5],
    "Predicted Close": future_prices[:5]
}))
