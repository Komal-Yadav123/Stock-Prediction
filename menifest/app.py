import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os

st.title("📈 Stock Price Prediction App")

# User input for stock ticker
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL").upper()

# Date range selection
start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2023-12-31"))

if start_date >= end_date:
    st.error("Error: End date must fall after start date.")
else:
    # Fetch historical data
    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        st.warning("No data found for the given ticker and date range.")
    else:
        st.subheader(f"Historical Closing Prices for {ticker}")
        st.line_chart(data['Close'])

        # Data preprocessing
        df = data[['Close']]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df)

        # Prepare training data
        sequence_length = 60
        X = []
        y = []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Load or train model
        model_file = f"{ticker}_model.h5"
        if os.path.exists(model_file):
            model = load_model(model_file)
        else:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense

            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
            model.add(LSTM(units=50))
            model.add(Dense(1))

            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X, y, epochs=5, batch_size=32)
            model.save(model_file)

        # Predict next day's price
        last_60_days = scaled_data[-sequence_length:]
        X_test = np.reshape(last_60_days, (1, sequence_length, 1))
        predicted_price = model.predict(X_test)
        predicted_price = scaler.inverse_transform(predicted_price)

        st.subheader(f"Predicted Closing Price for Next Day: ${predicted_price[0][0]:.2f}")

