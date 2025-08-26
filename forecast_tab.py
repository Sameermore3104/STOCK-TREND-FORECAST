# forecast_tab.py

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def forecast_tab():
    st.header("📈 Stock Forecast (Next 7 Days)")
    st.markdown("Predict the **next 7 closing prices** for any stock (like `TCS.NS`, `AAPL`, etc.) using an LSTM model.")

    model = load_model("stock_model.h5")
    scaler = joblib.load("scaler.save")

    stock_symbol = st.text_input("Enter Stock Symbol (e.g., TCS.NS, AAPL)", value="TCS.NS")
    predict_button = st.button("🔍 Predict")

    if predict_button:
        try:
            df = yf.download(stock_symbol, period="100d", interval="1d")
            df = df[["Close"]]
            df.dropna(inplace=True)

            if df.empty or len(df) < 60:
                st.error("❌ Not enough data. Try a different symbol.")
                return

            # Step 2: Prepare input
            last_60 = df[-60:].values
            scaled_input = scaler.transform(last_60)
            X_input = np.reshape(scaled_input, (1, scaled_input.shape[0], 1))

            future_predictions = []
            input_seq = X_input

            for _ in range(7):
                pred = model.predict(input_seq, verbose=0)
                future_predictions.append(pred[0][0])
                input_seq = np.append(input_seq[:, 1:, :], [[[pred[0][0]]]], axis=1)

            future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

            # Dates
            future_dates = [(datetime.today() + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(7)]
            results = pd.DataFrame({"Date": future_dates, "Predicted_Close": future_predictions.flatten()})

            # Show result
            st.success(f"✅ Forecast complete for {stock_symbol}")
            st.dataframe(results, use_container_width=True)

            # Plotting: Past 30 + Future 7
            past = df[-30:].copy()
            past.reset_index(inplace=True)
            past = past[["Date", "Close"]]
            past.columns = ["Date", "Actual_Close"]

            future = results.copy()
            future["Date"] = pd.to_datetime(future["Date"])
            combined = pd.merge(past, future, on="Date", how="outer")
            combined.set_index("Date", inplace=True)

            # Custom bar + line plot
            st.subheader("📊 Forecast Chart")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(combined["Actual_Close"], label="Past Close", marker='o')
            ax.plot(combined["Predicted_Close"], label="Predicted", linestyle='--', marker='x', color='orange')
            ax.set_title(f"{stock_symbol} - Last 30 Days & Next 7 Days Forecast")
            ax.set_ylabel("Price")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # Download button
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"forecast_{stock_symbol.replace('.', '_')}.csv",
                mime='text/csv'
            )

        except Exception as e:
            st.error(f"🚫 Error: {str(e)}")
