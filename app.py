import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load LSTM model
model = load_model("stock_model.h5")

# Streamlit Page Config
st.set_page_config(page_title="📊 Stock Forecast", layout="wide")
st.title("🚀 Stock Market Trend Forecasting (AI Based)")
st.markdown("Built with **LSTM Neural Network** and real-time **Yahoo Finance** data 📈")

# Sidebar
st.sidebar.title("📈 Stock Forecast Dashboard")
tab = st.sidebar.radio("Select Option", ["📈 Forecast", "📊 Compare Stocks", "🤖 AI-Based Suggestion"])

# Forecast Function
def forecast_stock(symbol):
    try:
        df = yf.download(symbol, period="90d", interval="1d")
        if df.empty or len(df) < 60:
            return None, "Insufficient data for forecasting."

        close_data = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close_data)

        X = [scaled_data[-60:]]
        X = np.array(X).reshape((1, 60, 1))

        prediction = model.predict(X)
        predicted_price = float(scaler.inverse_transform(prediction)[0][0])
        current_price = float(df['Close'].iloc[-1])
        trend = "📈 Up" if predicted_price > current_price else "📉 Down"

        return {
            "predicted_price": round(predicted_price, 2),
            "current_price": round(current_price, 2),
            "trend": trend
        }, None
    except Exception as e:
        return None, str(e)

# 📈 Forecast Tab
if tab == "📈 Forecast":
    st.markdown("## 📈 Forecast Summary")
    with st.form("forecast_form"):
        symbol = st.text_input("Enter Stock Symbol (e.g., TATAMOTORS.NS or AAPL)")
        submitted = st.form_submit_button("🔍 Predict")
    if submitted and symbol:
        with st.spinner("🔄 Fetching data and predicting, please wait..."):
            result, error = forecast_stock(symbol.upper())
        if error:
            st.error(f"❌ Error: {error}")
        else:
            st.success("✅ Forecast generated successfully!")
            st.markdown(f"""
                <div style='background-color:#f0f2f6;padding:20px;border-radius:10px'>
                    <h3>🔹 Predicted Price: ₹{result['predicted_price']}</h3>
                    <p>📌 Current Price: ₹{result['current_price']}</p>
                    <p>📊 Trend: {result['trend']}</p>
                </div>
            """, unsafe_allow_html=True)

# 📊 Compare Stocks Tab
elif tab == "📊 Compare Stocks":
    st.markdown("## 📊 Compare Two Stocks")
    with st.form("compare_form"):
        col1, col2 = st.columns(2)
        with col1:
            stock1 = st.text_input("Enter Stock Symbol A", value="TCS.NS")
        with col2:
            stock2 = st.text_input("Enter Stock Symbol B", value="INFY.NS")
        compare_btn = st.form_submit_button("📈 Compare")
    if compare_btn and stock1 and stock2:
        try:
            df1 = yf.download(stock1, period="3mo")["Close"]
            df2 = yf.download(stock2, period="3mo")["Close"]
            df = pd.concat([df1, df2], axis=1)
            df.columns = [stock1.upper(), stock2.upper()]
            df.dropna(inplace=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=df[stock1.upper()],
                                     mode="lines+markers", name=stock1.upper()))
            fig.add_trace(go.Scatter(x=df.index, y=df[stock2.upper()],
                                     mode="lines+markers", name=stock2.upper()))
            fig.update_layout(title="Stock Comparison",
                              xaxis_title="Date", yaxis_title="Price",
                              template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error: {e}")

# 🤖 AI-Based Suggestion Tab
elif tab == "🤖 AI-Based Suggestion":
    st.markdown("## 🤖 AI-Based Investment Suggestion")
    with st.form("ai_form"):
        symbol = st.text_input("Enter Stock Symbol for Suggestion", value="RELIANCE.NS")
        suggest_btn = st.form_submit_button("💡 AI Suggest")
    if suggest_btn and symbol:
        with st.spinner("🧠 Analyzing market and generating suggestion..."):
            result, error = forecast_stock(symbol.upper())
        if error:
            st.error(f"❌ Error: {error}")
        else:
            current = result['current_price']
            predicted = result['predicted_price']
            percent = round(((predicted - current) / current) * 100, 2)

            if percent > 0.5:
                decision = "✅ BUY"
                color = "green"
            elif percent < -0.5:
                decision = "❌ SELL"
                color = "red"
            else:
                decision = "⚠️ HOLD"
                color = "orange"

            st.markdown(f"""
                <div style='background-color:#fefefe;padding:20px;border-radius:10px'>
                    <p>📌 Current Price: ₹{current}</p>
                    <p>🔮 Predicted Price: ₹{predicted}</p>
                    <p>📊 Change: {percent}%</p>
                    <h3>💡 Suggestion: <span style='color:{color}'>{decision}</span></h3>
                </div>
            """, unsafe_allow_html=True)

# ✅ Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: grey;'>
    🔒 Powered by LSTM Neural Network | Made with ❤️ by <b>Sameer More</b>
</div>
""", unsafe_allow_html=True)
