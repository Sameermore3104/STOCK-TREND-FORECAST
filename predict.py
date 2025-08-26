import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime, timedelta
import sys

# Get stock symbol from command line
if len(sys.argv) > 1:
    symbol = sys.argv[1]
else:
    symbol = "^NSEI"  # default NIFTY

# Load model and scaler
model = load_model('stock_model.h5')
scaler = joblib.load('scaler.save')

# Load last 100 days data
data = yf.download(symbol, period='100d', interval='1d')
data = data[['Close']]
data.dropna(inplace=True)

# Scale and reshape data
scaled_data = scaler.transform(data)
last_60 = scaled_data[-60:]
X_input = np.array(last_60).reshape(1, 60, 1)

# Predict next 7 days
predicted_prices = []
for _ in range(7):
    pred = model.predict(X_input)[0][0]
    predicted_prices.append(pred)
    X_input = np.append(X_input[:, 1:, :], [[[pred]]], axis=1)

# Inverse scale the predicted prices
predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1)).flatten()

# Prepare results
future_dates = [(datetime.today() + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(7)]
pred_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': predicted_prices})

# Save with stock name
output_file = f"future_predictions_{symbol.replace('.', '_')}.csv"
pred_df.to_csv(output_file, index=False)
print(f"✅ Saved prediction to {output_file}")
print(pred_df)
