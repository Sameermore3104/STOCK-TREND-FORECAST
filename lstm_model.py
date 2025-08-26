import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib

# Step 1: Load data
def load_data():
    df = yf.download('^NSEI', period='6mo', interval='1d')
    df = df[['Close']]
    df.dropna(inplace=True)
    return df

# Step 2: Prepare data
def prepare_data(df, window_size=60):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df)

    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# Step 3: Build and train LSTM model
def train_model(X, y):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=20, batch_size=32)
    return model


# Step 4: Save model and scaler
def save_model(model, scaler):
    model.save("stock_model.h5", include_optimizer=False)
    joblib.dump(scaler, "scaler.save")

# Step 5: Run all
if __name__ == "__main__":
    df = load_data()
    X, y, scaler = prepare_data(df)
    model = train_model(X, y)
    save_model(model, scaler)
    print("✅ Model trained and saved!")
