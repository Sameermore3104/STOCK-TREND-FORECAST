import yfinance as yf
import pandas as pd

def get_stock_data(ticker='^NSEI', period='3mo', interval='1d'):
    """
    Fetch historical stock data from Yahoo Finance.
    Default: Nifty50, past 3 months, daily data.
    """
    data = yf.download(ticker, period=period, interval=interval)
    data.dropna(inplace=True)
    return data

if __name__ == "__main__":
    df = get_stock_data()
    print(df.head())
