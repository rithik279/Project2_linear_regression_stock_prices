import yfinance as yf
import pandas as pd

def import_fcn():
    # Define tickers
    tickers = ["AAPL", "AMZN", "MSFT", "QQQ", "^GSPC"]
    
    # Download Close prices
    df = yf.download(tickers, start='2020-01-01', end='2025-07-30')["Close"]
    
    # Create lag features (t-1)
    for ticker in tickers:
        df[f"{ticker}(t-1)"] = df[ticker].shift(1)

    # Create 5-day moving averages
    for ticker in tickers:
        df[f"{ticker}_MA_5"] = df[ticker].rolling(window=5).mean()

    # Create prediction target: next-day AAPL closing price
    df['Target'] = df['AAPL'].shift(-1)

    # Drop rows with NaN values from shifting/rolling
    df = df.dropna()

    return df
