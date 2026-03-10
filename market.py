import yfinance as yf
import pandas as pd
import numpy as np

start_date = "2010-01-01"

tickers = {
    "DXY": "DX-Y.NYB",
    "BRENT": "BZ=F",
    "SP500": "^GSPC",
    "VIX": "^VIX",
    "US10Y": "^TNX"
}

def load_series(ticker, name):
    df = yf.download(ticker, start=start_date, interval="1d", progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"].iloc[:, 0]
    else:
        close = df["Close"]

    close.name = name
    return close

# 1. Load prices
series = []
for name, ticker in tickers.items():
    series.append(load_series(ticker, name))

market_df = pd.concat(series, axis=1)

# 2. Returns (SAFE)
price_cols = market_df.columns.tolist()
for col in price_cols:
    market_df[f"{col}_RET"] = np.log(
        market_df[col] / market_df[col].shift(1)
    )

# 3. Volatility
market_df["BRENT_VOL_10"] = market_df["BRENT_RET"].rolling(10).std()
market_df["SP500_VOL_10"] = market_df["SP500_RET"].rolling(10).std()

# 4. Cleanup
market_df = market_df.dropna()

market_df.to_csv("market_modality.csv", sep=";")
market_df.to_excel("market_modality.xlsx")

print("Market modality saved")
print(market_df.head())
print(market_df.tail())
print("Columns:", market_df.columns.tolist())
