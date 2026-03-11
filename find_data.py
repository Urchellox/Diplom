import yfinance as yf
import pandas as pd
import numpy as np

start_date = "2010-01-01"

def get_close_series(ticker: str, start: str) -> pd.Series:
    """
    Надежно вытаскивает Close как Series, даже если yfinance вернул MultiIndex.
    """
    df = yf.download(ticker, start=start, interval="1d", progress=False)

    # Если колонки MultiIndex: ('Close', 'EURUSD=X')
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"]  # это будет DataFrame с одним столбцом (ticker)
        # берем первый столбец как Series
        close = close.iloc[:, 0]
    else:
        # Обычный случай: 'Close' — это Series
        close = df["Close"]

    close = close.rename(ticker)  # имя пока тикер, позже переименуем
    return close

# 1) Загрузка Close как Series
eurusd = get_close_series("EURUSD=X", start_date).rename("EURUSD")
usdkzt = get_close_series("USDKZT=X", start_date).rename("USDKZT")

# 2) Объединение
df = pd.concat([eurusd, usdkzt], axis=1)

# На всякий случай — контроль
print("Колонки после concat:", df.columns.tolist())
print(df.head())

# 3) Кросс-валютные признаки
df["EURKZT"] = df["EURUSD"] * df["USDKZT"]
df["CROSS_RATIO"] = df["EURUSD"] / df["USDKZT"]

# 4) Доходности
df["EURUSD_RET"] = np.log(df["EURUSD"] / df["EURUSD"].shift(1))
df["USDKZT_RET"] = np.log(df["USDKZT"] / df["USDKZT"].shift(1))
df["EURKZT_RET"] = np.log(df["EURKZT"] / df["EURKZT"].shift(1))

# 5) Технические индикаторы

# SMA
df["SMA_5"] = df["EURKZT"].rolling(5).mean()
df["SMA_10"] = df["EURKZT"].rolling(10).mean()
df["SMA_20"] = df["EURKZT"].rolling(20).mean()

# EMA
df["EMA_5"] = df["EURKZT"].ewm(span=5, adjust=False).mean()
df["EMA_10"] = df["EURKZT"].ewm(span=10, adjust=False).mean()
df["EMA_20"] = df["EURKZT"].ewm(span=20, adjust=False).mean()

# RSI (14)
delta = df["EURKZT"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss
df["RSI_14"] = 100 - (100 / (1 + rs))

# MACD
ema_12 = df["EURKZT"].ewm(span=12, adjust=False).mean()
ema_26 = df["EURKZT"].ewm(span=26, adjust=False).mean()
df["MACD"] = ema_12 - ema_26
df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()

# Volatility
df["VOL_10"] = df["EURKZT_RET"].rolling(10).std()
df["VOL_20"] = df["EURKZT_RET"].rolling(20).std()

# 6) Очистка и сохранение
df = df.dropna()

df.to_csv("fx_multimodal_dataset_2010_present.csv", sep=";")
df.to_excel("fx_multimodal_dataset_2010_present.xlsx")

print(f"Датасет сохранён")
print(df.head())
print(df.tail())
print(f"Колонок: {df.shape[1]}")
