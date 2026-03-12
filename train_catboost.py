import math
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

SEP = ";"
PLOT_N = 150


def add_lag_features(df: pd.DataFrame, col: str, lags=(1, 2, 3, 5, 7)):
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, col: str, windows=(3, 7)):
    for w in windows:
        df[f"{col}_ma{w}"] = df[col].rolling(w).mean()
        df[f"{col}_std{w}"] = df[col].rolling(w).std()
    return df


def calc_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape


# =========================
# 1. Загрузка данных
# =========================
df = pd.read_csv("fx_core_dataset.csv", sep=SEP, index_col=0)
df.index = pd.to_datetime(df.index)
df.index.name = "Date"
df = df.reset_index()

# =========================
# 2. Target как изменение курса
# =========================
df["target_eur_delta"] = df["EURKZT"].shift(-1) - df["EURKZT"]
df["target_usd_delta"] = df["USDKZT"].shift(-1) - df["USDKZT"]

# Для оценки уровня завтра сохраним реальные next values
df["actual_eur_next"] = df["EURKZT"].shift(-1)
df["actual_usd_next"] = df["USDKZT"].shift(-1)

# =========================
# 3. Признаки
# =========================
base_cols = ["EURKZT", "USDKZT", "EURUSD", "DXY", "BRENT", "VIX"]

for col in base_cols:
    df = add_lag_features(df, col, lags=(1, 2, 3, 5, 7))
    df = add_rolling_features(df, col, windows=(3, 7))

for col in base_cols:
    df[f"{col}_ret1"] = df[col].pct_change(1)
    df[f"{col}_ret3"] = df[col].pct_change(3)

df = df.dropna().reset_index(drop=True)

# =========================
# 4. Список признаков
# =========================
exclude_cols = [
    "Date",
    "target_eur_delta",
    "target_usd_delta",
    "actual_eur_next",
    "actual_usd_next",
]

features_eur = [c for c in df.columns if c not in exclude_cols]
features_usd = [c for c in df.columns if c not in exclude_cols]

# =========================
# 5. Разделение train/test
# =========================
split = int(len(df) * 0.8)

train = df.iloc[:split].copy()
test = df.iloc[split:].copy()

# EUR model
X_train_eur = train[features_eur]
y_train_eur = train["target_eur_delta"]

X_test_eur = test[features_eur]
y_test_eur_delta = test["target_eur_delta"]

# USD model
X_train_usd = train[features_usd]
y_train_usd = train["target_usd_delta"]

X_test_usd = test[features_usd]
y_test_usd_delta = test["target_usd_delta"]

# Текущий уровень курса на момент t
current_eur_test = test["EURKZT"].values
current_usd_test = test["USDKZT"].values

# Реальный уровень курса на t+1
actual_eur_next = test["actual_eur_next"].values
actual_usd_next = test["actual_usd_next"].values

# =========================
# 6. Модели
# =========================
model_eur = CatBoostRegressor(
    iterations=800,
    depth=6,
    learning_rate=0.03,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=100
)

model_usd = CatBoostRegressor(
    iterations=800,
    depth=6,
    learning_rate=0.03,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=100
)

# =========================
# 7. Обучение
# =========================
model_eur.fit(
    X_train_eur,
    y_train_eur,
    eval_set=(X_test_eur, y_test_eur_delta),
    use_best_model=True
)

model_usd.fit(
    X_train_usd,
    y_train_usd,
    eval_set=(X_test_usd, y_test_usd_delta),
    use_best_model=True
)

# =========================
# 8. Прогноз дельты
# =========================
pred_eur_delta = model_eur.predict(X_test_eur)
pred_usd_delta = model_usd.predict(X_test_usd)

# Восстановление прогноза уровня
pred_eur_next = current_eur_test + pred_eur_delta
pred_usd_next = current_usd_test + pred_usd_delta

# =========================
# 9. Naive baseline
# =========================
# Простейший прогноз: завтра = сегодня
naive_eur_next = current_eur_test.copy()
naive_usd_next = current_usd_test.copy()

# =========================
# 10. Метрики
# =========================
mae_eur, rmse_eur, mape_eur = calc_metrics(actual_eur_next, pred_eur_next)
mae_usd, rmse_usd, mape_usd = calc_metrics(actual_usd_next, pred_usd_next)

mae_eur_naive, rmse_eur_naive, mape_eur_naive = calc_metrics(actual_eur_next, naive_eur_next)
mae_usd_naive, rmse_usd_naive, mape_usd_naive = calc_metrics(actual_usd_next, naive_usd_next)

print("=== CatBoost metrics: EUR/KZT ===")
print("MAE:", mae_eur)
print("RMSE:", rmse_eur)
print("MAPE:", mape_eur)

print("\n=== Naive baseline: EUR/KZT ===")
print("MAE:", mae_eur_naive)
print("RMSE:", rmse_eur_naive)
print("MAPE:", mape_eur_naive)

print("\n=== CatBoost metrics: USD/KZT ===")
print("MAE:", mae_usd)
print("RMSE:", rmse_usd)
print("MAPE:", mape_usd)

print("\n=== Naive baseline: USD/KZT ===")
print("MAE:", mae_usd_naive)
print("RMSE:", rmse_usd_naive)
print("MAPE:", mape_usd_naive)

# =========================
# 11. Сохранение прогнозов
# =========================
results = pd.DataFrame({
    "Date": test["Date"].values,
    "Current_EURKZT": current_eur_test,
    "Actual_EURKZT_Next": actual_eur_next,
    "Predicted_EURKZT_Next": pred_eur_next,
    "Naive_EURKZT_Next": naive_eur_next,
    "Current_USDKZT": current_usd_test,
    "Actual_USDKZT_Next": actual_usd_next,
    "Predicted_USDKZT_Next": pred_usd_next,
    "Naive_USDKZT_Next": naive_usd_next,
    "Predicted_EUR_Delta": pred_eur_delta,
    "Predicted_USD_Delta": pred_usd_delta,
})
results.to_csv("catboost_predictions.csv", sep=SEP, index=False)

# =========================
# 12. График EUR/KZT
# =========================
plot_dates = test["Date"].iloc[-PLOT_N:]

plt.figure(figsize=(10, 5))
plt.plot(plot_dates, actual_eur_next[-PLOT_N:], label="Actual EUR/KZT")
plt.plot(plot_dates, pred_eur_next[-PLOT_N:], label="Predicted EUR/KZT")
plt.plot(plot_dates, naive_eur_next[-PLOT_N:], label="Naive EUR/KZT", linestyle="--")
plt.title("CatBoost Forecast: EUR/KZT")
plt.xlabel("Date")
plt.ylabel("EUR/KZT")
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("catboost_eurkzt_prediction.png")
plt.close()

# =========================
# 13. График USD/KZT
# =========================
plt.figure(figsize=(10, 5))
plt.plot(plot_dates, actual_usd_next[-PLOT_N:], label="Actual USD/KZT")
plt.plot(plot_dates, pred_usd_next[-PLOT_N:], label="Predicted USD/KZT")
plt.plot(plot_dates, naive_usd_next[-PLOT_N:], label="Naive USD/KZT", linestyle="--")
plt.title("CatBoost Forecast: USD/KZT")
plt.xlabel("Date")
plt.ylabel("USD/KZT")
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("catboost_usdkzt_prediction.png")
plt.close()

# =========================
# 14. Feature importance
# =========================
importance_eur = pd.DataFrame({
    "feature": features_eur,
    "importance": model_eur.get_feature_importance()
}).sort_values("importance", ascending=False)

importance_usd = pd.DataFrame({
    "feature": features_usd,
    "importance": model_usd.get_feature_importance()
}).sort_values("importance", ascending=False)

importance_eur.to_csv("feature_importance_eurkzt.csv", sep=SEP, index=False)
importance_usd.to_csv("feature_importance_usdkzt.csv", sep=SEP, index=False)

print("\nTop-10 important features for EUR/KZT:")
print(importance_eur.head(10))

print("\nTop-10 important features for USD/KZT:")
print(importance_usd.head(10))

print("\nСохранены файлы:")
print("- catboost_eurkzt_prediction.png")
print("- catboost_usdkzt_prediction.png")
print("- catboost_predictions.csv")
print("- feature_importance_eurkzt.csv")
print("- feature_importance_usdkzt.csv")