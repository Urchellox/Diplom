import math
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv("fx_multimodal_dataset_2010_present.csv", sep=";")
df["Date"] = pd.to_datetime(df["Date"])

# target
df["target_next"] = df["EURKZT"].shift(-1)
df = df.dropna().reset_index(drop=True)

features = [c for c in df.columns if c not in ["Date", "target_next"]]

split = int(len(df) * 0.8)

train = df.iloc[:split]
test = df.iloc[split:]

X_train = train[features]
y_train = train["target_next"]

X_test = test[features]
y_test = test["target_next"]

model = CatBoostRegressor(
    iterations=300,
    depth=6,
    learning_rate=0.03,
    loss_function="RMSE",
    random_seed=42,
    verbose=False
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = math.sqrt(mean_squared_error(y_test, pred))
mape = np.mean(np.abs((y_test - pred) / y_test)) * 100

print("CatBoost metrics")
print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE:", mape)

plt.figure(figsize=(10,5))
plt.plot(test["Date"][:150], y_test[:150], label="Actual")
plt.plot(test["Date"][:150], pred[:150], label="Predicted")
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig("catboost_prediction.png")