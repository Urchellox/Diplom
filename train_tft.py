import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

SEP = ";"
WINDOW = 30
BATCH_SIZE = 64
EPOCHS = 15
LR = 0.001
PLOT_N = 150

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calc_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape


def add_lag_features(df: pd.DataFrame, col: str, lags=(1, 2, 3, 5, 7)):
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, col: str, windows=(3, 7)):
    for w in windows:
        df[f"{col}_ma{w}"] = df[col].rolling(w).mean()
        df[f"{col}_std{w}"] = df[col].rolling(w).std()
    return df


class SeqDataset(Dataset):
    def __init__(self, values, targets_delta, current_levels, actual_next, start, end, window):
        self.values = values
        self.targets_delta = targets_delta
        self.current_levels = current_levels
        self.actual_next = actual_next
        self.start = max(start, window)
        self.end = end
        self.window = window

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, i):
        idx = self.start + i
        x = self.values[idx - self.window:idx]
        y_delta = self.targets_delta[idx]
        current = self.current_levels[idx]
        actual = self.actual_next[idx]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y_delta, dtype=torch.float32),
            torch.tensor(current, dtype=torch.float32),
            torch.tensor(actual, dtype=torch.float32),
        )


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim * 2)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class GRN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None, dropout=0.1):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.glu = GLU(output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        residual = self.skip(x)
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.glu(x)
        return self.norm(x + residual)


class TFTStyle(nn.Module):
    def __init__(self, n_features, hidden=64, num_heads=4, dropout=0.1):
        super().__init__()

        self.hidden = hidden

        # Variable selection / feature transformation
        self.var_grn = GRN(n_features, hidden, n_features, dropout=dropout)
        self.var_weight = nn.Linear(n_features, n_features)

        # Temporal encoder
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        # After LSTM
        self.post_lstm_grn = GRN(hidden, hidden, hidden, dropout=dropout)

        # Temporal attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )

        # After attention
        self.post_attn_grn = GRN(hidden, hidden, hidden, dropout=dropout)

        # Output head
        self.output_grn = GRN(hidden, hidden, hidden, dropout=dropout)
        self.fc_out = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: [batch, seq_len, n_features]

        # 1) Variable selection
        x_proj = self.var_grn(x)
        weights = torch.softmax(self.var_weight(x_proj), dim=-1)
        x_selected = x * weights

        # 2) LSTM encoder
        h, _ = self.lstm(x_selected)
        h = self.post_lstm_grn(h)

        # 3) Self-attention
        attn_out, _ = self.attn(h, h, h)
        h = self.post_attn_grn(attn_out + h)

        # 4) Take last step
        last = h[:, -1, :]

        # 5) Output block
        last = self.output_grn(last)
        out = self.fc_out(last)

        return out.squeeze(-1)


def train_one_model(model, train_loader, test_loader, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for xb, yb_delta, _, _ in train_loader:
            xb = xb.to(device)
            yb_delta = yb_delta.to(device)

            optimizer.zero_grad()
            pred_delta = model(xb)
            loss = loss_fn(pred_delta, yb_delta)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []

        with torch.no_grad():
            for xb, yb_delta, _, _ in test_loader:
                xb = xb.to(device)
                yb_delta = yb_delta.to(device)

                pred_delta = model(xb)
                loss = loss_fn(pred_delta, yb_delta)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)

        print(f"Epoch {epoch+1}/{epochs} | train_loss={avg_train:.6f} | val_loss={avg_val:.6f}")

        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def predict_levels(model, loader):
    model.eval()

    pred_deltas = []
    actual_deltas = []
    current_levels = []
    actual_next_levels = []

    with torch.no_grad():
        for xb, yb_delta, current, actual_next in loader:
            xb = xb.to(device)

            pred_delta = model(xb).cpu().numpy()

            pred_deltas.extend(pred_delta)
            actual_deltas.extend(yb_delta.numpy())
            current_levels.extend(current.numpy())
            actual_next_levels.extend(actual_next.numpy())

    pred_deltas = np.array(pred_deltas)
    actual_deltas = np.array(actual_deltas)
    current_levels = np.array(current_levels)
    actual_next_levels = np.array(actual_next_levels)

    pred_next_levels = current_levels + pred_deltas
    naive_next_levels = current_levels.copy()

    return pred_next_levels, naive_next_levels, actual_next_levels, pred_deltas, actual_deltas


# =========================
# 1. Загрузка данных
# =========================
df = pd.read_csv("fx_core_dataset.csv", sep=SEP, index_col=0)
df.index = pd.to_datetime(df.index)
df.index.name = "Date"
df = df.reset_index()

# =========================
# 2. Target как дельта
# =========================
df["target_eur_delta"] = df["EURKZT"].shift(-1) - df["EURKZT"]
df["target_usd_delta"] = df["USDKZT"].shift(-1) - df["USDKZT"]

df["actual_eur_next"] = df["EURKZT"].shift(-1)
df["actual_usd_next"] = df["USDKZT"].shift(-1)

# =========================
# 3. Признаки
# =========================
base_cols = ["EURKZT", "USDKZT", "EURUSD", "DXY", "BRENT", "VIX"]

for col in base_cols:
    df = add_lag_features(df, col)
    df = add_rolling_features(df, col)

for col in base_cols:
    df[f"{col}_ret1"] = df[col].pct_change(1)
    df[f"{col}_ret3"] = df[col].pct_change(3)

df = df.dropna().reset_index(drop=True)

feature_cols = [
    c for c in df.columns
    if c not in [
        "Date",
        "target_eur_delta",
        "target_usd_delta",
        "actual_eur_next",
        "actual_usd_next",
    ]
]

# =========================
# 4. Split + scaling
# =========================
split = int(len(df) * 0.8)

scaler = StandardScaler()
scaler.fit(df.loc[:split - 1, feature_cols])

values = scaler.transform(df[feature_cols]).astype(np.float32)

eur_delta = df["target_eur_delta"].values.astype(np.float32)
usd_delta = df["target_usd_delta"].values.astype(np.float32)

eur_current = df["EURKZT"].values.astype(np.float32)
usd_current = df["USDKZT"].values.astype(np.float32)

eur_actual_next = df["actual_eur_next"].values.astype(np.float32)
usd_actual_next = df["actual_usd_next"].values.astype(np.float32)

# =========================
# 5. Dataset / loaders
# =========================
train_eur_ds = SeqDataset(values, eur_delta, eur_current, eur_actual_next, 0, split, WINDOW)
test_eur_ds = SeqDataset(values, eur_delta, eur_current, eur_actual_next, split, len(values) - 1, WINDOW)

train_usd_ds = SeqDataset(values, usd_delta, usd_current, usd_actual_next, 0, split, WINDOW)
test_usd_ds = SeqDataset(values, usd_delta, usd_current, usd_actual_next, split, len(values) - 1, WINDOW)

train_eur_loader = DataLoader(train_eur_ds, batch_size=BATCH_SIZE, shuffle=True)
test_eur_loader = DataLoader(test_eur_ds, batch_size=128, shuffle=False)

train_usd_loader = DataLoader(train_usd_ds, batch_size=BATCH_SIZE, shuffle=True)
test_usd_loader = DataLoader(test_usd_ds, batch_size=128, shuffle=False)

# =========================
# 6. Модели
# =========================
model_eur = TFTStyle(len(feature_cols)).to(device)
model_usd = TFTStyle(len(feature_cols)).to(device)

# =========================
# 7. Обучение
# =========================
print("\n=== Training TFT-style for EUR/KZT ===")
model_eur = train_one_model(model_eur, train_eur_loader, test_eur_loader, EPOCHS, LR)

print("\n=== Training TFT-style for USD/KZT ===")
model_usd = train_one_model(model_usd, train_usd_loader, test_usd_loader, EPOCHS, LR)

# =========================
# 8. Прогноз
# =========================
pred_eur_next, naive_eur_next, actual_eur_next_test, pred_eur_delta, true_eur_delta = predict_levels(model_eur, test_eur_loader)
pred_usd_next, naive_usd_next, actual_usd_next_test, pred_usd_delta, true_usd_delta = predict_levels(model_usd, test_usd_loader)

# Соответствующие даты теста
test_dates = df["Date"].iloc[max(split, WINDOW):len(df) - 1].reset_index(drop=True)

# =========================
# 9. Метрики
# =========================
mae_eur, rmse_eur, mape_eur = calc_metrics(actual_eur_next_test, pred_eur_next)
mae_usd, rmse_usd, mape_usd = calc_metrics(actual_usd_next_test, pred_usd_next)

mae_eur_naive, rmse_eur_naive, mape_eur_naive = calc_metrics(actual_eur_next_test, naive_eur_next)
mae_usd_naive, rmse_usd_naive, mape_usd_naive = calc_metrics(actual_usd_next_test, naive_usd_next)

print("\n=== TFT-style metrics: EUR/KZT ===")
print("MAE:", mae_eur)
print("RMSE:", rmse_eur)
print("MAPE:", mape_eur)

print("\n=== Naive baseline: EUR/KZT ===")
print("MAE:", mae_eur_naive)
print("RMSE:", rmse_eur_naive)
print("MAPE:", mape_eur_naive)

print("\n=== TFT-style metrics: USD/KZT ===")
print("MAE:", mae_usd)
print("RMSE:", rmse_usd)
print("MAPE:", mape_usd)

print("\n=== Naive baseline: USD/KZT ===")
print("MAE:", mae_usd_naive)
print("RMSE:", rmse_usd_naive)
print("MAPE:", mape_usd_naive)

# =========================
# 10. Сохраняем прогнозы
# =========================
results = pd.DataFrame({
    "Date": test_dates.values,
    "Actual_EURKZT_Next": actual_eur_next_test,
    "Predicted_EURKZT_Next": pred_eur_next,
    "Naive_EURKZT_Next": naive_eur_next,
    "Actual_USDKZT_Next": actual_usd_next_test,
    "Predicted_USDKZT_Next": pred_usd_next,
    "Naive_USDKZT_Next": naive_usd_next,
    "Predicted_EUR_Delta": pred_eur_delta,
    "Predicted_USD_Delta": pred_usd_delta,
})
results.to_csv("tft_predictions.csv", sep=SEP, index=False)

# =========================
# 11. График EUR/KZT
# =========================
plot_dates = test_dates.iloc[-PLOT_N:]

plt.figure(figsize=(10, 5))
plt.plot(plot_dates, actual_eur_next_test[-PLOT_N:], label="Actual EUR/KZT")
plt.plot(plot_dates, pred_eur_next[-PLOT_N:], label="Predicted EUR/KZT")
plt.plot(plot_dates, naive_eur_next[-PLOT_N:], label="Naive EUR/KZT", linestyle="--")
plt.title("TFT-style Forecast: EUR/KZT")
plt.xlabel("Date")
plt.ylabel("EUR/KZT")
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("tft_eurkzt_prediction.png")
plt.close()

# =========================
# 12. График USD/KZT
# =========================
plt.figure(figsize=(10, 5))
plt.plot(plot_dates, actual_usd_next_test[-PLOT_N:], label="Actual USD/KZT")
plt.plot(plot_dates, pred_usd_next[-PLOT_N:], label="Predicted USD/KZT")
plt.plot(plot_dates, naive_usd_next[-PLOT_N:], label="Naive USD/KZT", linestyle="--")
plt.title("TFT-style Forecast: USD/KZT")
plt.xlabel("Date")
plt.ylabel("USD/KZT")
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("tft_usdkzt_prediction.png")
plt.close()

print("\nСохранены файлы:")
print("- tft_eurkzt_prediction.png")
print("- tft_usdkzt_prediction.png")
print("- tft_predictions.csv")