import math
import copy
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

SEP = ";"
SEED = 42
WINDOW = 90
BATCH_SIZE = 64
EPOCHS = 60
LR = 1e-3
WEIGHT_DECAY = 1e-5
HIDDEN_SIZE = 64
DROPOUT = 0.15
PLOT_N = 150
TARGET_QUANTILES = [0.1, 0.5, 0.9]
EARLY_STOPPING_PATIENCE = 12

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.style.use("seaborn-v0_8-whitegrid")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calc_metrics_percent(y_true, y_pred):
    """
    Возвращает метрики в процентах:
    - MAE_%  = MAE / mean(y_true) * 100
    - RMSE_% = RMSE / mean(y_true) * 100
    - MAPE_% = mean(abs((y_true - y_pred) / y_true)) * 100
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mean_true = np.mean(y_true)

    mae_pct = (mae / mean_true) * 100 if mean_true != 0 else np.nan
    rmse_pct = (rmse / mean_true) * 100 if mean_true != 0 else np.nan
    mape_pct = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)

    return mae_pct, rmse_pct, mape_pct, r2


def add_lag_features(df: pd.DataFrame, col: str, lags=(1, 2, 3, 5, 7, 14)):
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, col: str, windows=(3, 7, 14)):
    for w in windows:
        roll = df[col].rolling(w)
        df[f"{col}_ma{w}"] = roll.mean()
        df[f"{col}_std{w}"] = roll.std()
    return df


def plot_forecast(dates, actual, predicted, naive, title, ylabel, filename, n_last=150, metrics_text=None):
    dates = dates.iloc[-n_last:]
    actual = actual[-n_last:]
    predicted = predicted[-n_last:]
    naive = naive[-n_last:]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)

    ax.plot(
        dates, actual,
        label="Actual",
        linewidth=2.6,
        color="#1f3b73"
    )
    ax.plot(
        dates, predicted,
        label="TFT forecast",
        linewidth=2.2,
        color="#d62728"
    )
    ax.plot(
        dates, naive,
        label="Naive baseline",
        linewidth=1.8,
        linestyle="--",
        color="#7f7f7f"
    )

    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(loc="best", frameon=True, fancybox=True, shadow=False, fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if metrics_text is not None:
        ax.text(
            0.98, 0.02, metrics_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="gray")
        )

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

def plot_r2_progress(actual, predicted, filename):
    r2_vals = []
    steps = []

    for i in range(30, len(actual)):
        r2 = r2_score(actual[:i], predicted[:i])
        r2_vals.append(r2)
        steps.append(i)

    plt.figure(figsize=(10, 5))
    plt.plot(steps, r2_vals)
    plt.title("R2 Score Progress (TFT)")
    plt.xlabel("Samples")
    plt.ylabel("R2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


class OneStepDataset(Dataset):
    def __init__(self, values, target, current_level, actual_next, start_idx, end_idx, window):
        self.values = values
        self.target = target
        self.current_level = current_level
        self.actual_next = actual_next
        self.window = window

        self.indices = np.arange(max(start_idx, window - 1), end_idx)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]

        x = self.values[idx - self.window + 1: idx + 1]
        y = self.target[idx]
        current = self.current_level[idx]
        actual = self.actual_next[idx]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
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


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None, dropout=0.1, context_dim=None):
        super().__init__()
        output_dim = output_dim or input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.glu = GLU(output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x, context=None):
        residual = self.skip(x)
        out = self.fc1(x)
        if self.context_fc is not None and context is not None:
            out = out + self.context_fc(context)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.glu(out)
        return self.norm(out + residual)


class VariableSelectionNetwork(nn.Module):
    def __init__(self, num_features, hidden_size, dropout):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout=dropout)
            for _ in range(num_features)
        ])
        self.flattened_grn = GatedResidualNetwork(
            num_features * hidden_size, hidden_size, num_features, dropout=dropout
        )
        self.input_projections = nn.ModuleList([
            nn.Linear(1, hidden_size) for _ in range(num_features)
        ])

    def forward(self, x):
        # x: [B, T, F]
        transformed = []
        for i in range(self.num_features):
            feat = x[:, :, i:i+1]
            emb = self.input_projections[i](feat)
            transformed.append(self.feature_grns[i](emb))
        transformed = torch.stack(transformed, dim=-2)  # [B, T, F, H]
        flat = transformed.reshape(transformed.size(0), transformed.size(1), -1)
        weights = torch.softmax(self.flattened_grn(flat), dim=-1)  # [B, T, F]
        combined = (transformed * weights.unsqueeze(-1)).sum(dim=-2)  # [B, T, H]
        return combined, weights


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_layers = nn.ModuleList([nn.Linear(hidden_size, self.head_dim) for _ in range(num_heads)])
        self.k_layers = nn.ModuleList([nn.Linear(hidden_size, self.head_dim) for _ in range(num_heads)])
        self.v_layer = nn.Linear(hidden_size, self.head_dim)
        self.out_proj = nn.Linear(self.head_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, T, H = x.shape
        shared_v = self.v_layer(x)
        heads = []
        attn_weights = []
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)

        for ql, kl in zip(self.q_layers, self.k_layers):
            q = ql(x)
            k = kl(x)
            scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            scores = scores.masked_fill(causal_mask, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            weights = self.dropout(weights)
            head = torch.matmul(weights, shared_v)
            heads.append(head)
            attn_weights.append(weights)

        mean_head = torch.stack(heads, dim=0).mean(dim=0)  # [B, T, D]
        out = self.out_proj(mean_head)
        weights = torch.stack(attn_weights, dim=0).mean(dim=0)
        return out, weights


class TemporalFusionTransformer(nn.Module):
    def __init__(self, num_features, hidden_size=64, lstm_layers=1, num_heads=4, dropout=0.1, quantiles=None):
        super().__init__()
        self.quantiles = quantiles or [0.1, 0.5, 0.9]

        self.vsn = VariableSelectionNetwork(num_features, hidden_size, dropout)
        self.input_gate = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout=dropout)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.post_lstm_gate = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout=dropout)
        self.attention = InterpretableMultiHeadAttention(hidden_size, num_heads=num_heads, dropout=dropout)
        self.post_attn_gate = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout=dropout)
        self.positionwise_grn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout=dropout)
        self.output_layer = nn.Linear(hidden_size, len(self.quantiles))

    def forward(self, x):
        selected, weights = self.vsn(x)
        selected = self.input_gate(selected)
        lstm_out, _ = self.lstm(selected)
        temporal = self.post_lstm_gate(lstm_out)
        attn_out, attn_weights = self.attention(temporal)
        temporal = self.post_attn_gate(temporal + attn_out)
        decoded = self.positionwise_grn(temporal[:, -1, :])
        out = self.output_layer(decoded)
        return out, weights, attn_weights


class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        target = target.unsqueeze(-1)
        errors = target - preds
        losses = []
        for i, q in enumerate(self.quantiles):
            e = errors[:, i]
            losses.append(torch.maximum((q - 1) * e, q * e).unsqueeze(-1))
        return torch.mean(torch.sum(torch.cat(losses, dim=-1), dim=-1))


@dataclass
class SplitConfig:
    train_end: int
    val_end: int
    test_end: int


def make_splits(n_rows: int) -> SplitConfig:
    train_end = int(n_rows * 0.7)
    val_end = int(n_rows * 0.85)
    return SplitConfig(train_end=train_end, val_end=val_end, test_end=n_rows - 1)


def train_one_model(model, train_loader, val_loader, epochs, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=4, factor=0.5)
    loss_fn = QuantileLoss(TARGET_QUANTILES)

    best_state = None
    best_val = float("inf")
    wait = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for xb, yb, _, _ in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred_q, _, _ = model(xb)
            loss = loss_fn(pred_q, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb, _, _ in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred_q, _, _ = model(xb)
                loss = loss_fn(pred_q, yb)
                val_losses.append(loss.item())

        avg_train = float(np.mean(train_losses))
        avg_val = float(np.mean(val_losses))
        scheduler.step(avg_val)

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"train_loss={avg_train:.6f} | val_loss={avg_val:.6f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6g}"
        )

        if avg_val < best_val:
            best_val = avg_val
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_levels(model, loader):
    model.eval()
    pred_delta = []
    current_levels = []
    actual_next_levels = []

    with torch.no_grad():
        for xb, _, current, actual_next in loader:
            xb = xb.to(device)
            pred_q, _, _ = model(xb)
            pred_median = pred_q[:, 1].cpu().numpy()

            pred_delta.extend(pred_median)
            current_levels.extend(current.numpy())
            actual_next_levels.extend(actual_next.numpy())

    pred_delta = np.array(pred_delta)
    current_levels = np.array(current_levels)
    actual_next_levels = np.array(actual_next_levels)

    pred_next_levels = current_levels + pred_delta
    naive_next_levels = current_levels.copy()
    return pred_next_levels, naive_next_levels, actual_next_levels, pred_delta


def build_feature_frame(df: pd.DataFrame):
    base_cols = ["EURKZT", "USDKZT", "EURUSD", "DXY", "BRENT", "VIX"]

    for col in base_cols:
        df = add_lag_features(df, col)
        df = add_rolling_features(df, col)

    for col in base_cols:
        df[f"{col}_ret1"] = df[col].pct_change(1)
        df[f"{col}_ret3"] = df[col].pct_change(3)
        df[f"{col}_ret5"] = df[col].pct_change(5)

    # calendar features
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["is_month_start"] = df["Date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["Date"].dt.is_month_end.astype(int)

    # cyclical calendar encoding
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["target_eur_delta"] = df["EURKZT"].shift(-3) - df["EURKZT"]
    df["target_usd_delta"] = df["USDKZT"].shift(-3) - df["USDKZT"]

    df["actual_eur_next"] = df["EURKZT"].shift(-3)
    df["actual_usd_next"] = df["USDKZT"].shift(-3)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    feature_cols = [
        c for c in df.columns
        if c not in {
            "Date", "target_eur_delta", "target_usd_delta", "actual_eur_next", "actual_usd_next"
        }
    ]
    return df, feature_cols


def make_loaders(values, target, current, actual_next, split_cfg):
    train_ds = OneStepDataset(values, target, current, actual_next, 0, split_cfg.train_end, WINDOW)
    val_ds = OneStepDataset(values, target, current, actual_next, split_cfg.train_end, split_cfg.val_end, WINDOW)
    test_ds = OneStepDataset(values, target, current, actual_next, split_cfg.val_end, split_cfg.test_end, WINDOW)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    return train_loader, val_loader, test_loader


def run_target(df, feature_cols, target_col, current_col, actual_next_col, label_prefix):
    split_cfg = make_splits(len(df))
    scaler = StandardScaler()
    scaler.fit(df.loc[: split_cfg.train_end - 1, feature_cols])

    values = scaler.transform(df[feature_cols]).astype(np.float32)
    target = df[target_col].values.astype(np.float32)
    current = df[current_col].values.astype(np.float32)
    actual_next = df[actual_next_col].values.astype(np.float32)

    train_loader, val_loader, test_loader = make_loaders(values, target, current, actual_next, split_cfg)

    model = TemporalFusionTransformer(
        num_features=len(feature_cols),
        hidden_size=HIDDEN_SIZE,
        lstm_layers=1,
        num_heads=8,
        dropout=DROPOUT,
        quantiles=TARGET_QUANTILES,
    ).to(device)

    print(f"\n=== Training canonical-ish TFT for {label_prefix} ===")
    model = train_one_model(model, train_loader, val_loader, EPOCHS, LR)

    pred_next, naive_next, actual_next_test, pred_logret = predict_levels(model, test_loader)

    mae_pct, rmse_pct, mape_pct, r2 = calc_metrics_percent(actual_next_test, pred_next)
    mae_n_pct, rmse_n_pct, mape_n_pct, r2_naive = calc_metrics_percent(actual_next_test, naive_next)
    
    test_indices = test_loader.dataset.indices
    test_dates = df["Date"].iloc[test_indices].reset_index(drop=True)

    print(f"\n=== TFT metrics (%): {label_prefix} ===")
    print("MAE (%):", mae_pct)
    print("RMSE (%):", rmse_pct)
    print("MAPE (%):", mape_pct)
    print("R2:", r2)

    print(f"\n=== Naive baseline (%): {label_prefix} ===")
    print("MAE (%):", mae_n_pct)
    print("RMSE (%):", rmse_n_pct)
    print("MAPE (%):", mape_n_pct)
    print("Naive R2:", r2_naive)

    return {
        "model": model,
        "pred_next": pred_next,
        "naive_next": naive_next,
        "actual_next": actual_next_test,
        "pred_logret": pred_logret,
        "test_dates": test_dates,
        "metrics": (mae_pct, rmse_pct, mape_pct, r2),
        "naive_metrics": (mae_n_pct, rmse_n_pct, mape_n_pct, r2_naive),
    }


def main():
    set_seed(SEED)

    df = pd.read_csv("fx_core_dataset.csv", sep=SEP, index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.reset_index()

    df, feature_cols = build_feature_frame(df)

    eur_res = run_target(df, feature_cols, "target_eur_delta", "EURKZT", "actual_eur_next", "EUR/KZT")
    usd_res = run_target(df, feature_cols, "target_usd_delta", "USDKZT", "actual_usd_next", "USD/KZT")

    # =========================
    # 1. Сохранение прогнозов
    # =========================
    results = pd.DataFrame({
        "Date": eur_res["test_dates"].values,
        "Actual_EURKZT_Next": eur_res["actual_next"],
        "Predicted_EURKZT_Next": eur_res["pred_next"],
        "Naive_EURKZT_Next": eur_res["naive_next"],
        "Actual_USDKZT_Next": usd_res["actual_next"],
        "Predicted_USDKZT_Next": usd_res["pred_next"],
        "Naive_USDKZT_Next": usd_res["naive_next"],
        "Predicted_EUR_LogRet": eur_res["pred_logret"],
        "Predicted_USD_LogRet": usd_res["pred_logret"],
    })
    results.to_csv("tft_predictions_real.csv", sep=SEP, index=False)

    # =========================
    # 2. Красивые графики
    # =========================
    eur_mae, eur_rmse, eur_mape, eur_r2 = eur_res["metrics"]
    usd_mae, usd_rmse, usd_mape, usd_r2 = usd_res["metrics"]

    plot_forecast(
        dates=eur_res["test_dates"],
        actual=eur_res["actual_next"],
        predicted=eur_res["pred_next"],
        naive=eur_res["naive_next"],
        title="Temporal Fusion Transformer Forecast for EUR/KZT",
        ylabel="Exchange Rate",
        filename="tft_real_eurkzt_prediction.png",
        n_last=PLOT_N,
        metrics_text=(
            f"MAE: {eur_mae:.3f}%\n"
            f"RMSE: {eur_rmse:.3f}%\n"
            f"MAPE: {eur_mape:.3f}%\n"
            f"R2: {eur_r2:.4f}%\n"
        )
    )

    plot_forecast(
        dates=usd_res["test_dates"],
        actual=usd_res["actual_next"],
        predicted=usd_res["pred_next"],
        naive=usd_res["naive_next"],
        title="Temporal Fusion Transformer Forecast for USD/KZT",
        ylabel="Exchange Rate",
        filename="tft_real_usdkzt_prediction.png",
        n_last=PLOT_N,
        metrics_text=(
            f"MAE: {usd_mae:.3f}%\n"
            f"RMSE: {usd_rmse:.3f}%\n"
            f"MAPE: {usd_mape:.3f}%\n"
            f"R2: {usd_r2:.4f}"
        )
    )

    plot_r2_progress(
        eur_res["actual_next"],
        eur_res["pred_next"],
        "tft_r2_eurkzt.png"
    )

    plot_r2_progress(
        usd_res["actual_next"],
        usd_res["pred_next"],
        "tft_r2_usdkzt.png"
    )
    

    # =========================
    # 3. Текстовая сводка результатов
    # =========================
    eur_mae_n, eur_rmse_n, eur_mape_n = eur_res["naive_metrics"]
    usd_mae_n, usd_rmse_n, usd_mape_n = usd_res["naive_metrics"]

    summary_text = f"""
=== TFT metrics (%): EUR/KZT ===
MAE (%): {eur_mae:.6f}
RMSE (%): {eur_rmse:.6f}
MAPE (%): {eur_mape:.6f}

=== Naive baseline (%): EUR/KZT ===
MAE (%): {eur_mae_n:.6f}
RMSE (%): {eur_rmse_n:.6f}
MAPE (%): {eur_mape_n:.6f}

=== TFT metrics (%): USD/KZT ===
MAE (%): {usd_mae:.6f}
RMSE (%): {usd_rmse:.6f}
MAPE (%): {usd_mape:.6f}

=== Naive baseline (%): USD/KZT ===
MAE (%): {usd_mae_n:.6f}
RMSE (%): {usd_rmse_n:.6f}
MAPE (%): {usd_mape_n:.6f}
""".strip()

    with open("results_summary_tft.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    print("\nСохранены файлы:")
    print("- tft_real_eurkzt_prediction.png")
    print("- tft_real_usdkzt_prediction.png")
    print("- tft_predictions_real.csv")
    print("- results_summary_tft.txt")


if __name__ == "__main__":
    main()
