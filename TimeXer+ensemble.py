import os
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


# =========================================================
# TimeXer-style forecasting + ensemble with CatBoost and TFT
# =========================================================
# This script is designed to fit into your current project structure.
# It:
# 1) trains a simplified TimeXer-like model for EUR/KZT and USD/KZT,
# 2) saves standalone TimeXer predictions and metrics,
# 3) optionally builds a weighted ensemble if existing CatBoost and TFT
#    prediction CSV files are present in the working directory.
#
# Expected existing files (optional for ensemble):
# - catboost_predictions.csv
# - tft_predictions_real.csv
#
# Required dataset:
# - fx_core_dataset.csv
#
# Main targets:
# - EUR/KZT next-day level
# - USD/KZT next-day level
#
# Prediction strategy:
# - predict next-day log-return
# - reconstruct next-day level as current_level * exp(pred_logret)
# =========================================================

SEP = ";"
SEED = 42
WINDOW = 64
BATCH_SIZE = 64
EPOCHS = 60
LR = 1e-3
WEIGHT_DECAY = 1e-5
D_MODEL = 64
DROPOUT = 0.15
PLOT_N = 150
EARLY_STOPPING_PATIENCE = 12
PATCH_LEN = 8
PATCH_STRIDE = 8
VAL_WEIGHT_GRID = np.arange(0.0, 1.01, 0.1)

plt.style.use("seaborn-v0_8-whitegrid")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Utilities
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calc_metrics_percent(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mean_true = np.mean(y_true)
    mae_pct = (mae / mean_true) * 100 if mean_true != 0 else np.nan
    rmse_pct = (rmse / mean_true) * 100 if mean_true != 0 else np.nan
    mape_pct = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return mae_pct, rmse_pct, mape_pct, r2

def plot_r2_vs_epochs(train_r2, val_r2, title, filename):
    epochs = range(1, len(train_r2) + 1)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    ax.plot(epochs, train_r2, label="Train R²", linewidth=2.2, color="#1f3b73")
    ax.plot(epochs, val_r2, label="Validation R²", linewidth=2.2, color="#d62728")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("R²", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best", fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def plot_forecast(dates, actual, predicted, baseline, title, ylabel, filename, n_last=150, metrics_text=None, pred_label="Forecast"):
    dates = pd.Series(dates).iloc[-n_last:]
    actual = np.asarray(actual)[-n_last:]
    predicted = np.asarray(predicted)[-n_last:]
    baseline = np.asarray(baseline)[-n_last:]

    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    ax.plot(dates, actual, label="Actual", linewidth=2.6, color="#1f3b73")
    ax.plot(dates, predicted, label=pred_label, linewidth=2.2, color="#d62728")
    ax.plot(dates, baseline, label="Naive baseline", linewidth=1.8, linestyle="--", color="#7f7f7f")

    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(loc="best", frameon=True, fancybox=True, shadow=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if metrics_text is not None:
        ax.text(
            0.98,
            0.02,
            metrics_text,
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


# =========================
# Data preparation
# =========================
def add_lag_features(df: pd.DataFrame, col: str, lags=(1, 2, 3, 5, 7, 14)):
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, col: str, windows=(3, 7, 14, 30)):
    for w in windows:
        roll = df[col].rolling(w)
        df[f"{col}_ma{w}"] = roll.mean()
        df[f"{col}_std{w}"] = roll.std()
        df[f"{col}_min{w}"] = roll.min()
        df[f"{col}_max{w}"] = roll.max()
    return df


def build_feature_frame(df: pd.DataFrame):
    base_cols = ["EURKZT", "USDKZT", "EURUSD", "DXY", "BRENT", "VIX"]

    for col in base_cols:
        df = add_lag_features(df, col)
        df = add_rolling_features(df, col)

    for col in base_cols:
        df[f"{col}_ret1"] = df[col].pct_change(1)
        df[f"{col}_ret3"] = df[col].pct_change(3)
        df[f"{col}_ret5"] = df[col].pct_change(5)
        df[f"{col}_log"] = np.log(df[col])

    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["is_month_start"] = df["Date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["Date"].dt.is_month_end.astype(int)

    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["target_eur_logret"] = np.log(df["EURKZT"].shift(-1) / df["EURKZT"])
    df["target_usd_logret"] = np.log(df["USDKZT"].shift(-1) / df["USDKZT"])
    df["actual_eur_next"] = df["EURKZT"].shift(-1)
    df["actual_usd_next"] = df["USDKZT"].shift(-1)

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    feature_cols = [
        c for c in df.columns
        if c not in {"Date", "target_eur_logret", "target_usd_logret", "actual_eur_next", "actual_usd_next"}
    ]
    return df, feature_cols


@dataclass
class SplitConfig:
    train_end: int
    val_end: int
    test_end: int


def make_splits(n_rows: int) -> SplitConfig:
    train_end = int(n_rows * 0.7)
    val_end = int(n_rows * 0.85)
    return SplitConfig(train_end=train_end, val_end=val_end, test_end=n_rows - 1)


# =========================
# Dataset
# =========================
class OneStepDataset(Dataset):
    def __init__(self, values, target_logret, current_level, actual_next, start_idx, end_idx, window):
        self.values = values
        self.target_logret = target_logret
        self.current_level = current_level
        self.actual_next = actual_next
        self.window = window
        self.indices = np.arange(max(start_idx, window - 1), end_idx)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        x = self.values[idx - self.window + 1: idx + 1]
        y = self.target_logret[idx]
        current = self.current_level[idx]
        actual = self.actual_next[idx]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(current, dtype=torch.float32),
            torch.tensor(actual, dtype=torch.float32),
            torch.tensor(idx, dtype=torch.long),
        )


# =========================
# Simplified TimeXer-like model
# =========================
class PatchEmbedding(nn.Module):
    def __init__(self, input_len: int, patch_len: int, stride: int, in_channels: int, d_model: int):
        super().__init__()
        self.input_len = input_len
        self.patch_len = patch_len
        self.stride = stride
        self.in_channels = in_channels
        self.num_patches = 1 + (input_len - patch_len) // stride
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x):
        # x: [B, T, C]
        # Return: [B, num_patches, C, d_model]
        B, T, C = x.shape
        patches = []
        for start in range(0, T - self.patch_len + 1, self.stride):
            patch = x[:, start:start + self.patch_len, :]  # [B, patch_len, C]
            patch = patch.permute(0, 2, 1)                 # [B, C, patch_len]
            patches.append(self.proj(patch))               # [B, C, d_model]
        return torch.stack(patches, dim=1)                 # [B, P, C, d_model]


class TimeXerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.patch_self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.var_cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, target_patches, exog_summary):
        # target_patches: [B, P, D]
        # exog_summary:   [B, F_exog, D]
        a1, _ = self.patch_self_attn(target_patches, target_patches, target_patches)
        x = self.norm1(target_patches + self.dropout(a1))

        a2, _ = self.var_cross_attn(x, exog_summary, exog_summary)
        x = self.norm2(x + self.dropout(a2))

        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x


class TimeXerLike(nn.Module):
    def __init__(self, num_features: int, d_model=64, num_heads=4, num_layers=2, dropout=0.15, patch_len=8, patch_stride=8, window=64, target_feature_index=0):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.target_feature_index = target_feature_index
        self.patch_embed = PatchEmbedding(window, patch_len, patch_stride, num_features, d_model)
        self.global_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.blocks = nn.ModuleList([
            TimeXerBlock(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        # x: [B, T, F]
        patches = self.patch_embed(x)                      # [B, P, F, D]
        target_patches = patches[:, :, self.target_feature_index, :]  # [B, P, D]

        exog_indices = [i for i in range(self.num_features) if i != self.target_feature_index]
        exog_repr = patches[:, :, exog_indices, :]         # [B, P, F_exog, D]
        exog_summary = exog_repr.mean(dim=1)               # [B, F_exog, D]

        global_tok = self.global_token.expand(x.size(0), -1, -1)      # [B, 1, D]
        target_patches = torch.cat([global_tok, target_patches], dim=1)

        for block in self.blocks:
            target_patches = block(target_patches, exog_summary)

        global_repr = target_patches[:, 0, :]
        pred = self.head(self.dropout(global_repr)).squeeze(-1)
        return pred


# =========================
# Training helpers
# =========================
def make_loaders(values, target, current, actual_next, split_cfg):
    train_ds = OneStepDataset(values, target, current, actual_next, 0, split_cfg.train_end, WINDOW)
    val_ds = OneStepDataset(values, target, current, actual_next, split_cfg.train_end, split_cfg.val_end, WINDOW)
    test_ds = OneStepDataset(values, target, current, actual_next, split_cfg.val_end, split_cfg.test_end, WINDOW)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    return train_loader, val_loader, test_loader


def evaluate_loss(model, loader, loss_fn):
    model.eval()
    losses = []
    with torch.no_grad():
        for xb, yb, _, _, _ in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else np.nan


def train_one_model(model, train_loader, val_loader, epochs, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=4, factor=0.5
    )
    loss_fn = nn.HuberLoss(delta=1.0)

    best_state = None
    best_val = float("inf")
    wait = 0

    train_r2_history = []
    val_r2_history = []

    for epoch in range(epochs):
        # -------------------------
        # Train
        # -------------------------
        model.train()
        train_losses = []
        y_true_train = []
        y_pred_train = []

        for xb, yb, _, _, _ in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            y_true_train.extend(yb.detach().cpu().numpy())
            y_pred_train.extend(pred.detach().cpu().numpy())

        avg_train = float(np.mean(train_losses)) if train_losses else np.nan
        train_r2 = r2_score(y_true_train, y_pred_train) if len(y_true_train) > 1 else np.nan
        train_r2_history.append(train_r2)

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        val_losses = []
        y_true_val = []
        y_pred_val = []

        with torch.no_grad():
            for xb, yb, _, _, _ in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb)
                loss = loss_fn(pred, yb)

                val_losses.append(loss.item())
                y_true_val.extend(yb.detach().cpu().numpy())
                y_pred_val.extend(pred.detach().cpu().numpy())

        avg_val = float(np.mean(val_losses)) if val_losses else np.nan
        val_r2 = r2_score(y_true_val, y_pred_val) if len(y_true_val) > 1 else np.nan
        val_r2_history.append(val_r2)

        scheduler.step(avg_val)

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"train_loss={avg_train:.6f} | val_loss={avg_val:.6f} | "
            f"train_r2={train_r2:.6f} | val_r2={val_r2:.6f} | "
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

    return model, train_r2_history, val_r2_history


def predict_levels(model, loader):
    model.eval()
    pred_logret = []
    current_levels = []
    actual_next_levels = []
    row_indices = []

    with torch.no_grad():
        for xb, _, current, actual_next, idx in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()

            pred_logret.extend(pred)
            current_levels.extend(current.numpy())
            actual_next_levels.extend(actual_next.numpy())
            row_indices.extend(idx.numpy())

    pred_logret = np.array(pred_logret)
    current_levels = np.array(current_levels)
    actual_next_levels = np.array(actual_next_levels)
    row_indices = np.array(row_indices)

    pred_next_levels = current_levels * np.exp(pred_logret)
    naive_next_levels = current_levels.copy()
    return pred_next_levels, naive_next_levels, actual_next_levels, pred_logret, row_indices


# =========================
# Validation prediction helpers for ensemble weights
# =========================
def predict_on_loader(model, loader):
    model.eval()
    pred_logret = []
    current_levels = []
    actual_next_levels = []
    row_indices = []

    with torch.no_grad():
        for xb, _, current, actual_next, idx in loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            pred_logret.extend(pred)
            current_levels.extend(current.numpy())
            actual_next_levels.extend(actual_next.numpy())
            row_indices.extend(idx.numpy())

    pred_logret = np.array(pred_logret)
    current_levels = np.array(current_levels)
    actual_next_levels = np.array(actual_next_levels)
    row_indices = np.array(row_indices)
    pred_next_levels = current_levels * np.exp(pred_logret)
    return pred_next_levels, actual_next_levels, row_indices


# =========================
# Main training routine per target
# =========================
def run_target(df, feature_cols, target_col, current_col, actual_next_col, label_prefix, target_feature_name):
    split_cfg = make_splits(len(df))
    scaler = StandardScaler()
    scaler.fit(df.loc[: split_cfg.train_end - 1, feature_cols])

    values = scaler.transform(df[feature_cols]).astype(np.float32)
    target = df[target_col].values.astype(np.float32)
    current = df[current_col].values.astype(np.float32)
    actual_next = df[actual_next_col].values.astype(np.float32)

    train_loader, val_loader, test_loader = make_loaders(values, target, current, actual_next, split_cfg)

    target_feature_index = feature_cols.index(target_feature_name)
    model = TimeXerLike(
        num_features=len(feature_cols),
        d_model=D_MODEL,
        num_heads=4,
        num_layers=2,
        dropout=DROPOUT,
        patch_len=PATCH_LEN,
        patch_stride=PATCH_STRIDE,
        window=WINDOW,
        target_feature_index=target_feature_index,
    ).to(device)

    print(f"\n=== Training TimeXer-like model for {label_prefix} ===")
    model, train_r2_history, val_r2_history = train_one_model(
        model, train_loader, val_loader, EPOCHS, LR
    )

    # График R² по эпохам
    safe_label = label_prefix.lower().replace("/", "").replace(" ", "_")
    plot_r2_vs_epochs(
        train_r2=train_r2_history,
        val_r2=val_r2_history,
        title=f"R² vs Epochs for {label_prefix}",
        filename=f"r2_vs_epochs_{safe_label}.png",
    )

    val_pred_next, val_actual_next, val_indices = predict_on_loader(model, val_loader)
    test_pred_next, test_naive_next, test_actual_next, test_pred_logret, test_indices = predict_levels(model, test_loader)

    mae_pct, rmse_pct, mape_pct, r2 = calc_metrics_percent(test_actual_next, test_pred_next)
    mae_n_pct, rmse_n_pct, mape_n_pct, r2_n = calc_metrics_percent(test_actual_next, test_naive_next)

    test_dates = df["Date"].iloc[test_indices].reset_index(drop=True)
    val_dates = df["Date"].iloc[val_indices].reset_index(drop=True)

    print(f"\n=== TimeXer metrics (%): {label_prefix} ===")
    print("MAE (%):", mae_pct)
    print("RMSE (%):", rmse_pct)
    print("MAPE (%):", mape_pct)
    print("R2:", r2)

    print(f"\n=== Naive baseline (%): {label_prefix} ===")
    print("MAE (%):", mae_n_pct)
    print("RMSE (%):", rmse_n_pct)
    print("MAPE (%):", mape_n_pct)
    print("R2:", r2_n)

    return {
        "model": model,
        "pred_next": test_pred_next,
        "naive_next": test_naive_next,
        "actual_next": test_actual_next,
        "pred_logret": test_pred_logret,
        "test_dates": test_dates,
        "test_indices": test_indices,
        "val_pred_next": val_pred_next,
        "val_actual_next": val_actual_next,
        "val_dates": val_dates,
        "val_indices": val_indices,
        "metrics": (mae_pct, rmse_pct, mape_pct, r2),
        "naive_metrics": (mae_n_pct, rmse_n_pct, mape_n_pct, r2_n),
        "train_r2_history": train_r2_history,
        "val_r2_history": val_r2_history,
    }

# =========================
# Ensemble helpers
# =========================
def load_existing_prediction_files():
    cat_path = "catboost_predictions.csv"
    tft_path = "tft_predictions_real.csv"

    cat_df = pd.read_csv(cat_path, sep=SEP) if os.path.exists(cat_path) else None
    tft_df = pd.read_csv(tft_path, sep=SEP) if os.path.exists(tft_path) else None

    if cat_df is not None and "Date" in cat_df.columns:
        cat_df["Date"] = pd.to_datetime(cat_df["Date"])
    if tft_df is not None and "Date" in tft_df.columns:
        tft_df["Date"] = pd.to_datetime(tft_df["Date"])

    return cat_df, tft_df


def build_timexer_prediction_df(res_eur, res_usd):
    return pd.DataFrame({
        "Date": pd.to_datetime(res_eur["test_dates"]),
        "Actual_EURKZT_Next": res_eur["actual_next"],
        "Predicted_EURKZT_Next": res_eur["pred_next"],
        "Naive_EURKZT_Next": res_eur["naive_next"],
        "Actual_USDKZT_Next": res_usd["actual_next"],
        "Predicted_USDKZT_Next": res_usd["pred_next"],
        "Naive_USDKZT_Next": res_usd["naive_next"],
        "Predicted_EUR_LogRet": res_eur["pred_logret"],
        "Predicted_USD_LogRet": res_usd["pred_logret"],
    })


def _simple_average_ensemble(pred_arrays):
    stacked = np.vstack(pred_arrays)
    return stacked.mean(axis=0)


def _grid_search_weights_three(pred1, pred2, pred3, y_true):
    best = None
    best_rmse = float("inf")
    for w1 in VAL_WEIGHT_GRID:
        for w2 in VAL_WEIGHT_GRID:
            w3 = 1.0 - w1 - w2
            if w3 < 0 or w3 > 1:
                continue
            pred = w1 * pred1 + w2 * pred2 + w3 * pred3
            rmse = math.sqrt(mean_squared_error(y_true, pred))
            if rmse < best_rmse:
                best_rmse = rmse
                best = (round(float(w1), 4), round(float(w2), 4), round(float(w3), 4))
    return best


def make_weighted_ensemble(cat_pred, tft_pred, timexer_pred, weights):
    w_cat, w_tft, w_timexer = weights
    return w_cat * cat_pred + w_tft * tft_pred + w_timexer * timexer_pred


def build_ensemble_on_test(cat_df, tft_df, timexer_df):
    if cat_df is None or tft_df is None:
        print("\n[INFO] Ensemble skipped: catboost_predictions.csv and/or tft_predictions_real.csv not found.")
        return None, None, None

    merged = timexer_df.merge(
        cat_df[["Date", "Predicted_EURKZT_Next", "Predicted_USDKZT_Next"]].rename(columns={
            "Predicted_EURKZT_Next": "CatBoost_EUR",
            "Predicted_USDKZT_Next": "CatBoost_USD",
        }),
        on="Date",
        how="inner",
    ).merge(
        tft_df[["Date", "Predicted_EURKZT_Next", "Predicted_USDKZT_Next"]].rename(columns={
            "Predicted_EURKZT_Next": "TFT_EUR",
            "Predicted_USDKZT_Next": "TFT_USD",
        }),
        on="Date",
        how="inner",
    )

    merged = merged.rename(columns={
        "Predicted_EURKZT_Next": "TimeXer_EUR",
        "Predicted_USDKZT_Next": "TimeXer_USD",
    })

    # default equal weights
    eq_weights = (1 / 3, 1 / 3, 1 / 3)

    merged["Ensemble_EUR"] = make_weighted_ensemble(
        merged["CatBoost_EUR"].values,
        merged["TFT_EUR"].values,
        merged["TimeXer_EUR"].values,
        eq_weights,
    )
    merged["Ensemble_USD"] = make_weighted_ensemble(
        merged["CatBoost_USD"].values,
        merged["TFT_USD"].values,
        merged["TimeXer_USD"].values,
        eq_weights,
    )

    eur_metrics = calc_metrics_percent(merged["Actual_EURKZT_Next"].values, merged["Ensemble_EUR"].values)
    usd_metrics = calc_metrics_percent(merged["Actual_USDKZT_Next"].values, merged["Ensemble_USD"].values)
    return merged, eur_metrics, usd_metrics


# =========================
# Main
# =========================
def main():
    set_seed(SEED)

    df = pd.read_csv("fx_core_dataset.csv", sep=SEP, index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.reset_index()

    df, feature_cols = build_feature_frame(df)

    eur_res = run_target(
        df=df,
        feature_cols=feature_cols,
        target_col="target_eur_logret",
        current_col="EURKZT",
        actual_next_col="actual_eur_next",
        label_prefix="EUR/KZT",
        target_feature_name="EURKZT",
    )

    usd_res = run_target(
        df=df,
        feature_cols=feature_cols,
        target_col="target_usd_logret",
        current_col="USDKZT",
        actual_next_col="actual_usd_next",
        label_prefix="USD/KZT",
        target_feature_name="USDKZT",
    )

    # -----------------------------------------------------
    # Save TimeXer predictions
    # -----------------------------------------------------
    timexer_df = build_timexer_prediction_df(eur_res, usd_res)
    timexer_df.to_csv("timexer_predictions.csv", sep=SEP, index=False)

    eur_mae, eur_rmse, eur_mape, eur_r2 = eur_res["metrics"]
    usd_mae, usd_rmse, usd_mape, usd_r2 = usd_res["metrics"]

    plot_forecast(
        dates=eur_res["test_dates"],
        actual=eur_res["actual_next"],
        predicted=eur_res["pred_next"],
        baseline=eur_res["naive_next"],
        title="TimeXer Forecast for EUR/KZT",
        ylabel="Exchange Rate",
        filename="timexer_eurkzt_prediction.png",
        n_last=PLOT_N,
        pred_label="TimeXer forecast",
        metrics_text=(
            f"MAE: {eur_mae:.3f}%\n"
            f"RMSE: {eur_rmse:.3f}%\n"
            f"MAPE: {eur_mape:.3f}%\n"
            f"R2: {eur_r2:.3f}"
        ),
    )

    plot_forecast(
        dates=usd_res["test_dates"],
        actual=usd_res["actual_next"],
        predicted=usd_res["pred_next"],
        baseline=usd_res["naive_next"],
        title="TimeXer Forecast for USD/KZT",
        ylabel="Exchange Rate",
        filename="timexer_usdkzt_prediction.png",
        n_last=PLOT_N,
        pred_label="TimeXer forecast",
        metrics_text=(
            f"MAE: {usd_mae:.3f}%\n"
            f"RMSE: {usd_rmse:.3f}%\n"
            f"MAPE: {usd_mape:.3f}%\n"
            f"R2: {usd_r2:.3f}"
        ),
    )

    # -----------------------------------------------------
    # Save TimeXer summary
    # -----------------------------------------------------
    eur_mae_n, eur_rmse_n, eur_mape_n, eur_r2_n = eur_res["naive_metrics"]
    usd_mae_n, usd_rmse_n, usd_mape_n, usd_r2_n = usd_res["naive_metrics"]

    summary_text = f"""
=== TimeXer metrics (%): EUR/KZT ===
MAE (%): {eur_mae:.6f}
RMSE (%): {eur_rmse:.6f}
MAPE (%): {eur_mape:.6f}
R2: {eur_r2:.6f}

=== Naive baseline (%): EUR/KZT ===
MAE (%): {eur_mae_n:.6f}
RMSE (%): {eur_rmse_n:.6f}
MAPE (%): {eur_mape_n:.6f}
R2: {eur_r2_n:.6f}

=== TimeXer metrics (%): USD/KZT ===
MAE (%): {usd_mae:.6f}
RMSE (%): {usd_rmse:.6f}
MAPE (%): {usd_mape:.6f}
R2: {usd_r2:.6f}

=== Naive baseline (%): USD/KZT ===
MAE (%): {usd_mae_n:.6f}
RMSE (%): {usd_rmse_n:.6f}
MAPE (%): {usd_mape_n:.6f}
R2: {usd_r2_n:.6f}
""".strip()

    with open("results_summary_timexer.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    # -----------------------------------------------------
    # Ensemble block
    # -----------------------------------------------------
    cat_df, tft_df = load_existing_prediction_files()
    ensemble_df, ensemble_eur_metrics, ensemble_usd_metrics = build_ensemble_on_test(cat_df, tft_df, timexer_df)

    comparison_rows = [
        {
            "Model": "TimeXer",
            "Target": "EUR/KZT",
            "MAE_%": eur_mae,
            "RMSE_%": eur_rmse,
            "MAPE_%": eur_mape,
            "R2": eur_r2,
        },
        {
            "Model": "Naive baseline",
            "Target": "EUR/KZT",
            "MAE_%": eur_mae_n,
            "RMSE_%": eur_rmse_n,
            "MAPE_%": eur_mape_n,
            "R2": eur_r2_n,
        },
        {
            "Model": "TimeXer",
            "Target": "USD/KZT",
            "MAE_%": usd_mae,
            "RMSE_%": usd_rmse,
            "MAPE_%": usd_mape,
            "R2": usd_r2,
        },
        {
            "Model": "Naive baseline",
            "Target": "USD/KZT",
            "MAE_%": usd_mae_n,
            "RMSE_%": usd_rmse_n,
            "MAPE_%": usd_mape_n,
            "R2": usd_r2_n,
        },
    ]

    if ensemble_df is not None:
        ensemble_df.to_csv("ensemble_predictions_equal_weights.csv", sep=SEP, index=False)

        eur_e_mae, eur_e_rmse, eur_e_mape, eur_e_r2 = ensemble_eur_metrics
        usd_e_mae, usd_e_rmse, usd_e_mape, usd_e_r2 = ensemble_usd_metrics

        comparison_rows.extend([
            {
                "Model": "Ensemble (equal weights)",
                "Target": "EUR/KZT",
                "MAE_%": eur_e_mae,
                "RMSE_%": eur_e_rmse,
                "MAPE_%": eur_e_mape,
                "R2": eur_e_r2,
            },
            {
                "Model": "Ensemble (equal weights)",
                "Target": "USD/KZT",
                "MAE_%": usd_e_mae,
                "RMSE_%": usd_e_rmse,
                "MAPE_%": usd_e_mape,
                "R2": usd_e_r2,
            },
        ])

        plot_forecast(
            dates=ensemble_df["Date"],
            actual=ensemble_df["Actual_EURKZT_Next"],
            predicted=ensemble_df["Ensemble_EUR"],
            baseline=ensemble_df["Naive_EURKZT_Next"],
            title="Ensemble Forecast for EUR/KZT",
            ylabel="Exchange Rate",
            filename="ensemble_eurkzt_prediction.png",
            n_last=PLOT_N,
            pred_label="Ensemble forecast",
            metrics_text=(
                f"MAE: {eur_e_mae:.3f}%\n"
                f"RMSE: {eur_e_rmse:.3f}%\n"
                f"MAPE: {eur_e_mape:.3f}%\n"
                f"R2: {eur_e_r2:.3f}"
            ),
        )

        plot_forecast(
            dates=ensemble_df["Date"],
            actual=ensemble_df["Actual_USDKZT_Next"],
            predicted=ensemble_df["Ensemble_USD"],
            baseline=ensemble_df["Naive_USDKZT_Next"],
            title="Ensemble Forecast for USD/KZT",
            ylabel="Exchange Rate",
            filename="ensemble_usdkzt_prediction.png",
            n_last=PLOT_N,
            pred_label="Ensemble forecast",
            metrics_text=(
                f"MAE: {usd_e_mae:.3f}%\n"
                f"RMSE: {usd_e_rmse:.3f}%\n"
                f"MAPE: {usd_e_mape:.3f}%\n"
                f"R2: {usd_e_r2:.3f}"
            ),
        )
        
        

        with open("results_summary_ensemble.txt", "w", encoding="utf-8") as f:
            f.write(
                f"""
=== Ensemble (equal weights): EUR/KZT ===
MAE (%): {eur_e_mae:.6f}
RMSE (%): {eur_e_rmse:.6f}
MAPE (%): {eur_e_mape:.6f}
R2: {eur_e_r2:.6f}

=== Ensemble (equal weights): USD/KZT ===
MAE (%): {usd_e_mae:.6f}
RMSE (%): {usd_e_rmse:.6f}
MAPE (%): {usd_e_mape:.6f}
R2: {usd_e_r2:.6f}
""".strip()
            )

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv("timexer_model_comparison.csv", sep=SEP, index=False)

    print("\n=== Model comparison ===")
    print(comparison)

    print("\nSaved files:")
    print("- timexer_predictions.csv")
    print("- timexer_eurkzt_prediction.png")
    print("- timexer_usdkzt_prediction.png")
    print("- results_summary_timexer.txt")
    print("- timexer_model_comparison.csv")
    if ensemble_df is not None:
        print("- ensemble_predictions_equal_weights.csv")
        print("- ensemble_eurkzt_prediction.png")
        print("- ensemble_usdkzt_prediction.png")
        print("- results_summary_ensemble.txt")


if __name__ == "__main__":
    main()
