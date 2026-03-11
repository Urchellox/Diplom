import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import matplotlib.pyplot as plt

df = pd.read_csv("fx_multimodal_dataset_2010_present.csv", sep=";")
df["Date"] = pd.to_datetime(df["Date"])

df["target_next"] = df["EURKZT"].shift(-1)
df = df.dropna().reset_index(drop=True)

window = 30
features = [c for c in df.columns if c != "Date"]

split = int(len(df) * 0.8)

scaler = StandardScaler()
scaler.fit(df.loc[:split-1, features])

values = scaler.transform(df[features]).astype(np.float32)
targets = df["target_next"].values.astype(np.float32)

class SeqDataset(Dataset):

    def __init__(self, values, targets, start, end, window):
        self.values = values
        self.targets = targets
        self.start = max(start, window)
        self.end = end
        self.window = window

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, i):
        idx = self.start + i
        x = self.values[idx-self.window:idx]
        y = self.targets[idx]
        return torch.tensor(x), torch.tensor(y)

train_ds = SeqDataset(values, targets, 0, split, window)
test_ds = SeqDataset(values, targets, split, len(values)-1, window)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=128)

class TFTStyle(nn.Module):

    def __init__(self, n_features):
        super().__init__()

        hidden = 32

        self.varsel = nn.Linear(n_features, n_features)
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.attn = nn.MultiheadAttention(hidden, 2, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):

        w = torch.sigmoid(self.varsel(x))
        x = x * w

        h, _ = self.lstm(x)

        attn, _ = self.attn(h, h, h)

        out = self.fc(attn[:, -1])

        return out.squeeze()

model = TFTStyle(len(features))

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(5):

    for xb, yb in train_loader:

        optimizer.zero_grad()

        pred = model(xb)

        loss = loss_fn(pred, yb)

        loss.backward()

        optimizer.step()

preds = []
truth = []

with torch.no_grad():

    for xb, yb in test_loader:

        p = model(xb)

        preds.extend(p.numpy())
        truth.extend(yb.numpy())

preds = np.array(preds)
truth = np.array(truth)

mae = mean_absolute_error(truth, preds)
rmse = math.sqrt(mean_squared_error(truth, preds))
mape = np.mean(np.abs((truth - preds) / truth)) * 100

print("TFT metrics")
print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE:", mape)

plt.figure(figsize=(10,5))

plt.plot(truth[:150], label="Actual")
plt.plot(preds[:150], label="Predicted")

plt.legend()
plt.tight_layout()

plt.savefig("tft_prediction.png")