import subprocess
import sys

print("Training CatBoost...")
subprocess.run([sys.executable, "train_catboost.py"], check=True)

print("Training TFT model...")
subprocess.run([sys.executable, "train_tft.py"], check=True)