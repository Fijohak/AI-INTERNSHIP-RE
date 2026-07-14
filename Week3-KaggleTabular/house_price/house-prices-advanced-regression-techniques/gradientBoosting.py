"""房价预测 — GradientBoostingRegressor"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import os

BASE = os.path.dirname(__file__)
PROC = os.path.join(BASE, "processed")

X_train_full = np.load(f"{PROC}/X_train.npy")
y_train_full = np.load(f"{PROC}/y_train.npy")
X_test       = np.load(f"{PROC}/X_test.npy")
test_ids     = np.load(f"{PROC}/test_ids.npy")

print(f"train: {X_train_full.shape}, test: {X_test.shape}")

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

model = GradientBoostingRegressor(
    n_estimators=500, max_depth=5, min_samples_leaf=5,
    learning_rate=0.05, subsample=0.8, random_state=42,
)
model.fit(X_tr, y_tr)

y_val_pred = model.predict(X_val)
y_val_real = np.expm1(y_val)
y_pred_real = np.expm1(y_val_pred)

rmse_log = np.sqrt(mean_squared_error(y_val, y_val_pred))
rmse_dollar = np.sqrt(mean_squared_error(y_val_real, y_pred_real))
mae_dollar = mean_absolute_error(y_val_real, y_pred_real)
r2 = r2_score(y_val, y_val_pred)

print(f"Val RMSE (log): {rmse_log:.5f}")
print(f"Val RMSE ($):   ${rmse_dollar:,.0f}")
print(f"Val MAE  ($):   ${mae_dollar:,.0f}")
print(f"Val R2  (log):  {r2:.4f}")

model.fit(X_train_full, y_train_full)
y_test_pred_log = model.predict(X_test)
y_test_pred = np.expm1(y_test_pred_log)

submission = pd.DataFrame({"Id": test_ids.astype(int), "SalePrice": y_test_pred})
submission.to_csv(f"{BASE}/submission_gbr.csv", index=False)
print(f"Saved: submission_gbr.csv")

with open(f"{PROC}/feature_names.txt", "r", encoding="utf-8") as f:
    names = [line.strip() for line in f]
importances = model.feature_importances_
for idx in np.argsort(importances)[::-1][:15]:
    print(f"  {names[idx]:<30s} {importances[idx]:.4f}")
