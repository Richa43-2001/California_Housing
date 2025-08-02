import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, mean_squared_error

# Load data
X, y = fetch_california_housing(return_X_y=True)

# === Load original trained model ===
try:
    model = joblib.load("model.joblib")
    print("✅ Loaded model.joblib")
except FileNotFoundError:
    print("❌ model.joblib not found. Run train.py first.")
    exit(1)

# === Save Unquantized Params Manually ===
unquant_params = {
    "coef": model.coef_,
    "intercept": model.intercept_
}
joblib.dump(unquant_params, "unquant_params.joblib")
print("✅ Saved unquantized params to unquant_params.joblib")

# === Quantize to float32 ===
quantized_coef = model.coef_.astype(np.float32)
quantized_intercept = np.float32(model.intercept_)

quant_params = {
    "coef": quantized_coef,
    "intercept": quantized_intercept
}
joblib.dump(quant_params, "quant_params.joblib")
print("✅ Saved quantized params to quant_params.joblib")

# === Predictions ===

# 1. Trained model (full precision)
y_pred_trained = model.predict(X)

# 2. Unquantized params (manual but still full precision)
unq_coef = unquant_params["coef"]
unq_intercept = unquant_params["intercept"]
y_pred_unquant = X @ unq_coef + unq_intercept

# 3. Quantized (float32) params
q_coef = quant_params["coef"]
q_intercept = quant_params["intercept"]
y_pred_quant = X @ q_coef + q_intercept

# === Metrics ===
r2_trained = r2_score(y, y_pred_trained)
mse_trained = mean_squared_error(y, y_pred_trained)

r2_unquant = r2_score(y, y_pred_unquant)
mse_unquant = mean_squared_error(y, y_pred_unquant)

r2_quant = r2_score(y, y_pred_quant)
mse_quant = mean_squared_error(y, y_pred_quant)

# === Final Comparison Table ===
print("\n📊 Final Model Comparison")
print("-" * 60)
print(f"{'Model Version':<20} {'R2 Score':<12} {'MSE':<12}")
print("-" * 60)
print(f"{'Trained Model':<20} {r2_trained:<12.4f} {mse_trained:<12.4f}")
print(f"{'Unquantized Params':<20} {r2_unquant:<12.4f} {mse_unquant:<12.4f}")
print(f"{'Quantized Params':<20} {r2_quant:<12.4f} {mse_quant:<12.4f}")
print("-" * 60)
