import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Load trained model
print("Loading trained model...")
model = joblib.load("model.joblib")

# Save unquantized parameters
params = {
    "coef": model.coef_,
    "intercept": model.intercept_
}
joblib.dump(params, "unquant_params.joblib")
print("Saved unquantized parameters to 'unquant_params.joblib'")

# Manual quantization
quantized_coef = (model.coef_ * 100).astype(np.uint8)
quantized_intercept = int(model.intercept_ * 100)

quant_params = {
    "coef": quantized_coef,
    "intercept": quantized_intercept
}
joblib.dump(quant_params, "quant_params.joblib")
print("Saved quantized parameters to 'quant_params.joblib'")

# Dequantization
dequant_coef = quantized_coef.astype(np.float64) / 100
dequant_intercept = quantized_intercept / 100

# Prediction using dequantized weights
y_pred_quant = X @ dequant_coef + dequant_intercept

# Prediction using original model
y_pred_unquant = model.predict(X)

# Evaluation
r2_unquant = r2_score(y, y_pred_unquant)
r2_quant = r2_score(y, y_pred_quant)

mse_unquant = mean_squared_error(y, y_pred_unquant)
mse_quant = mean_squared_error(y, y_pred_quant)

# Comparison Table
print("\n📊 Model Comparison")
print("-" * 40)
print(f"{'Model':<15} {'R2 Score':<10} {'MSE':<10}")
print("-" * 40)
print(f"{'Unquantized':<15} {r2_unquant:<10.4f} {mse_unquant:<10.4f}")
print(f"{'Quantized':<15} {r2_quant:<10.4f} {mse_quant:<10.4f}")
print("-" * 40)
