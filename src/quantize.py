import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, mean_squared_error

# Load data
X, y = fetch_california_housing(return_X_y=True)

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

# Float32 quantization (simulate lower-precision storage)
quantized_coef = model.coef_.astype(np.float32)
quantized_intercept = np.float32(model.intercept_)

quant_params = {
    "coef": quantized_coef,
    "intercept": quantized_intercept
}
joblib.dump(quant_params, "quant_params.joblib")
print("Saved quantized parameters to 'quant_params.joblib'")

# Inference with quantized weights
y_pred_quant = X @ quantized_coef + quantized_intercept

# Inference with original model
y_pred_orig = model.predict(X)

# Evaluation
r2_orig = r2_score(y, y_pred_orig)
mse_orig = mean_squared_error(y, y_pred_orig)

r2_quant = r2_score(y, y_pred_quant)
mse_quant = mean_squared_error(y, y_pred_quant)

# Comparison table
print("\n📊 Model Comparison (Original vs Quantized float32)")
print("-" * 50)
print(f"{'Model':<15} {'R2 Score':<12} {'MSE':<12}")
print("-" * 50)
print(f"{'Original':<15} {r2_orig:<12.4f} {mse_orig:<12.4f}")
print(f"{'Quantized':<15} {r2_quant:<12.4f} {mse_quant:<12.4f}")
print("-" * 50)
