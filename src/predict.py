import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing

# Load data
data = fetch_california_housing()
X = data.data

# Load original model
model = joblib.load("model.joblib")
y_pred_original = model.predict(X[:5])
print("Original Model Predictions:", y_pred_original)

# Load quantized params
quant_params = joblib.load("quant_params.joblib")
quantized_coef = quant_params["coef"]
quantized_intercept = quant_params["intercept"]

# Dequantize
dequant_coef = quantized_coef.astype(np.float64) / 100
dequant_intercept = quantized_intercept / 100

# Predict using dequantized weights
y_pred_quantized = X[:5] @ dequant_coef + dequant_intercept
print("Quantized Model Predictions:", y_pred_quantized)
