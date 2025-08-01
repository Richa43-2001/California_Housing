import joblib
import numpy as np

model = joblib.load("model.joblib")

# Save unquantized params
params = {
    "coef": model.coef_,
    "intercept": model.intercept_
}
joblib.dump(params, "unquant_params.joblib")

# Quantization
quantized_coef = (model.coef_ * 100).astype(np.uint8)
quantized_intercept = int(model.intercept_ * 100)

quant_params = {
    "coef": quantized_coef,
    "intercept": quantized_intercept
}
joblib.dump(quant_params, "quant_params.joblib")

# Inference with dequantized weights
dequant_coef = quantized_coef.astype(np.float64) / 100
dequant_intercept = quantized_intercept / 100

def predict(X):
    return X @ dequant_coef + dequant_intercept
