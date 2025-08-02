from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np

# Load dataset
X, y = fetch_california_housing(return_X_y=True)

# Train model
model = LinearRegression()
model.fit(X, y)

# Evaluate
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

print(f"✅ R2 Score: {r2:.4f}")
print(f"✅ MSE Loss: {mse:.4f}")

# Save trained model
joblib.dump(model, "model.joblib")
print("✅ model.joblib saved")

# Save unquantized params for predict.py
unquant_params = {
    "coef": model.coef_,
    "intercept": model.intercept_
}
joblib.dump(unquant_params, "unquant_params.joblib")
print("✅ unquant_params.joblib saved")
