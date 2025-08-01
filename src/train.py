from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Train model
model = LinearRegression()
model.fit(X, y)

# Evaluate
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
loss = mean_squared_error(y, y_pred)

print(f"R2 Score: {r2}")
print(f"MSE Loss: {loss}")

# Save model
joblib.dump(model, "model.joblib")
