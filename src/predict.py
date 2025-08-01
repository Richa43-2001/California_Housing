import joblib
from sklearn.datasets import fetch_california_housing

model = joblib.load("model.joblib")
data = fetch_california_housing()
X = data.data

predictions = model.predict(X[:5])
print("Sample predictions:", predictions)
