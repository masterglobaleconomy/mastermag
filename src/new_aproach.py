import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes
from pathlib import Path
from config import DATA_DIR

def model(t, N, L):
    return N * np.exp(-t * L)

data = pd.read_csv(DATA_DIR / 'pkn_d.csv')
X = np.array(data.index.array)
y = np.array(data.Zamkniecie.array)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lasso = Lasso(alpha=0.1)

lasso.fit(X_train.reshape(-1, 1), y_train)

y_pred = lasso.predict(X_test.reshape(-1, 1))

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Wyniki Lasso Regression:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

print("\nWspółczynniki regresji:")
print(f"{lasso.coef_[0]:.4f}")
