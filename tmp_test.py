#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate sample data for two features (X1, X2)
np.random.seed(42)
X1 = 2 * np.random.rand(100, 1)
X2 = 3 * np.random.rand(100, 1)
y = 5 + 2 * X1 + 3 * X2 + 1.5 * X1**2 + 2 * X1 * X2 + np.random.randn(100, 1)

# Combine features into a single array
X = np.hstack((X1, X2))

# Create polynomial features (degree=2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Fit the polynomial regression model
model = LinearRegression()
model.fit(X_poly, y)

# Get model coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Predict on the training set
y_pred = model.predict(X_poly)

# Compute Mean Squared Error
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error:", mse)

#%%
# Visualize results (on 2D projection for simplicity)
plt.scatter(y, y_pred, color='blue', label='Predicted vs True')
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label='Perfect Fit')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.title("Multivariate Polynomial Fitting (Degree=2)")
plt.show()

# %%
