import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split

# 1. Generate synthetic data: 100 features, but only 5 are actually informative
X, y, true_coefficients = make_regression(
    n_samples=40, 
    n_features=100, 
    n_informative=5, 
    noise=1.0, 
    coef=True, 
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train Standard Linear Regression (No Regularization)
lr = LinearRegression()
lr.fit(X_train, y_train)

# 3. Train Lasso Regression (L1 Regularization, alpha is our lambda)
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)

# 4. Analyze the results
print(f"True number of active features: 5")
print(f"Linear Regression active features (weight != 0): {np.sum(lr.coef_ != 0)}")
print(f"Lasso (L1) active features (weight != 0): {np.sum(lasso.coef_ != 0)}")

# 5. Evaluate Performance
print(f"\nLinear Regression Test R^2 Score: {lr.score(X_test, y_test):.4f}")
print(f"Lasso (L1) Test R^2 Score: {lasso.score(X_test, y_test):.4f}")