import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score, GridSearchCV

print("=== Step 1: Generate Synthetic Data ===")
# Generate synthetic data: y = 2*x + 1 + some noise
np.random.seed(42)  # For reproducibility
X = 2 * np.random.rand(100, 1)  # 100 random numbers between 0 and 2
y = 1 + 2 * X + np.random.randn(100, 1)  # y = 1 + 2x + noise

# Plot and save the raw data
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.7, label='Raw Data')
plt.xlabel('X (Feature)')
plt.ylabel('y (Target)')
plt.title('Synthetic Data for Linear Regression: y = 1 + 2x + noise')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('synthetic_data.png')
plt.close()
print("Saved 'synthetic_data.png'")
print(f"Generated {len(X)} data points")

print("\n=== Step 2: Implement Gradient Descent from Scratch ===")
# Initialize parameters (theta0 and theta1) randomly
theta = np.random.randn(2, 1)  # Random initialization
print(f"Initial random parameters: theta0 = {theta[0][0]:.3f}, theta1 = {theta[1][0]:.3f}")

# Add a column of 1s to X for the intercept term (theta0)
X_b = np.c_[np.ones((100, 1)), X]  # Now X_b is [1, x]

# Hyperparameters
learning_rate = 0.1
iterations = 1000
m = len(X_b)

print(f"Starting gradient descent with learning_rate={learning_rate}, iterations={iterations}")

# Gradient Descent Loop
for iteration in range(iterations):
    # 1. Calculate gradients (feel the slope)
    gradients = (2/m) * X_b.T.dot(X_b.dot(theta) - y)
    
    # 2. Update parameters (take a step downhill)
    theta = theta - learning_rate * gradients
    
    # Print progress every 100 iterations
    if iteration % 100 == 0:
        # Calculate current cost for monitoring
        current_predictions = X_b.dot(theta)
        current_error = current_predictions - y
        cost = (1/m) * np.sum(current_error ** 2)
        print(f"Iteration {iteration}: theta0 = {theta[0][0]:.3f}, theta1 = {theta[1][0]:.3f}, Cost = {cost:.3f}")

print(f"\nFinal parameters after {iterations} iterations:")
print(f"theta0 (Intercept) = {theta[0][0]:.3f}")
print(f"theta1 (Slope) = {theta[1][0]:.3f}")
print("Expected values should be close to: theta0 = 1.0, theta1 = 2.0")

print("\n=== Step 3: Make Predictions & Plot Linear Regression Result ===")
# Make predictions using our final model
y_pred = X_b.dot(theta)

# Plot the linear regression result
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.7, label='Raw Data')
plt.plot(X, y_pred, 'r-', linewidth=2, label=f'Linear Fit (Scratch): y = {theta[0][0]:.2f} + {theta[1][0]:.2f}x')
plt.xlabel('X (Feature)')
plt.ylabel('y (Target)')
plt.title('Linear Regression Fit (From Scratch)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('linear_regression_fit_scratch.png')
plt.close()
print("Saved 'linear_regression_fit_scratch.png'")

# Calculate final error for linear regression (scratch)
final_error = y_pred - y
final_cost = (1/m) * np.sum(final_error ** 2)
print(f"Final Mean Squared Error (Linear Scratch): {final_cost:.3f}")

print("\n=== Step 4: Compare with scikit-learn Linear Regression ===")
# Train using scikit-learn for comparison
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_sklearn = lin_reg.predict(X)

print(f"scikit-learn Linear parameters: Intercept = {lin_reg.intercept_[0]:.3f}, Slope = {lin_reg.coef_[0][0]:.3f}")
print(f"Our Linear parameters:         Intercept = {theta[0][0]:.3f}, Slope = {theta[1][0]:.3f}")

# Check if they're close
tolerance = 0.1
if (abs(lin_reg.intercept_[0] - theta[0][0]) < tolerance and 
    abs(lin_reg.coef_[0][0] - theta[1][0]) < tolerance):
    print("✅ Success! Our linear implementation matches scikit-learn closely.")
else:
    print("❌ Linear results differ significantly. Check the implementation.")

print("\n=== Step 5: Polynomial Regression (Degree 2) ===")
# Transform features to include polynomial terms (degree 2)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)  # X_poly is [x, x^2]

# Train polynomial regression model
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
print(f"Polynomial Regression parameters: Intercept = {poly_reg.intercept_[0]:.3f}, "
      f"Coefficients = {[f'{coef:.3f}' for coef in poly_reg.coef_[0]]}")

# Make predictions for plotting (sort X for smooth curve)
X_plot = np.linspace(0, 2, 100).reshape(-1, 1)
X_plot_poly = poly_features.transform(X_plot)
y_plot_poly = poly_reg.predict(X_plot_poly)

# Calculate MSE for polynomial regression
y_pred_poly = poly_reg.predict(X_poly)
poly_error = y_pred_poly - y
poly_cost = (1/m) * np.sum(poly_error ** 2)
print(f"Final Mean Squared Error (Polynomial): {poly_cost:.3f}")

print("\n=== Step 6: Ridge Regression (Linear and Polynomial) ===")
# Ridge Regression for linear features
ridge_reg = Ridge(alpha=1.0)  # alpha controls regularization strength
ridge_reg.fit(X, y.ravel())  # Use y.ravel() to ensure 1D target
y_pred_ridge = ridge_reg.predict(X)
print(f"Ridge Linear parameters: Intercept = {ridge_reg.intercept_:.3f}, Slope = {ridge_reg.coef_[0]:.3f}")

# Ridge Regression for polynomial features
ridge_poly_reg = Ridge(alpha=1.0)
ridge_poly_reg.fit(X_poly, y.ravel())
y_plot_ridge_poly = ridge_poly_reg.predict(X_plot_poly)
print(f"Ridge Polynomial parameters: Intercept = {ridge_poly_reg.intercept_:.3f}, "
      f"Coefficients = {[f'{coef:.3f}' for coef in ridge_poly_reg.coef_]}")

# Calculate MSE for Ridge models
y_pred_ridge_poly = ridge_poly_reg.predict(X_poly)
ridge_poly_error = y_pred_ridge_poly - y.ravel()
ridge_poly_cost = (1/m) * np.sum(ridge_poly_error ** 2)
print(f"Final Mean Squared Error (Ridge Polynomial): {ridge_poly_cost:.3f}")

print("\n=== Step 7: Lasso Regression (Linear and Polynomial) ===")
# Lasso Regression for linear features
lasso_reg = Lasso(alpha=1.0)
lasso_reg.fit(X, y.ravel())
y_pred_lasso = lasso_reg.predict(X)
print(f"Lasso Linear parameters: Intercept = {lasso_reg.intercept_:.3f}, Slope = {lasso_reg.coef_[0]:.3f}")

# Lasso Regression for polynomial features
lasso_poly_reg = Lasso(alpha=1.0)
lasso_poly_reg.fit(X_poly, y.ravel())
y_plot_lasso_poly = lasso_poly_reg.predict(X_plot_poly)
print(f"Lasso Polynomial parameters: Intercept = {lasso_poly_reg.intercept_:.3f}, "
      f"Coefficients = {[f'{coef:.3f}' for coef in lasso_poly_reg.coef_]}")

# Calculate MSE for Lasso models
y_pred_lasso_poly = lasso_poly_reg.predict(X_poly)
lasso_poly_error = y_pred_lasso_poly - y.ravel()
lasso_poly_cost = (1/m) * np.sum(lasso_poly_error ** 2)
print(f"Final Mean Squared Error (Lasso Polynomial): {lasso_poly_cost:.3f}")

print("\n=== Step 8: Cross-Validation ===")
# Perform 5-fold cross-validation
cv_folds = 5
lin_cv_scores = cross_val_score(lin_reg, X, y.ravel(), scoring='neg_mean_squared_error', cv=cv_folds)
poly_cv_scores = cross_val_score(poly_reg, X_poly, y.ravel(), scoring='neg_mean_squared_error', cv=cv_folds)
ridge_cv_scores = cross_val_score(ridge_reg, X, y.ravel(), scoring='neg_mean_squared_error', cv=cv_folds)
ridge_poly_cv_scores = cross_val_score(ridge_poly_reg, X_poly, y.ravel(), scoring='neg_mean_squared_error', cv=cv_folds)
lasso_cv_scores = cross_val_score(lasso_reg, X, y.ravel(), scoring='neg_mean_squared_error', cv=cv_folds)
lasso_poly_cv_scores = cross_val_score(lasso_poly_reg, X_poly, y.ravel(), scoring='neg_mean_squared_error', cv=cv_folds)

print(f"Cross-Validation MSE (Linear): {-lin_cv_scores.mean():.3f} (+/- {lin_cv_scores.std() * 2:.3f})")
print(f"Cross-Validation MSE (Polynomial): {-poly_cv_scores.mean():.3f} (+/- {poly_cv_scores.std() * 2:.3f})")
print(f"Cross-Validation MSE (Ridge Linear): {-ridge_cv_scores.mean():.3f} (+/- {ridge_cv_scores.std() * 2:.3f})")
print(f"Cross-Validation MSE (Ridge Polynomial): {-ridge_poly_cv_scores.mean():.3f} (+/- {ridge_poly_cv_scores.std() * 2:.3f})")
print(f"Cross-Validation MSE (Lasso Linear): {-lasso_cv_scores.mean():.3f} (+/- {lasso_cv_scores.std() * 2:.3f})")
print(f"Cross-Validation MSE (Lasso Polynomial): {-lasso_poly_cv_scores.mean():.3f} (+/- {lasso_poly_cv_scores.std() * 2:.3f})")

print("\n=== Step 9: Hyperparameter Tuning (Grid Search) ===")
# Grid search for Ridge and Lasso alpha
alpha_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}

# Ridge Linear
ridge_grid = GridSearchCV(Ridge(), alpha_grid, scoring='neg_mean_squared_error', cv=cv_folds)
ridge_grid.fit(X, y.ravel())
print(f"Best Ridge Linear alpha: {ridge_grid.best_params_['alpha']}, Best CV MSE: {-ridge_grid.best_score_:.3f}")

# Ridge Polynomial
ridge_poly_grid = GridSearchCV(Ridge(), alpha_grid, scoring='neg_mean_squared_error', cv=cv_folds)
ridge_poly_grid.fit(X_poly, y.ravel())
print(f"Best Ridge Polynomial alpha: {ridge_poly_grid.best_params_['alpha']}, Best CV MSE: {-ridge_poly_grid.best_score_:.3f}")

# Lasso Linear
lasso_grid = GridSearchCV(Lasso(), alpha_grid, scoring='neg_mean_squared_error', cv=cv_folds)
lasso_grid.fit(X, y.ravel())
print(f"Best Lasso Linear alpha: {lasso_grid.best_params_['alpha']}, Best CV MSE: {-lasso_grid.best_score_:.3f}")

# Lasso Polynomial
lasso_poly_grid = GridSearchCV(Lasso(), alpha_grid, scoring='neg_mean_squared_error', cv=cv_folds)
lasso_poly_grid.fit(X_poly, y.ravel())
print(f"Best Lasso Polynomial alpha: {lasso_poly_grid.best_params_['alpha']}, Best CV MSE: {-lasso_poly_grid.best_score_:.3f}")

print("\n=== Step 10: Plot All Models ===")
# Plot all fits
plt.figure(figsize=(12, 8))
plt.scatter(X, y, alpha=0.7, label='Raw Data')
plt.plot(X, y_pred, 'r-', linewidth=2, label=f'Linear Fit (Scratch): y = {theta[0][0]:.2f} + {theta[1][0]:.2f}x')
plt.plot(X_plot, y_plot_poly, 'g--', linewidth=2, label='Polynomial Fit (Degree 2)')
plt.plot(X_plot, ridge_reg.predict(X_plot), 'b-.', linewidth=2, label=f'Ridge Linear: y = {ridge_reg.intercept_:.2f} + {ridge_reg.coef_[0]:.2f}x')
plt.plot(X_plot, y_plot_ridge_poly, 'm:', linewidth=2, label='Ridge Polynomial (Degree 2)')
plt.plot(X_plot, lasso_reg.predict(X_plot), 'c-', linewidth=2, label=f'Lasso Linear: y = {lasso_reg.intercept_:.2f} + {lasso_reg.coef_[0]:.2f}x')
plt.plot(X_plot, y_plot_lasso_poly, 'y--', linewidth=2, label='Lasso Polynomial (Degree 2)')
plt.xlabel('X (Feature)')
plt.ylabel('y (Target)')
plt.title('Linear vs Polynomial vs Ridge vs Lasso Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('all_regression_fits.png')
plt.close()
print("Saved 'all_regression_fits.png'")

print("\n=== Script Complete ===")