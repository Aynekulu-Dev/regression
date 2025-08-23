# simple_linear_regression.py
import numpy as np
import matplotlib.pyplot as plt

print("=== Step 1: Generate Synthetic Data ===")
# Generate synthetic data: y = 2*x + 1 + some noise
np.random.seed(42) # For reproducibility
X = 2 * np.random.rand(100, 1) # 100 random numbers between 0 and 2
y = 1 + 2 * X + np.random.randn(100, 1) # y = 1 + 2x + noise

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
theta = np.random.randn(2, 1) # Random initialization
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

print("\n=== Step 3: Make Predictions & Plot the Result ===")
# Make predictions using our final model
y_pred = X_b.dot(theta)

# Plot the final result
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.7, label='Raw Data')
plt.plot(X, y_pred, 'r-', linewidth=2, label=f'Best Fit: y = {theta[0][0]:.2f} + {theta[1][0]:.2f}x')
plt.xlabel('X (Feature)')
plt.ylabel('y (Target)')
plt.title('Linear Regression Fit (From Scratch)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('linear_regression_fit_scratch.png')
plt.close()
print("Saved 'linear_regression_fit_scratch.png'")

# Calculate final error
final_error = y_pred - y
final_cost = (1/m) * np.sum(final_error ** 2)
print(f"Final Mean Squared Error: {final_cost:.3f}")

print("\n=== Bonus: Compare with scikit-learn ===")
from sklearn.linear_model import LinearRegression

# Train using scikit-learn for comparison
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_sklearn = lin_reg.predict(X)

print(f"scikit-learn parameters: Intercept = {lin_reg.intercept_[0]:.3f}, Slope = {lin_reg.coef_[0][0]:.3f}")
print(f"Our parameters:           Intercept = {theta[0][0]:.3f}, Slope = {theta[1][0]:.3f}")

# Check if they're close (they should be!)
tolerance = 0.1
if (abs(lin_reg.intercept_[0] - theta[0][0]) < tolerance and 
    abs(lin_reg.coef_[0][0] - theta[1][0]) < tolerance):
    print("✅ Success! Our implementation matches scikit-learn closely.")
else:
    print("❌ Results differ significantly. Check the implementation.")

print("\n=== Script Complete ===")