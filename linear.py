# simple_linear_regression.py
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data: y = 2*x + 1 + some noise
np.random.seed(42) # For reproducibility
X = 2 * np.random.rand(100, 1) # 100 random numbers between 0 and 2
y = 1 + 2 * X + np.random.randn(100, 1) # y = 1 + 2x + noise

# Plot and save the data
plt.scatter(X, y)
plt.xlabel('X (Feature)')
plt.ylabel('y (Target)')
plt.title('Synthetic Data for Linear Regression')
plt.savefig('synthetic_data.png')
plt.close()