from scipy.integrate import odeint
import numpy as np



A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
addition = A + B
multiplication = np.dot(A, B)  # linalg.multi_dot for multiple matrices
transpose_A = A.T
print("A + B:\n", addition)
print("A * B:\n", multiplication)
print("A^T:\n", transpose_A)