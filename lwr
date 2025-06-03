import numpy as np
import matplotlib.pyplot as plt

def lwr(x, X, y, tau):
    X = np.c_[np.ones(len(X)), X]
    x = np.array([1, x])
    W = np.diag(np.exp(-np.sum((X - x)**2, axis=1) / (2 * tau**2)))
    theta = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y
    return x @ theta

def plot_lwr(X, y, tau):
    X_test = np.linspace(X.min(), X.max(), 100)
    y_pred = np.array([lwr(x, X, y, tau) for x in X_test])
    plt.scatter(X, y, color='blue', label='Data Points')
    plt.plot(X_test, y_pred, color='red', label=f'LWR Curve (τ={tau})')
    plt.title(f'Locally Weighted Regression (τ={tau})')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

np.random.seed(42)
X = np.linspace(0, 10, 100)
y = np.sin(X) + np.random.normal(0, 0.2, size=X.shape)

for tau in [0.1, 0.5, 1.0, 5.0]:
    plot_lwr(X, y, tau)
