import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(42)
x = np.random.rand(100)
y = np.array(['Class1' if i <= 0.5 else 'Class2' for i in x[:50]])
X_train, y_train = x[:50].reshape(-1, 1), y
X_test = x[50:].reshape(-1, 1)

for k in [1, 2, 3, 5, 20, 30]:
    model = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"\nResults for k={k}:\n")
    for i, (val, p) in enumerate(zip(x[50:], preds), start=51):
        print(f"x{i}={val:.6f}, predicted class={p}")

plt.figure(figsize=(10, 6))
colors = ['blue' if c == 'Class1' else 'red' for c in y_train]
plt.scatter(X_train, [1]*50, c=colors, label='Training')
plt.scatter(X_test, [0]*50, c='green', label='Test')
plt.axvline(0.5, color='gray', linestyle='--', label='Boundary x=0.5')
plt.yticks([])
plt.xlabel("x values")
plt.title("Training and Test Points")
plt.legend(loc='upper center')
plt.grid(True)
plt.show()
