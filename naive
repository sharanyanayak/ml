import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

X, y = fetch_olivetti_faces(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model = GaussianNB().fit(Xtr, ytr)
ypred = model.predict(Xte)

print(f"Accuracy: {accuracy_score(yte, ypred):.2f}")
print(classification_report(yte, ypred, target_names=[f"Class {i}" for i in np.unique(y)]))

idx = np.random.choice(len(Xte), 8, replace=False)
for i, j in enumerate(idx):
    plt.subplot(2, 4, i+1)
    plt.imshow(Xte[j].reshape(64, 64), cmap='gray')
    plt.title(f"T:{yte[j]}\nP:{ypred[j]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
