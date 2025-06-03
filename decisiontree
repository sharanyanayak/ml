import pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = load_breast_cancer()
X, y = pd.DataFrame(data.data, columns=data.feature_names), data.target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier().fit(Xtr, ytr)
ypred = clf.predict(Xte)

print(f'Accuracy: {accuracy_score(yte, ypred):.2f}')
print('Classification Report:\n', classification_report(yte, ypred))
print('Confusion Matrix:\n', confusion_matrix(yte, ypred))

plt.figure(figsize=(15, 7))
plot_tree(clf, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.show()

pred = clf.predict([X.iloc[0]])
print(f'Predicted class for new sample: {data.target_names[pred[0]]}')
