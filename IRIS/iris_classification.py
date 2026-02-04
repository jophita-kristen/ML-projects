import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = pd.read_csv("IRIS/Iris.csv")

#encoding
label_map = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}
data["Species"] = data["Species"].map(label_map)

X = data.iloc[:, 1:-1].values
y = data["Species"].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

class LogisticRegressionMulti:
    def __init__(self, lr=0.05, epochs=2000):
        self.lr = lr
        self.epochs = epochs

    def softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)  #numerical stability
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]
        y_onehot = np.eye(3)[y]

        self.W = np.zeros((X.shape[1], 3))
        self.loss = []

        for _ in range(self.epochs):
            scores = np.dot(X, self.W)
            probs = self.softmax(scores)

            error = probs - y_onehot
            grad = np.dot(X.T, error) / X.shape[0]
            self.W -= self.lr * grad

            loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-9), axis=1))
            self.loss.append(loss)

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        scores = np.dot(X, self.W)
        probs = self.softmax(scores)
        return np.argmax(probs, axis=1)

model = LogisticRegressionMulti()
model.fit(x_train, y_train)

train_pred = model.predict(x_train)
test_pred = model.predict(x_test)

cm = confusion_matrix(y_test, test_pred)

plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

metrics = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Training": [
        accuracy_score(y_train, train_pred),
        precision_score(y_train, train_pred, average="macro"),
        recall_score(y_train, train_pred, average="macro"),
        f1_score(y_train, train_pred, average="macro")
    ],
    "Testing": [
        accuracy_score(y_test, test_pred),
        precision_score(y_test, test_pred, average="macro"),
        recall_score(y_test, test_pred, average="macro"),
        f1_score(y_test, test_pred, average="macro")
    ]
})

print("\nPerformance Metrics:\n")
print(metrics)

plt.plot(model.loss)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()