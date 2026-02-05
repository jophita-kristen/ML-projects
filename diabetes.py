import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("diabetes.csv")
print(df)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

train_x, test_x, train_y, test_y = train_test_split(
    X, y, test_size=0.3, random_state=1, shuffle=True
)

scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

print("[INFO] Train Shape :", train_x.shape)
print("[INFO] Test Shape  :", test_x.shape)

class BinaryLogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.betas = None
        self.losses = []

    def concat_ones(self, x):
        ones = np.ones((x.shape[0], 1))
        return np.concatenate((ones, x), axis=1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def log_loss(self, y, y_hat):
        epsilon = 1e-9
        return -np.mean(
            y * np.log(y_hat + epsilon) +
            (1 - y) * np.log(1 - y_hat + epsilon)
        )

    def fit(self, x, y):
        x = self.concat_ones(x)
        y = y.reshape(-1, 1)

        self.betas = np.zeros((x.shape[1], 1))

        for _ in range(self.epochs):
            z = np.dot(x, self.betas)
            y_hat = self.sigmoid(z)

            loss = self.log_loss(y, y_hat)
            self.losses.append(loss)

            gradient = np.dot(x.T, (y_hat - y)) / len(y)
            self.betas -= self.lr * gradient

    def predict_proba(self, x):
        x = self.concat_ones(x)
        return self.sigmoid(np.dot(x, self.betas)).ravel()

    def predict(self, x, threshold=0.4):
        return (self.predict_proba(x) >= threshold).astype(int)

blr = BinaryLogisticRegression(lr=0.01, epochs=1500)
blr.fit(train_x, train_y)

#loss curve
plt.figure(figsize=(5, 4))
plt.plot(blr.losses)
plt.xlabel("Epochs")
plt.ylabel("Log Loss")
plt.title("Training Loss Curve")
plt.grid()
plt.show()

train_pred = blr.predict(train_x)
train_prob = blr.predict_proba(train_x)

def confusion_matrix(actual, predicted):
    actual = actual.astype(int)
    predicted = predicted.astype(int)
    cm = np.zeros((2, 2), dtype=int)
    for i in range(len(actual)):
        cm[actual[i], predicted[i]] += 1
    return cm

cm = confusion_matrix(train_y, train_pred)

#model confusion matrix
plt.figure(figsize=(3, 3))
sbn.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0.5, 1.5], ["Normal", "Abnormal"])
plt.yticks([0.5, 1.5], ["Normal", "Abnormal"])
plt.title("Confusion Matrix")
plt.show()

tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) != 0 else 0
recall = tp / (tp + fn) if (tp + fn) != 0 else 0

print("[INFO] Accuracy  ::", accuracy)
print("[INFO] Precision ::", precision)
print("[INFO] Recall    ::", recall)

test_pred = blr.predict(test_x)
test_prob = blr.predict_proba(test_x)
test_cm = confusion_matrix(test_y, test_pred)

#test confusion matrix
plt.figure(figsize=(3, 3))
sbn.heatmap(test_cm, annot=True, fmt="d", cmap="Greens", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0.5, 1.5], ["Normal", "Abnormal"])
plt.yticks([0.5, 1.5], ["Normal", "Abnormal"])
plt.title("Test Confusion Matrix")
plt.show()

tn, fp, fn, tp = test_cm.ravel()
test_accuracy = (tp + tn) / (tp + tn + fp + fn)
test_precision = tp / (tp + fp) if (tp + fp) != 0 else 0
test_recall = tp / (tp + fn) if (tp + fn) != 0 else 0

print("\n[TEST DATA RESULTS]")
print("[INFO] Accuracy  ::", test_accuracy)
print("[INFO] Precision ::", test_precision)
print("[INFO] Recall    ::", test_recall)

# Random Person Diabetes Prediction
# [Pregnancies, Glucose, BloodPressure, SkinThickness,Insulin, BMI, DiabetesPedigreeFunction, Age]
random_person = np.array([[3, 120, 70, 25, 80, 28.5, 0.45, 35]])

random_person_scaled = scaler.transform(random_person)
random_prob = blr.predict_proba(random_person_scaled)[0]
random_pred = blr.predict(random_person_scaled)[0]

print("\n[RANDOM PERSON TEST RESULT]")
print("Input Data       :", random_person)
print("Predicted Prob.  :", round(random_prob, 4))

if random_pred == 1:
    print("Prediction: Person HAS Diabetes")
else:
    print("Prediction: Person does NOT have Diabetes")