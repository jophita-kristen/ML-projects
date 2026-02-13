import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv(r"BREAST CANCER\Breast_Cancer.csv")
print(df.head())

# Alive = 0 , Dead = 1
le = LabelEncoder()
df["Status"] = le.fit_transform(df["Status"])

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop("Status", axis=1).values
y = df["Status"].values

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=1, shuffle=True)

scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

print("[INFO] Train Shape :", train_x.shape)
print("[INFO] Test Shape  :", test_x.shape)

class BinaryLogisticRegression:
    def __init__(self, lr=0.01, epochs=1500):
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

    def predict(self, x, threshold=0.5):
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

#confusion matrix
def confusion_matrix(actual, predicted):
    actual = actual.astype(int)
    predicted = predicted.astype(int)
    cm = np.zeros((2, 2), dtype=int)
    for i in range(len(actual)):
        cm[actual[i], predicted[i]] += 1
    return cm

train_pred = blr.predict(train_x)
train_cm = confusion_matrix(train_y, train_pred)

plt.figure(figsize=(3, 3))
sbn.heatmap(train_cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Train Confusion Matrix")
plt.show()

tn, fp, fn, tp = train_cm.ravel()
train_accuracy = (tp + tn) / (tp + tn + fp + fn)
train_precision = tp / (tp + fp) if (tp + fp) != 0 else 0
train_recall = tp / (tp + fn) if (tp + fn) != 0 else 0

print("[TRAIN RESULTS]")
print("Accuracy  ::", train_accuracy)
print("Precision ::", train_precision)
print("Recall    ::", train_recall)

test_pred = blr.predict(test_x)
test_cm = confusion_matrix(test_y, test_pred)

plt.figure(figsize=(3, 3))
sbn.heatmap(test_cm, annot=True, fmt="d", cmap="Greens", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Test Confusion Matrix")
plt.show()

tn, fp, fn, tp = test_cm.ravel()
test_accuracy = (tp + tn) / (tp + tn + fp + fn)
test_precision = tp / (tp + fp) if (tp + fp) != 0 else 0
test_recall = tp / (tp + fn) if (tp + fn) != 0 else 0

print("\n[TEST RESULTS]")
print("Accuracy  ::", test_accuracy)
print("Precision ::", test_precision)
print("Recall    ::", test_recall)

#random patient prediction
random_patient = np.array([train_x[0]])  # sample patient

random_prob = blr.predict_proba(random_patient)[0]
random_pred = blr.predict(random_patient)[0]

print("\n[RANDOM PATIENT RESULT]")
print("Predicted Probability :", round(random_prob, 4))

if random_pred == 1:
    print("Prediction: Patient is likely to DIE")
else:
    print("Prediction: Patient is likely to SURVIVE")