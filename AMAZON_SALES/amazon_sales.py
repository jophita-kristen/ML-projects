import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score
)
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("amazon_sales_dataset.csv")
print(data.head())

X = data.drop(columns=["total_revenue", "order_id", "order_date"])
y = data["total_revenue"]

categorical_cols = ["product_category", "customer_region", "payment_method"]
numeric_cols = [
    "price",
    "discount_percent",
    "quantity_sold",
    "rating",
    "review_count",
    "discounted_price"
]

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded_cat = encoder.fit_transform(X[categorical_cols])

encoded_cat_df = pd.DataFrame(
    encoded_cat,
    columns=encoder.get_feature_names_out(categorical_cols)
)

numeric_df = X[numeric_cols].reset_index(drop=True)
X_final = pd.concat([numeric_df, encoded_cat_df], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n[REGRESSION METRICS]")
print("-------------------")
print("MAE  :", mean_absolute_error(y_test, y_pred))
print("MSE  :", mean_squared_error(y_test, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2   :", r2_score(y_test, y_pred))

#graph1:Actual vs Predicted
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Actual vs Predicted Revenue")
plt.grid()
plt.show()

#graph2:Residual Distribution
residuals = y_test - y_pred
plt.figure(figsize=(6,4))
sns.histplot(residuals, kde=True)
plt.xlabel("Residual Error")
plt.title("Residual Distribution")
plt.show()

#graph3:Residuals vs Predicted
plt.figure(figsize=(6,4))
plt.scatter(y_pred, residuals)
plt.axhline(0)
plt.xlabel("Predicted Revenue")
plt.ylabel("Residual Error")
plt.title("Residuals vs Predicted")
plt.grid()
plt.show()

#graph4:Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[-10:]
plt.figure(figsize=(6,4))
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), X_final.columns[indices])
plt.title("Top 10 Feature Importances")
plt.show()

#graph5:Predicted Revenue Distribution
plt.figure(figsize=(6,4))
sns.histplot(y_pred, kde=True)
plt.xlabel("Predicted Revenue")
plt.title("Predicted Revenue Distribution")
plt.show()

#graph6:Quantity Sold vs Revenue (EDA)
plt.figure(figsize=(6,4))
plt.scatter(data["quantity_sold"], data["total_revenue"])
plt.xlabel("Quantity Sold")
plt.ylabel("Total Revenue")
plt.title("Quantity Sold vs Revenue")
plt.grid()
plt.show()

#graph7:Average Revenue by Product Category
plt.figure(figsize=(7,4))
data.groupby("product_category")["total_revenue"].mean().plot(kind="bar")
plt.ylabel("Average Revenue")
plt.title("Average Revenue by Product Category")
plt.show()

#graph8:confusion matrix
threshold = y.median()
y_test_class = (y_test >= threshold).astype(int)
y_pred_class = (y_pred >= threshold).astype(int)
cm = confusion_matrix(y_test_class, y_pred_class)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (High vs Low Revenue)")
plt.show()

print("\n[CLASSIFICATION VIEW METRICS]")
print("Accuracy :", accuracy_score(y_test_class, y_pred_class))
print("Precision:", precision_score(y_test_class, y_pred_class))
print("Recall   :", recall_score(y_test_class, y_pred_class))

#Test Data Prediction
test_data = pd.DataFrame({
    "price": [1500],
    "discount_percent": [20],
    "quantity_sold": [3],
    "rating": [4.5],
    "review_count": [250],
    "discounted_price": [1200],
    "product_category": ["Electronics"],
    "customer_region": ["Asia"],
    "payment_method": ["UPI"]
})

encoded_test_cat = encoder.transform(test_data[categorical_cols])
encoded_test_cat_df = pd.DataFrame(
    encoded_test_cat,
    columns=encoder.get_feature_names_out(categorical_cols)
)
test_numeric = test_data[numeric_cols].reset_index(drop=True)
test_final = pd.concat([test_numeric, encoded_test_cat_df], axis=1)
test_final = test_final.reindex(columns=X_final.columns, fill_value=0)
predicted_revenue = model.predict(test_final)
print("\n[CUSTOM TEST RESULT]")
print("Predicted Total Revenue: ₹", round(predicted_revenue[0], 2))