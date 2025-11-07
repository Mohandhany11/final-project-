# ------------------------------
# 🏠 HOUSE PRICE PREDICTION USING BOSTON DATA
# ------------------------------
# Tools: NumPy, Pandas, Matplotlib, Scikit-learn
# IDE: Cursor / Jupyter / VS Code
# Dataset: boston.csv
# ------------------------------

# Step 1️⃣: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2️⃣: Load the Dataset
# ---------------------------------
# تأكد إن ملف boston.csv موجود في نفس مجلد الكود
data = pd.read_csv("boston.csv")

print("✅ Data Loaded Successfully!")
print(data.head())

# Step 3️⃣: Explore the Data
# ---------------------------------
print("\n🔍 Dataset Info:")
print(data.info())
print("\n📊 Summary Statistics:")
print(data.describe())

# Step 4️⃣: Define Features and Target
# ---------------------------------
# عمود السعر اسمه عادة MEDV في Boston dataset (Median value of owner-occupied homes)
X = data.drop(columns=['MEDV'])
y = data['MEDV']

# Step 5️⃣: Visualize a Relationship
# ---------------------------------
plt.figure(figsize=(6,4))
plt.scatter(data['RM'], data['MEDV'], color='blue')
plt.title("Relationship between Average Rooms (RM) and House Price (MEDV)")
plt.xlabel("Average Number of Rooms (RM)")
plt.ylabel("Median Value of Homes (MEDV)")
plt.show()

# Step 6️⃣: Split the Data
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\n✅ Data split into training and testing sets.")

# Step 7️⃣: Train the Linear Regression Model
# ---------------------------------
model = LinearRegression()
model.fit(X_train, y_train)
print("\n🤖 Model trained successfully!")

# Step 8️⃣: Predict and Evaluate
# ---------------------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📈 Model Evaluation Results:")
print("Mean Squared Error (MSE):", mse)
print("R-Square (R²):", r2)

# Step 9️⃣: Visualization of Predictions
# ---------------------------------
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred, color='green')
plt.title("Actual vs Predicted House Prices")
plt.xlabel("Actual Prices (MEDV)")
plt.ylabel("Predicted Prices (MEDV)")
plt.show()

# Step 🔟: Summary
# ---------------------------------
print("\n📄 PROJECT SUMMARY:")
print("""
In this project, we used the classic Boston housing dataset to predict house prices.
We analyzed the relationship between different factors (like number of rooms, crime rate,
and property tax) and the median value of homes.

After training a Linear Regression model, we evaluated it using MSE and R² metrics.
A higher R² value indicates a better fit between predicted and actual prices.

✅ Conclusion:
Houses with more rooms (RM) tend to have higher prices (MEDV).
The model can be improved by adding regularization or using non-linear methods.
""")
