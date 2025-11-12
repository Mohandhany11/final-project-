import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("boston.csv")

print("✅ Data Loaded Successfully!")
print(data.head())


print("\n🔍 Dataset Info:")
print(data.info())
print("\n📊 Summary Statistics:")
print(data.describe())


X = data.drop(columns=['MEDV'])
y = data['MEDV']


plt.figure(figsize=(6,4))
plt.scatter(data['RM'], data['MEDV'], color='blue')
plt.title("Relationship between Average Rooms (RM) and House Price (MEDV)")
plt.xlabel("Average Number of Rooms (RM)")
plt.ylabel("Median Value of Homes (MEDV)")
plt.show()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\n✅ Data split into training and testing sets.")


model = LinearRegression()
model.fit(X_train, y_train)
print("\n🤖 Model trained successfully!")


y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📈 Model Evaluation Results:")
print("Mean Squared Error (MSE):", mse)
print("R-Square (R²):", r2)


plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred, color='green')
plt.title("Actual vs Predicted House Prices")
plt.xlabel("Actual Prices (MEDV)")
plt.ylabel("Predicted Prices (MEDV)")
plt.show()


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

