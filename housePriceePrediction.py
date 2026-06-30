# House Price Prediction using Linear Regression

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("Housing.csv")

# Display first 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# Dataset Information
print("\nDataset Information:")
print(df.info())

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Encode Categorical Columns
le = LabelEncoder()

categorical_columns = [
    'mainroad',
    'guestroom',
    'basement',
    'hotwaterheating',
    'airconditioning',
    'prefarea',
    'furnishingstatus'
]

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Features and Target
X = df.drop('price', axis=1)
y = df['price']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nModel Performance")
print("---------------------")
print("R2 Score :", round(r2, 4))
print("MSE      :", round(mse, 2))
print("RMSE     :", round(rmse, 2))

# Actual vs Predicted
comparison = pd.DataFrame({
    'Actual Price': y_test,
    'Predicted Price': y_pred
})

print("\nActual vs Predicted Prices")
print(comparison.head(10))

# Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")

plt.show()

# Example Prediction
# Format:
# area, bedrooms, bathrooms, stories,
# mainroad, guestroom, basement,
# hotwaterheating, airconditioning,
# parking, prefarea, furnishingstatus

new_house = pd.DataFrame(
    [[7420, 4, 2, 3, 1, 0, 0, 0, 1, 2, 1, 2]],
    columns=X.columns
)

predicted_price = model.predict(new_house)

print("\nPredicted House Price:")
print("₹", round(predicted_price[0], 2))