import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from custom_regression_model import CustomRegression
from sklearn.preprocessing import StandardScaler


class InitializationClass:
    def __init__(self):
        self.new_house = pd.DataFrame({
            "sqft": [2200], "bedrooms": [3], "bathrooms": [3],
            "age_years": [10], "garage_spaces": [2], "lot_size_sqft": [6000],
            "distance_to_city_km": [8.5], "school_rating": [7]
        })


class LinearRegressionModel:
    def __init__(self):
        self.billing_data = pd.read_csv('housing-data.csv')
        self.sqft = self.billing_data['sqft']
        self.price = self.billing_data['price']

    def startModel(self):
        X = np.array(self.sqft).reshape(-1, 1)
        y = np.array(self.price)
        print(np.array(self.sqft).shape)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 6. Evaluate
        print("R²:", r2_score(y_test, y_pred))
        print("RMSE:", root_mean_squared_error(y_test, y_pred))
        # 7. Predict a new house
        new_house = pd.DataFrame([1600])
        print("Predicted price:", model.predict(new_house)[0])


class MultipleLinearRegression:
    def __init__(self):
        self.data = pd.read_csv('housing-data.csv')
        X = self.data.drop(columns=['price'])
        y = self.data['price']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        model_r2 = r2_score(y_test, y_pred)
        model_rmse = root_mean_squared_error(y_test, y_pred)
        print(
            f"\nLinear Refression Multiple Model  -> R²: {model_r2:.4f} | RMSE: ${model_rmse:,.2f} | ${model.predict(InitializationClass().new_house)}")


class CustomLinearRegression:
    def __init__(self):
        self.data = pd.read_csv('housing-data.csv')
        X = self.data.drop(columns=['price'])
        y = self.data['price']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        CustomRegressionModel = CustomRegression(
            learning_rate=0.01, iterations=1500)
        CustomRegressionModel.fit(X_train_scaled, y_train)
        my_r2 = CustomRegressionModel.score(X_test_scaled, y_test)
        my_pred = CustomRegressionModel.predict(X_test_scaled)
        my_rmse = np.sqrt(np.mean((y_test - my_pred) ** 2))
        new_house_scaled = scaler.transform(InitializationClass().new_house)

        print(
            f"\nMy Model  -> R²: {my_r2:.4f} | RMSE: ${my_rmse:,.2f} | ${CustomRegressionModel.predict(new_house_scaled)}")
