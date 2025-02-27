import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# Lists to store data
dates = []
prices = []

# Function to read data from CSV
def get_data(filename):
    with open(filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip header

        for row in csv_reader:
            try:
                date_obj = datetime.strptime(row[0], "%m/%d/%Y")  # Convert MM/DD/YYYY to datetime
                dates.append(date_obj.toordinal())  # Convert date to numerical format
                prices.append(float(row[1]))  # Convert price to float
            except ValueError as e:
                print(f"Skipping row due to error: {row} -> {e}")  # Handle bad data gracefully

# Function to train models and predict prices
def predict_prices(dates, prices, future_date):
    dates = np.array(dates).reshape(-1, 1)
    prices = np.array(prices).reshape(-1, 1)

    # Scale the date values
    date_scaler = MinMaxScaler()
    dates_scaled = date_scaler.fit_transform(dates)  # Normalize dates between 0 and 1

    # Scale price values
    price_scaler = MinMaxScaler()
    prices_scaled = price_scaler.fit_transform(prices).flatten()

    # Train Support Vector Regression (SVR) models
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=100, degree=2)  # Lower degree to avoid overfitting

    svr_rbf.fit(dates_scaled, prices_scaled)
    svr_lin.fit(dates_scaled, prices_scaled)
    svr_poly.fit(dates_scaled, prices_scaled)

    # Train a Linear Regression model for comparison
    lin_reg = LinearRegression()
    lin_reg.fit(dates_scaled, prices_scaled)

    # Convert future_date to scaled format
    future_date_scaled = date_scaler.transform(np.array([[future_date]]))

    # Predict the price for the given future date
    predicted_rbf_scaled = svr_rbf.predict(future_date_scaled)[0]
    predicted_lin_scaled = svr_lin.predict(future_date_scaled)[0]
    predicted_poly_scaled = svr_poly.predict(future_date_scaled)[0]

    # Convert predictions back to original scale
    predicted_rbf = price_scaler.inverse_transform([[predicted_rbf_scaled]])[0][0]
    predicted_lin = price_scaler.inverse_transform([[predicted_lin_scaled]])[0][0]
    predicted_poly = price_scaler.inverse_transform([[predicted_poly_scaled]])[0][0]

    # Plot data and models
    plt.figure(figsize=(10, 5))
    plt.scatter(dates_scaled, prices_scaled, color='black', label='Data')
    plt.plot(dates_scaled, svr_rbf.predict(dates_scaled), color='red', label='RBF Model')
    plt.plot(dates_scaled, svr_lin.predict(dates_scaled), color='green', label='Linear Model')
    plt.plot(dates_scaled, svr_poly.predict(dates_scaled), color='blue', label='Polynomial Model')
    plt.xlabel('Normalized Date')
    plt.ylabel('Stock Price (Scaled)')
    plt.title('Stock Price Prediction using SVR')
    plt.legend()
    plt.show()

    return predicted_rbf, predicted_lin, predicted_poly

# Load data from CSV
get_data('Zomato Stock Price History.csv')

# Predict stock price for a given future date
future_date = datetime.strptime("02/23/2025", "%m/%d/%Y").toordinal()  # Convert future date to ordinal
predicted_rbf, predicted_lin, predicted_poly = predict_prices(dates, prices, future_date)
print(f"Predicted Price (RBF): ${predicted_rbf:.2f}")
print(f"Predicted Price (Linear): ${predicted_lin:.2f}")
print(f"Predicted Price (Polynomial): ${predicted_poly:.2f}")
