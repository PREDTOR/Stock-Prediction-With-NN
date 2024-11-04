# Stock-Prediction-With-NN
In this code I attempt to predict closing prices for 2 days in future using Models with neural networks. 

# -*- coding: utf-8 -*-
"""
Original file is located at
https://colab.research.google.com/drive/1VWdz1Mfd6qRdtq2t7dVzch6slCSirxRy

"""

#Imports and Setup
import pandas as pd
import numpy as np
from google.colab import files
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, SimpleRNN, Dropout, Dense,Reshape, Activation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import requests
import zipfile
import os

#saving files in df named as ticker for that perticular stock,
# We stored all the stocks in a new file named selected stocks for easy uploading
AAPL = pd.read_csv('aapl.us.txt')
AAPL.head()

#uploaded = files.upload()
INTC = pd.read_csv('intc.us.txt')
INTC.head()

NVDA = pd.read_csv('nvda.us.txt')
NVDA.head()

# Then read the file
TSLA = pd.read_csv('tsla.us.txt')
TSLA.head()

# Then read the file
MSFT = pd.read_csv('msft.us.txt')
MSFT.head()

# Then read the file
COKE = pd.read_csv('coke.us.txt')
COKE.head()

#uploaded = files.upload()

# Then read the file
WMT = pd.read_csv('wmt.us.txt')
WMT.head()

#uploaded = files.upload()

# Then read the file
GOOG = pd.read_csv('goog.us.txt')
GOOG.head()

# Store each DataFrame in a dictionary for easy access
stocks_data = {
    'NVDA': NVDA,
    'INTC': INTC,
    'TSLA': TSLA,
    'AAPL': AAPL,
    'MSFT': MSFT,
    'COKE': COKE,
    'WMT': WMT,
    'GOOG': GOOG
}

"""# Preprocessing Data"""

def prepare_data(df, look_back=10, future_days=2):
    """Prepares input and label arrays for time sequence prediction."""
    X, y = [], []
    prices = df['Close'].values
    for i in range(len(prices) - look_back - future_days):
        X.append(prices[i: i + look_back])
        y.append(prices[i + look_back + future_days - 1])
    return np.array(X), np.array(y)

# Set look-back period
look_back = 10

# Prepare datasets for training, validation, and testing
train_stocks = ['NVDA', 'INTC', 'TSLA', 'AAPL']
val_stocks = ['MSFT', 'COKE']
test_stocks = ['WMT', 'GOOG']

# Plot closing prices with dates for each stock
for stock in train_stocks + val_stocks + test_stocks:
    df = globals()[stock]
    df['Date'] = pd.to_datetime(df['Date'])  # Convert date to datetime format
    plt.figure(figsize=(10, 4))
    plt.plot(df['Date'], df['Close'], label=f"{stock} Closing Prices")
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title(f'{stock} Closing Price Over Time')
    plt.legend()
    plt.show()

# Prepare data for each stock in the train, validation, and test sets
train_X, train_y = zip(*(prepare_data(globals()[ticker]) for ticker in train_stocks))
val_X, val_y = zip(*(prepare_data(globals()[ticker]) for ticker in val_stocks))
test_X, test_y = zip(*(prepare_data(globals()[ticker]) for ticker in test_stocks))

# Concatenate data from different stocks into single arrays for training, validation, and testing
train_X, train_y = np.concatenate(train_X), np.concatenate(train_y)
val_X, val_y = np.concatenate(val_X), np.concatenate(val_y)
test_X, test_y = np.concatenate(test_X), np.concatenate(test_y)

# Print shapes to confirm
print("Train X shape:", train_X.shape, "Train y shape:", train_y.shape)
print("Validation X shape:", val_X.shape, "Validation y shape:", val_y.shape)
print("Test X shape:", test_X.shape, "Test y shape:", test_y.shape)

"""# Scaling data"""

# Scaling the data (carefully done to avoid information leakage)
scaler = MinMaxScaler()
train_X = scaler.fit_transform(train_X.reshape(-1, train_X.shape[1])).reshape(train_X.shape[0], train_X.shape[1], 1)
val_X = scaler.transform(val_X.reshape(-1, val_X.shape[1])).reshape(val_X.shape[0], val_X.shape[1], 1)
test_X = scaler.transform(test_X.reshape(-1, test_X.shape[1])).reshape(test_X.shape[0], test_X.shape[1], 1)

"""# Modelling"""

# Define a function to create a Sequential model with configurable layers and parameters
def create_model(conv_filters, rnn_units, dropout_rate):
    model = Sequential([
        Conv1D(filters=conv_filters, kernel_size=3, activation='relu', input_shape=(train_X.shape[1], 1)),
        SimpleRNN(units=rnn_units),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# List of different model configurations to experiment with
configs = [
    {'conv_filters': 32, 'rnn_units': 32, 'dropout_rate': 0.3},
    {'conv_filters': 64, 'rnn_units': 64, 'dropout_rate': 0.4},
    {'conv_filters': 128, 'rnn_units': 64, 'dropout_rate': 0.2}
]

models = []
training_losses = []
validation_losses = []

# Training each model, recording losses
for i, config in enumerate(configs):
    print(f"Training Model {i+1} with Config: {config}")
    model = create_model(**config)
    history = model.fit(train_X, train_y, epochs=10, batch_size=32, verbose=1)
    models.append(model)

    # Record training loss
    training_loss = history.history['loss'][-1]
    training_losses.append(training_loss)

    # Validation predictions and loss
    val_predictions = model.predict(val_X)
    val_loss = mean_squared_error(val_y, val_predictions)
    validation_losses.append(val_loss)

    # Plotting validation predictions
    plt.figure(figsize=(10, 4))
    plt.plot(val_y, label='True Prices')
    plt.plot(val_predictions, label='Predicted Prices')
    plt.title(f'Model {i+1} - Validation Predictions')
    plt.xlabel('Index')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    print(f"Model {i+1} - Training Loss: {training_loss:.4f}, Validation Loss: {val_loss:.4f}")

"""# Selecting best model"""

# Identifying the best model based on the lowest validation loss
best_model_index = np.argmin(validation_losses)
best_model = models[best_model_index]
print(f"\nBest Model: Model {best_model_index+1} with Validation Loss: {validation_losses[best_model_index]:.4f}")

# Testing the best model
test_predictions = best_model.predict(test_X)
test_loss = mean_squared_error(test_y, test_predictions)
print(f"Test Loss for Best Model: {test_loss:.4f}")

# Calculate the MSE and RMSE for testing data
test_mse = mean_squared_error(test_y, test_predictions)
test_rmse = test_mse ** 0.5

print(f"Testing MSE: {test_mse:.4f}")
print(f"Testing RMSE: {test_rmse:.4f}")

"""# Validation & Testing"""

# Scatter plot comparing true and predicted test values
plt.figure(figsize=(10, 4))
plt.scatter(range(len(test_y)), test_y, label='True Prices', color='blue', s=10)
plt.scatter(range(len(test_predictions)), test_predictions, label='Predicted Prices', color='red', s=10)
plt.title('Test Predictions vs True Values Variation')
plt.xlabel('Index')
plt.ylabel('Price')
plt.legend()
plt.show()

# Plot actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.plot(test_y, label='True Price', color='blue')
plt.plot(test_predictions, label='Predicted Price', color='orange')
plt.xlabel('Time Steps')
plt.ylabel('Normalized Stock Price')
plt.title('True vs Predicted Prices on Testing Data')
plt.legend()
plt.show()

"""# Final Observation:
- after final evaluation of prediction and model prices predictions vs actual ones, the best model had few metrics to consider based on its final performance in both the training and validation. These choices were made by closely comparing MSE or mean squared error and RMSE or root mean squared error on both the cases.

- Using quantative metrics the chosen model : "Model 3 with Validation Loss: 4.4750", had the lowest validation RMSE and it shows that it generalizes well to unseen data predictions so, this model alanysis trends of stock prices effectively withot overfitting.
- Looking at it from naked eye shows that from the chosen model the predictions made on both the datas are very close to the actual price.

By looking at both the types of metrical and visual assesment we can say that the model 3 is the best model for the given case, With training loss of just 77.7 approx and val loss of 6.14 approx.

# Model Performance for 2 day closing prices:
- The model that is selected for these task show that it is capable of pridicting closing prices two days into the future at a remarkable rate with high accuracy, Achieving a RMSE of 131 approx, Showing variation in stock predictions error, Showing the model is able to predict a general sense of prices directions and general trends, This doest mean it can capture short term High volatile times but, It can predict general trends.

In other words, this model can serve as a very useful indicator for price forcasting and price movements over 2 day span. Most of the "predictions made by this model is relible in normal market situations except hig volatile times".
"""
