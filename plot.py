import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

# Fetching historical stock data using yfinance (replace with your data if needed)
symbol = 'AAPL'
start_date = '2023-01-01'
end_date = '2023-12-31'
data = yf.download(symbol, start=start_date, end=end_date)

# Generate actual vs. simulated predicted data
y_actual = data['Close'].values  # Actual stock closing prices
np.random.seed(42)  # For reproducibility
y_predicted = y_actual + np.random.normal(0, np.std(y_actual) * 0.05, len(y_actual))  # Simulated predictions

# Plot Actual vs. Predicted stock prices
plt.figure(figsize=(14, 7))
plt.plot(data.index, y_actual, label='Actual Stock Prices', color='blue')
plt.plot(data.index, y_predicted, label='Predicted Stock Prices', color='red', linestyle='--')
plt.title(f'Predicted vs. Actual Stock Prices for {symbol}')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.show()

# Histogram: Distribution of prediction errors
prediction_errors = y_actual - y_predicted  # Calculate errors
plt.figure(figsize=(10, 5))
plt.hist(prediction_errors, bins=30, color='purple', alpha=0.7)
plt.title(f'Distribution of Prediction Errors for RELIANCE')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# Simulated data for actual and predicted stock prices
y_actual = np.random.uniform(100, 200, 100)  # Replace with real data
y_predicted = y_actual + np.random.normal(0, 15, 100)  # Simulated predictions with noise

# Line graph: Predicted vs. Actual Stock Prices
plt.figure(figsize=(14, 7))
plt.plot(y_actual, label='Actual Stock Prices', color='blue')
plt.plot(y_predicted, label='Predicted Stock Prices', color='red', linestyle='--')
plt.title('Predicted vs. Actual Stock Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.show()

# Histogram: Distribution of Prediction Errors
prediction_errors = y_actual - y_predicted  # Calculate errors
plt.figure(figsize=(10, 5))
sns.histplot(prediction_errors, kde=True, bins=30, color='purple')
plt.title('Distribution of Prediction Errors')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Example data for error analysis by time periods
time_periods = ['Short-Term', 'Medium-Term', 'Long-Term']
mae_values = [45.10, 50.57, 54.13]  # Replace with calculated MAE values

# Line graph: MAE over Time Periods
plt.figure(figsize=(12, 6))
plt.plot(time_periods, mae_values, marker='o', linestyle='-', color='green')
plt.title('MAE over Time Periods')
plt.xlabel('Time Period')
plt.ylabel('Mean Absolute Error (MAE)')
plt.grid(True)
plt.show()

# Generate data for computational efficiency comparison
models = ['LSTM', 'ARIMA', 'Random Forest', 'MLP']
mae_scores = [51.2,53.85,52.4,54.3]  # Replace with actual values
rmse_scores = [75.35,78.50,76.25,79.15]  # Replace with actual values

# Bar charts: Comparative Performance for MAE and RMSE
fig, ax = plt.subplots(1, 2, figsize=(14, 7))
sns.barplot(x=models, y=mae_scores, ax=ax[0], palette='Blues_d')
ax[0].set_title('MAE Comparison Across Models')
ax[0].set_ylabel('Mean Absolute Error (MAE)')
ax[0].set_xlabel('Model')

sns.barplot(x=models, y=rmse_scores, ax=ax[1], palette='Reds_d')
ax[1].set_title('RMSE Comparison Across Models')
ax[1].set_ylabel('Root Mean Squared Error (RMSE)')
ax[1].set_xlabel('Model')

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Generating fake feature importance scores
features = ['Closing Price', 'Moving Average (50)', 'RSI', 'Trading Volume', 'MACD']
importance_scores = [0.75, 0.025, 0.025, 0.175, 0.025]  # Simulated importance scores

# Create a DataFrame for plotting
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance Score': importance_scores})
feature_importance_df.sort_values(by='Importance Score', ascending=False, inplace=True)

# Plot the feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance Score', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importance for LSTM Model')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

# Simulated model performance metrics
model_names = ['LSTM', 'Random Forest', 'MLP', 'ARIMA']
mae_scores = [51.20, 52.75, 54.10, 53.50]  # Simulated MAE values near 50
rmse_scores = [75.35, 76.80, 78.60, 77.90]  # Simulated RMSE values near 75

# Plot for MAE comparison
plt.figure(figsize=(12, 6))
sns.barplot(x=model_names, y=mae_scores, palette='Blues_d')
plt.title(' MAE Comparison Across Models')
plt.ylabel('Mean Absolute Error (MAE)')
plt.xlabel('Model')
plt.show()

# Plot for RMSE comparison
plt.figure(figsize=(12, 6))
sns.barplot(x=model_names, y=rmse_scores, palette='Reds_d')
plt.title('RMSE Comparison Across Models')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.xlabel('Model')
plt.show()
