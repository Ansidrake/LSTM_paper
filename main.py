import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
import yfinance as yf
import ta
import os

class DataPreprocessor:
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.scalers = {}

    def fetch_data(self):
        stock_data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
        # Fetch macroeconomic data
        try:
            from fredapi import Fred
            fred = Fred(api_key='YOUR_FRED_API_KEY')
            interest_rates = fred.get_series('DFF')
            inflation = fred.get_series('CPIAUCSL')
            gdp = fred.get_series('GDP')
            
            macro_data = pd.DataFrame({
                'interest_rate': interest_rates,
                'inflation': inflation,
                'gdp': gdp
            }).resample('D').ffill()
            data = pd.merge(stock_data, macro_data, left_index=True, right_index=True, how='left')
        except:
            data = stock_data
            
        return data

    def calculate_technical_indicators(self, data):
        data['returns'] = data['Close'].pct_change()
        data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
        data['SMA_200'] = ta.trend.sma_indicator(data['Close'], window=200)
        bb = ta.volatility.BollingerBands(data['Close'])
        data['BB_upper'] = bb.bollinger_hband()
        data['BB_lower'] = bb.bollinger_lband()
        data['RSI'] = ta.momentum.rsi(data['Close'])
        data['volatility'] = data['returns'].rolling(window=20).std()
        macd = ta.trend.MACD(data['Close'])
        data['MACD'] = macd.macd()
        data['MACD_signal'] = macd.macd_signal()
        data['volume_ma'] = ta.volume.volume_weighted_average_price(
            data['High'], data['Low'], data['Close'], data['Volume']
        )
        data['MFI'] = ta.volume.money_flow_index(
            data['High'], data['Low'], data['Close'], data['Volume']
        )
        
        return data

    def clean_data(self, data):
        data = data.fillna(method='ffill').fillna(method='bfill')
        for column in data.select_dtypes(include=[np.number]).columns:
            z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
            data[column] = data[column].mask(z_scores > 3, data[column].median())
        for column in data.select_dtypes(include=[np.number]).columns:
            self.scalers[column] = MinMaxScaler()
            data[column] = self.scalers[column].fit_transform(data[column].values.reshape(-1, 1))
        return data

class LSTMModel:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.model = None
        
    def build_model(self, input_shape):
        model = Sequential([
            LSTM(units=100, return_sequences=True, input_shape=input_shape, recurrent_dropout=0.2, kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True, recurrent_dropout=0.2, kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            Dropout(0.2),
            LSTM(units=25, recurrent_dropout=0.2, kernel_regularizer=tf.keras.regularizers.L2(0.01)),
            Dropout(0.2),
            Dense(units=1, activation='linear')
        ])
        self.model = model

    def compile_model(self, learning_rate=0.001):
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

class ComparativeModels:
    def __init__(self):
        self.models = {}

    def build_models(self):
        self.models['random_forest'] = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        self.models['mlp'] = MLPRegressor(hidden_layer_sizes=(100, 50, 25), activation='relu', solver='adam', random_state=42, max_iter=1000)
        
    def fit_arima(self, data):
        model = sm.tsa.ARIMA(data, order=(5, 1, 0))
        return model.fit()

class ModelEvaluator:
    def calculate_metrics(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)
        
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def main():
    symbol = 'AAPL'
    start_date = '2018-01-01'
    end_date = '2024-01-01'
    sequence_length = 60
    validation_split = 0.2
    test_split = 0.1

    model_dir = 'saved_models'
    os.makedirs(model_dir, exist_ok=True)

    try:
        preprocessor = DataPreprocessor(symbol, start_date, end_date)
        raw_data = preprocessor.fetch_data()
        data_with_indicators = preprocessor.calculate_technical_indicators(raw_data)
        cleaned_data = preprocessor.clean_data(data_with_indicators)

        if cleaned_data.empty:
            raise ValueError("No data available for the specified date range")
            
        X, y = create_sequences(cleaned_data.values, sequence_length)
        train_size = int(len(X) * (1 - test_split - validation_split))
        val_size = int(len(X) * validation_split)
        
        X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
        y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]
        
        lstm_model = LSTMModel(sequence_length)
        lstm_model.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        lstm_model.compile_model()
        
        model_path = os.path.join(model_dir, 'best_model.keras')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
        model_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True, verbose=1)

        history = lstm_model.model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        comparative = ComparativeModels()
        comparative.build_models()
        
        X_train_2d, X_val_2d, X_test_2d = X_train.reshape(X_train.shape[0], -1), X_val.reshape(X_val.shape[0], -1), X_test.reshape(X_test.shape[0], -1)
        
        for name, model in comparative.models.items():
            model.fit(X_train_2d, y_train)
        
        # 4. Evaluate all models
        print("\nEvaluating models...")
        evaluator = ModelEvaluator()

        #lstm_pred = lstm_model.model.predict(X_test).flatten()  # Flatten to 1D
        # lstm_metrics = evaluator.calculate_metrics(y_test, lstm_pred)
        
        
        # LSTM evaluation
        print("Evaluating LSTM model...")
        lstm_pred = lstm_model.model.predict(X_test)
        
        # Reshape predictions if necessary to match y_test
        if lstm_pred.shape != y_test.shape:
            lstm_pred = lstm_pred.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)
        
        lstm_metrics = evaluator.calculate_metrics(y_test, lstm_pred)
                # Comparative models evaluation
        comparative_metrics = {}
        for name, model in comparative.models.items():
            try:
                print(f"Evaluating {name} model...")
                predictions = model.predict(X_test_2d)
                
                # Ensure predictions and y_test have compatible shapes
                if predictions.shape != y_test.shape:
                    predictions = predictions.reshape(-1, 1)
                
                comparative_metrics[name] = evaluator.calculate_metrics(y_test, predictions)
            except Exception as e:
                print(f"Error evaluating {name} model: {str(e)}")
                continue
        print("\nModel Evaluation Results:")
        print("\nLSTM Model Metrics:")
        for metric, value in lstm_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        for model_name, metrics in comparative_metrics.items():
            print(f"\n{model_name} Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
                
        return {
            'lstm_model': lstm_model,
            'comparative_models': comparative.models,
            'evaluator': evaluator,
            'preprocessor': preprocessor
        }

    except Exception as e:
        print(f"An error occurred: {e}")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Initialize data processor and fetch/preprocess data
data_processor = DataPreprocessor(symbol='AAPL', start_date='2020-01-01', end_date='2023-01-01')
raw_data = data_processor.fetch_data()
data = data_processor.calculate_technical_indicators(raw_data)
data = data_processor.clean_data(data)

# Split data into training and test sets
sequence_length = 60
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

# Prepare sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 3])  # Assuming 3 is the index of the target variable, 'Close' price.
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data.values, sequence_length)
X_test, y_test = create_sequences(test_data.values, sequence_length)

# Initialize and train the LSTM model
lstm_model = LSTMModel(sequence_length=sequence_length)
lstm_model.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
lstm_model.compile_model(learning_rate=0.001)

history = lstm_model.model.fit(
    X_train, y_train, validation_data=(X_test, y_test),
    epochs=50, batch_size=32, callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
)

# Predict with LSTM
y_pred_lstm = lstm_model.model.predict(X_test)

# Initialize comparative models and train them
comp_models = ComparativeModels()
comp_models.build_models()

# Random Forest
rf_model = comp_models.models['random_forest']
rf_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
y_pred_rf = rf_model.predict(X_test.reshape(X_test.shape[0], -1))

# MLP
mlp_model = comp_models.models['mlp']
mlp_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
y_pred_mlp = mlp_model.predict(X_test.reshape(X_test.shape[0], -1))

# ARIMA - Fitting on the target variable (e.g., closing price series)
arima_model = comp_models.fit_arima(train_data['Close'])
y_pred_arima = arima_model.forecast(steps=len(test_data))

# Evaluate models
evaluator = ModelEvaluator()
metrics_lstm = evaluator.calculate_metrics(y_test, y_pred_lstm.flatten())
metrics_rf = evaluator.calculate_metrics(y_test, y_pred_rf)
metrics_mlp = evaluator.calculate_metrics(y_test, y_pred_mlp)
metrics_arima = evaluator.calculate_metrics(y_test, y_pred_arima)

# Display metrics
print("LSTM Metrics:", metrics_lstm)
print("Random Forest Metrics:", metrics_rf)
print("MLP Metrics:", metrics_mlp)
print("ARIMA Metrics:", metrics_arima)

# Plot Predicted vs Actual for LSTM
plt.figure(figsize=(14, 7))
plt.plot(y_test, label="Actual Prices")
plt.plot(y_pred_lstm, label="LSTM Predictions")
plt.legend()
plt.title("LSTM Model - Predicted vs Actual Stock Prices")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.show()

# Error Histogram
plt.figure(figsize=(10, 5))
sns.histplot(y_test - y_pred_lstm.flatten(), kde=True, bins=30)
plt.title("LSTM Prediction Error Distribution")
plt.xlabel("Prediction Error")
plt.show()

# Comparative Performance Bar Chart
models = ["LSTM", "Random Forest", "MLP", "ARIMA"]
mae_scores = [metrics_lstm['MAE'], metrics_rf['MAE'], metrics_mlp['MAE'], metrics_arima['MAE']]
rmse_scores = [metrics_lstm['RMSE'], metrics_rf['RMSE'], metrics_mlp['RMSE'], metrics_arima['RMSE']]

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
sns.barplot(x=models, y=mae_scores, ax=ax[0])
ax[0].set_title("Mean Absolute Error (MAE) Comparison")
sns.barplot(x=models, y=rmse_scores, ax=ax[1])
ax[1].set_title("Root Mean Squared Error (RMSE) Comparison")
plt.show()

if __name__ == "__main__":
    results = main()