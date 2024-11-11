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
import numpy as np
from datetime import datetime, timedelta

class DataPreprocessor:
    """
    Section 4.2: Data Collection and Preprocessing
    """
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.scalers = {}

    def fetch_data(self):
        """
        Section 4.2.1: Data Sources
        Fetches stock data and macroeconomic indicators
        """
        # Fetch stock data
        stock_data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
        
        # Fetch macroeconomic data (using FRED API if available)
        try:
            from fredapi import Fred
            fred = Fred(api_key='YOUR_FRED_API_KEY')
            interest_rates = fred.get_series('DFF')  # Federal Funds Rate
            inflation = fred.get_series('CPIAUCSL')  # Consumer Price Index
            gdp = fred.get_series('GDP')  # Gross Domestic Product
            
            # Resample macro data to match stock data frequency
            macro_data = pd.DataFrame({
                'interest_rate': interest_rates,
                'inflation': inflation,
                'gdp': gdp
            }).resample('D').ffill()
            
            # Merge with stock data
            data = pd.merge(stock_data, macro_data, left_index=True, right_index=True, how='left')
        except:
            # If FRED API is not available, proceed with only stock data
            data = stock_data
            
        return data

    def calculate_technical_indicators(self, data):
        """
        Section 4.2.2: Feature Selection
        Calculates technical indicators
        """
        # Price-related features
        data['returns'] = data['Close'].pct_change()
        data['log_returns'] = np.log(data['Close']/data['Close'].shift(1))
        
        # Moving averages
        data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
        data['SMA_200'] = ta.trend.sma_indicator(data['Close'], window=200)
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(data['Close'])
        data['BB_upper'] = bb.bollinger_hband()
        data['BB_lower'] = bb.bollinger_lband()
        
        # RSI
        data['RSI'] = ta.momentum.rsi(data['Close'])
        
        # Volatility
        data['volatility'] = data['returns'].rolling(window=20).std()
        
        # MACD
        macd = ta.trend.MACD(data['Close'])
        data['MACD'] = macd.macd()
        data['MACD_signal'] = macd.macd_signal()
        
        return data

    def clean_data(self, data):
        """
        Section 4.2.3: Data Cleaning
        """
        # Handle missing values
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Remove outliers using z-score method
        for column in data.select_dtypes(include=[np.number]).columns:
            z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
            data[column] = data[column].mask(z_scores > 3, data[column].median())
            
        # Normalize data
        for column in data.select_dtypes(include=[np.number]).columns:
            self.scalers[column] = MinMaxScaler()
            data[column] = self.scalers[column].fit_transform(data[column].values.reshape(-1, 1))
            
        return data

class LSTMModel:
    """
    Section 4.3: Model Development
    """
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.model = None
        
    def build_model(self, input_shape):
        """
        Section 4.3.1: LSTM Architecture Design
        """
        model = Sequential([
            # First LSTM layer with dropout
            LSTM(units=100, return_sequences=True, 
                 input_shape=input_shape,
                 recurrent_dropout=0.2),
            Dropout(0.2),
            
            # Second LSTM layer
            LSTM(units=50, return_sequences=True,
                 recurrent_dropout=0.2),
            Dropout(0.2),
            
            # Third LSTM layer
            LSTM(units=25, recurrent_dropout=0.2),
            Dropout(0.2),
            
            # Dense output layer
            Dense(units=1, activation='linear')
        ])
        
        return model
        
    def compile_model(self, learning_rate=0.001):
        """
        Section 4.3.2: Hyperparameter Optimization
        """
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse')
        
class ComparativeModels:
    """
    Section 4.4: Comparative Models
    """
    def __init__(self):
        self.models = {}
        
    def build_models(self):
        """
        Initialize all comparative models
        """
        # Random Forest model
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # MLP model
        self.models['mlp'] = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            random_state=42
        )
        
        # ARIMA model will be built separately due to different data requirements
        
    def fit_arima(self, data):
        """
        Fit ARIMA model
        """
        model = sm.tsa.ARIMA(data, order=(5,1,0))
        return model.fit()

class ModelEvaluator:
    """
    Section 4.5: Evaluation Framework
    """
    def __init__(self):
        self.metrics = {}
        
    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate all evaluation metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }

def create_sequences(data, sequence_length):
    """
    Create sequences for LSTM training
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def main():
    """
    Main training pipeline
    """
    # Initialize parameters
    symbol = 'AAPL'
    start_date = '2018-01-01'
    end_date = '2024-01-01'
    sequence_length = 60
    validation_split = 0.2
    test_split = 0.1
    
    # 1. Data Preprocessing
    preprocessor = DataPreprocessor(symbol, start_date, end_date)
    raw_data = preprocessor.fetch_data()
    data_with_indicators = preprocessor.calculate_technical_indicators(raw_data)
    cleaned_data = preprocessor.clean_data(data_with_indicators)
    
    # Create sequences for LSTM
    X, y = create_sequences(cleaned_data.values, sequence_length)
    
    # Split data
    train_size = int(len(X) * (1 - test_split - validation_split))
    val_size = int(len(X) * validation_split)
    
    X_train = X[:train_size]
    X_val = X[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    
    y_train = y[:train_size]
    y_val = y[train_size:train_size+val_size]
    y_test = y[train_size+val_size:]
    
    # 2. Build and train LSTM model
    lstm_model = LSTMModel(sequence_length)
    model = lstm_model.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    lstm_model.model = model
    lstm_model.compile_model()
    
    # Training callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        'best_model.h5',
        monitor='val_loss',
        save_best_only=True
    )
    
    # Train LSTM
    history = lstm_model.model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    # 3. Train comparative models
    comparative = ComparativeModels()
    comparative.build_models()
    
    # Reshape data for traditional models
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    X_val_2d = X_val.reshape(X_val.shape[0], -1)
    X_test_2d = X_test.reshape(X_test.shape[0], -1)
    
    # Train each comparative model
    for name, model in comparative.models.items():
        model.fit(X_train_2d, y_train)
    
    # Fit ARIMA
    arima_model = comparative.fit_arima(cleaned_data['Close'])
    
    # 4. Evaluate all models
    evaluator = ModelEvaluator()
    
    # LSTM evaluation
    lstm_pred = lstm_model.model.predict(X_test)
    lstm_metrics = evaluator.calculate_metrics(y_test, lstm_pred)
    
    # Comparative models evaluation
    comparative_metrics = {}
    for name, model in comparative.models.items():
        predictions = model.predict(X_test_2d)
        comparative_metrics[name] = evaluator.calculate_metrics(y_test, predictions)
    
    # Print results
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
        'preprocessor': preprocessor,
        'history': history,
        'metrics': {
            'lstm': lstm_metrics,
            'comparative': comparative_metrics
        }
    }

if __name__ == "__main__":
    results = main()
