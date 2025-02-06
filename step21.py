import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly.subplots as sp
import requests
import pickle

class StockDataFetcher:
    def fetch_stock_data(self, symbol, period='1y'):
        """Fetches historical stock data from Yahoo Finance."""
        data = yf.download(symbol, period=period)
        return data
    
    def fetch_live_data(self, symbol, api_key):
        """Fetches live stock data using Alpha Vantage API."""
        try:
            url = 'https://www.alphavantage.co/query'
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "apikey": api_key,
                "outputsize": "compact" # or "full" for more data
            }
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.json()

            if "Time Series (Daily)" in data:
                # Convert the data to a pandas DataFrame
                df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
                df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)  # Sort by date
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                df = df.astype(float)  # Convert columns to numeric

                return df
            else:
                print(f"Error fetching live data for {symbol}: {data.get('Error Message', 'No error message provided')}")
                return pd.DataFrame()
        except requests.exceptions.RequestException as e:
            print(f"Request failed for {symbol}: {e}")
            return pd.DataFrame()
        except (ValueError, KeyError) as e:
            print(f"Error parsing live data for {symbol}: {e}")
            return pd.DataFrame()


    def combine_data(self, historical_data, live_data):
        """Combines historical and live data, handling duplicates."""
        if historical_data.empty:
            return live_data
        if live_data.empty:
            return historical_data

        combined_data = pd.concat([historical_data, live_data])
        combined_data = combined_data[~combined_data.index.duplicated(keep='first')]  # Keep historical if duplicate
        return combined_data.sort_index()

class StockAnalysis:
    def __init__(self, data):
        self.data = data

    def preprocess_data(self):
        df = self.data.copy()
        df.index = pd.to_datetime(df.index)

        for col in df.select_dtypes(include=[np.number]).columns:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            upper_limit, lower_limit = q3 + 1.5 * iqr, q1 - 1.5 * iqr
            df[col] = df[col].clip(lower_limit, upper_limit)

        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
        scaler = MinMaxScaler()
        df[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])
        return df, scaler
class Indicators:
        def calculate_rsi(data, period=14):
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(period).mean()
            avg_loss = loss.rolling(period).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            data['RSI'] = rsi
            return data
        def calculate_macd(data):
            short_ema = data['Close'].ewm(span=12, adjust=False).mean()
            long_ema = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = short_ema - long_ema
            data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
            return data

        def recommend_stock_action(data):
            data = Indicators.calculate_macd(data)
            data = Indicators.calculate_rsi(data)


            data['SMA_200'] = data['Close'].rolling(window=200).mean()

            latest_rsi = data['RSI'].iloc[-1]
            latest_macd = data['MACD'].iloc[-1]
            latest_signal = data['Signal_Line'].iloc[-1]
            latest_sma50 = data['SMA_50'].iloc[-1]
            latest_sma200 = data['SMA_200'].iloc[-1]
            latest_ema50 = data['EMA_50'].iloc[-1]
            latest_ema200 = data['EMA_200'].iloc[-1]

            buy_signals, sell_signals = [], []

            if latest_sma50 > latest_sma200:
                buy_signals.append("SMA_50 above SMA_200 (Golden Cross)")
            elif latest_sma50 < latest_sma200:
                sell_signals.append("SMA_50 below SMA_200 (Death Cross)")

            if latest_ema50 > latest_ema200:
                buy_signals.append("EMA_50 above EMA_200 (Golden Cross)")
            elif latest_ema50 < latest_ema200:
                sell_signals.append("EMA_50 below EMA_200 (Death Cross)")

            if latest_rsi < 30:
                buy_signals.append("RSI below 30 (Oversold)")
            elif latest_rsi > 70:
                sell_signals.append("RSI above 70 (Overbought)")

            if latest_macd > latest_signal:
                buy_signals.append("MACD above Signal Line")
            elif latest_macd < latest_signal:
                sell_signals.append("MACD below Signal Line")

            if len(buy_signals) > len(sell_signals):
                return f"**Recommendation: BUY** 📈\nReasons: {', '.join(buy_signals)}"
            elif len(sell_signals) > len(buy_signals):
                return f"**Recommendation: SELL** 📉\nReasons: {', '.join(sell_signals)}"
            else:
                return "**Recommendation: HOLD** 🤔\nMarket is neutral or mixed signals."

class StockModelTrainer:
    def train_ml_model(self, df):
        X, y = df[['Open', 'High', 'Low', 'Volume']], df['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        # Open the file in binary write mode ('wb') to save the pickled model
        with open ('ml_model.pkl', 'wb') as f:  
            pickle.dump(model, f)

        print("Random Forest Model - Accuracy Metrics:")
        print(f"R-squared: {r2 * 100:.2f}%")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"Mean Absolute Percentage Error: {mape * 100:.2f}%")
        return model

    def train_xgb_model(self, df):
        X, y = df[['Open', 'High', 'Low', 'Volume']], df['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        with open ('xgb_model.pkl', 'wb') as f:
            pickle.dump(model, f)

        print("XGBoost Model - Accuracy Metrics:")
        print(f"R-squared: {r2 * 100:.2f}%")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"Mean Absolute Percentage Error: {mape * 100:.2f}%")
        return model

    def train_dl_model(self, df):
        X = df[['Open', 'High', 'Low', 'Volume']].values.reshape(df.shape[0], 1, 4)
        y = df['Close'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=(1, 4)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=1)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        model.save("Lstm_model.h5")
        print("LSTM Model - Accuracy Metrics:")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"Mean Absolute Percentage Error: {mape * 100:.2f}%")
        return model

class Predictions:
    def predict_current_price(self, data, ml_model, xgb_model, dl_model, scaler):
        latest_data = data[['Open', 'High', 'Low', 'Volume']].iloc[-1].values.reshape(1, -1)

        # ML Prediction
        ml_prediction = ml_model.predict(latest_data)[0]

        # XGB Prediction
        xgb_prediction = xgb_model.predict(latest_data)[0]

        # DL Prediction
        latest_data_dl = np.reshape(latest_data, (latest_data.shape[0], 1, latest_data.shape[1]))
        dl_prediction = dl_model.predict(latest_data_dl)[0][0]

        # Reverse Scaling
        original_min = scaler.data_min_[3]  # 'Close' column min
        original_max = scaler.data_max_[3]  # 'Close' column max

        ml_actual_price = ml_prediction * (original_max - original_min) + original_min
        xgb_actual_price = xgb_prediction * (original_max - original_min) + original_min
        dl_actual_price = dl_prediction * (original_max - original_min) + original_min

        print(f"ML Prediction: {ml_actual_price}")
        print(f"XGB Prediction: {xgb_actual_price}")
        print(f"DL Prediction: {dl_actual_price}")


        aggregated_price = (ml_actual_price + xgb_actual_price + dl_actual_price) / 3
        print(f"Aggregated Predicted Price: {aggregated_price}")

        return ml_actual_price, xgb_actual_price, dl_actual_price,aggregated_price
class StockVisualization:

    def generate_charts(self,data, symbol):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
                  x=data.index,
                  y=data['Close'],
                  mode='lines',
                  name='Close Price',
                  line=dict(color='blue')
              ))

              # Update the layout of the plot
        fig.update_layout(
                  title=f'{symbol} Stock Price',
                  xaxis_title='Date',
                  yaxis_title='Close Price',
                  template='plotly_dark',  # Optional: choose a plotly theme (you can customize it)
              )

              # Show the plot
        fig.show()

    # Create subplots: 2 rows, 2 columns
        fig = sp.make_subplots(rows=2, cols=2,
                              subplot_titles=(f'{symbol} Stock Price with SMA',
                                              f'{symbol} Stock Price with EMA',
                                              'Relative Strength Index',
                                              'MACD Indicator'),
                              vertical_spacing=0.2)

        # Close Price and SMA (50 & 200)
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode='lines', name='50-Day SMA', line=dict(color='red')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_200'], mode='lines', name='200-Day SMA', line=dict(color='green')), row=1, col=1)

        # EMA (50 & 200)
        fig.add_trace(go.Scatter(x=data.index, y=data['EMA_50'], mode='lines', name='50-Day EMA', line=dict(color='purple')), row=1, col=2)
        fig.add_trace(go.Scatter(x=data.index, y=data['EMA_200'], mode='lines', name='200-Day EMA', line=dict(color='orange')), row=1, col=2)

        # RSI with Overbought (70) and Oversold (30) Levels
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', line=dict(color='brown')), row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=[70]*len(data), mode='lines', name='Overbought (70)', line=dict(color='red', dash='dash')), row=2, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=[30]*len(data), mode='lines', name='Oversold (30)', line=dict(color='green', dash='dash')), row=2, col=1)

        # MACD and Signal Line (if MACD is present)
        if 'MACD' in data.columns and 'Signal_Line' in data.columns:
            fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD', line=dict(color='blue')), row=2, col=2)
            fig.add_trace(go.Scatter(x=data.index, y=data['Signal_Line'], mode='lines', name='Signal Line', line=dict(color='red', dash='dash')), row=2, col=2)

        # Update layout
        fig.update_layout(title=f'{symbol} Stock Analysis',
                          xaxis_title='Date',
                          yaxis_title='Price',
                          height=1200, width=1500,
                          showlegend=True)

        # Show the interactive plot
        fig.show()

def main():
    symbols = ['SBIN.NS']  # Example symbol
    trainer = StockModelTrainer()
    visualizer = StockVisualization()
    fetcher = StockDataFetcher()

    for symbol in symbols:
        data = fetcher.fetch_stock_data(symbol)

        if not data.empty:
            analysis = StockAnalysis(data)
            df, scaler = analysis.preprocess_data()

            # Train models and evaluate them
            ml_model = trainer.train_ml_model(df)
            xgb_model = trainer.train_xgb_model(df)
            dl_model = trainer.train_dl_model(df)

            # Predict current prices
            predictions = Predictions()
            ml_actual_price, xgb_actual_price, dl_actual_price, aggregated_price = predictions.predict_current_price(df, ml_model, xgb_model, dl_model, scaler)
            # Visualize charts
            df = Indicators.calculate_rsi(df)
            df = Indicators.calculate_macd(df)
            visualizer.generate_charts(df, symbol)
            macd,recommend=Indicators.calculate_macd(df),Indicators.recommend_stock_action(df)

            # Print results
            print(f"Predicted Prices for {symbol}:")
            print(f"ML Model Predicted Price: {ml_actual_price}")
            print(f"XGB Model Predicted Price: {xgb_actual_price}")
            print(f"DL Model Predicted Price: {dl_actual_price}")

            print(f"Aggregate predication Price:{aggregated_price}")
            print(f"MACD: {macd}")
            print(f"Recommendation: {recommend}")
            print("\n" + "-"*50)

if __name__ == "__main__":
    main()



