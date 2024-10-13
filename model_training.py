import ccxt
import os
import pandas as pd
from dotenv import load_dotenv
import datetime
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import talib
import asyncio

# Load environment variables from .env file
load_dotenv()

# Initialize the KuCoin exchange
exchange = ccxt.kucoin({
    'apiKey': os.getenv('KUCOIN_API_KEY'),
    'secret': os.getenv('KUCOIN_SECRET_KEY'),
    'password': os.getenv('KUCOIN_PASSWORD'),
    'enableRateLimit': True
})

# Constants
PAIRS = os.getenv('PAIRS').replace('"', '').split(',')
TIMEFRAME = '1h'  # Ensure we are fetching hourly data

# Function to fetch OHLCV data in batches and save to CSV
def fetch_historical_data(pair, timeframe, since, save_path):
    batch_size = 500
    all_ohlcv = []
    total_candles = 0
    last_timestamp = None
    repeated_candle_count = 0  # Track how many times we get the same candle

    while True:
        print(f"Fetching batch of {batch_size} candles for {pair} from {datetime.datetime.utcfromtimestamp(since / 1000)}")
        try:
            ohlcv_batch = exchange.fetch_ohlcv(pair, timeframe, since=since, limit=batch_size)
        except Exception as e:
            print(f"Error fetching candles for {pair}: {e}")
            break

        if not ohlcv_batch:
            print(f"No more candles fetched for {pair}. Exiting loop.")
            break

        # Track repeated candles to prevent an infinite loop if data is stuck
        if len(ohlcv_batch) == 1 and ohlcv_batch[0][0] == last_timestamp:
            repeated_candle_count += 1
            if repeated_candle_count >= 3:
                print(f"Repeated candle fetched 3 times for {pair}. Breaking loop.")
                break
        else:
            repeated_candle_count = 0  # Reset counter if new data is fetched
            last_timestamp = ohlcv_batch[-1][0]  # Update the last timestamp

        all_ohlcv += ohlcv_batch
        total_candles += len(ohlcv_batch)
        print(f"Fetched {len(ohlcv_batch)} candles. Total so far: {total_candles} candles.")

        # Stop if the last timestamp is close to the current time
        if last_timestamp and last_timestamp >= exchange.milliseconds():
            print(f"Reached the current time for {pair}. Stopping fetch.")
            break

        # Update 'since' to the last timestamp of the batch
        since = ohlcv_batch[-1][0] + 3600000  # Move to the next hour (3600000 ms = 1 hour)

    # Convert to DataFrame
    if all_ohlcv:
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Save to CSV
        csv_filename = f"{save_path}/{pair.replace('/', '_')}_historical_data.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Historical data saved to {csv_filename}")
        return df
    else:
        print(f"No historical data to save for {pair}.")
        return None  # Return None if no data is available

# Function to reduce the timeframe by 90 days if data is not available
def attempt_fetch_data(pair, timeframe, save_path):
    days = 5 * 365  # Start from 5 years (in days)
    while days >= 90:  # Attempt fetching until we reach a minimum of 90 days
        since = exchange.parse8601((datetime.datetime.now() - datetime.timedelta(days=days)).isoformat())
        df = fetch_historical_data(pair, timeframe, since, save_path)
        if df is not None and not df.empty:
            return df  # Return the fetched data if available
        else:
            print(f"Data not available for {days} days ago, reducing by 90 days and trying again...")
            days -= 90
    print(f"No data available for {pair} even after reducing the date range.")
    return None  # Return None if no data is available even after trying

# Prepare features and target for training the model
def prepare_data(df):
    df['ema50'] = talib.EMA(df['close'], timeperiod=50)
    df['ema200'] = talib.EMA(df['close'], timeperiod=200)
    df['macd'], df['macdsignal'], _ = talib.MACD(df['close'])
    df['rsi'] = talib.RSI(df['close'])
    df['psar'] = talib.SAR(df['high'], df['low'])
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'])

    # BBANDS returns three values: upper, middle, and lower bands
    df['bollinger_upper'], _, df['bollinger_lower'] = talib.BBANDS(df['close'], timeperiod=20)

    df['vwap'] = df['close']  # Replace this with the actual VWAP calculation if necessary

    # Create the target (buy/sell signal)
    df['target'] = df['close'].shift(-1) > df['close']
    df['target'] = df['target'].apply(lambda x: 1 if x else -1)
    df = df.dropna()

    # Include all the features used in the prediction
    X = df[['ema50', 'ema200', 'macd', 'macdsignal', 'rsi', 'psar', 'atr', 'bollinger_upper', 'bollinger_lower', 'vwap']]
    y = df['target']
    return X, y

# Train the model using aggregated historical data from multiple pairs
def train_and_save_model():
    save_path = "./historical_data"
    os.makedirs(save_path, exist_ok=True)

    all_data = []  # To store historical data from all pairs

    # Loop through all pairs
    for pair in PAIRS:
        print(f"Fetching data for training from pair: {pair}")
        historical_data = attempt_fetch_data(pair, TIMEFRAME, save_path)

        if historical_data is not None and not historical_data.empty:
            all_data.append(historical_data)
        else:
            print(f"No data available for {pair}.")

    # If we have data from at least one pair, continue with model training
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        X, y = prepare_data(combined_df)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        print("Model trained.")
        joblib.dump(model, "random_forest_model.pkl")
        print("Model saved to random_forest_model.pkl.")
        return model
    else:
        print(f"No data available for training.")
        return None

# Load the trained Random Forest model, or train it if no model exists
def load_model():
    try:
        model = joblib.load("random_forest_model.pkl")
        print("Model loaded from file.")
        return model
    except FileNotFoundError:
        print("Model file not found. Training a new model.")
        return train_and_save_model()

# Main function to fetch historical data for all pairs and train the model
async def main():
    model = load_model()
    if model is None:
        print("Failed to train or load the model.")
        return

    print("Model is ready for predictions or further use.")

if __name__ == "__main__":
    asyncio.run(main())
