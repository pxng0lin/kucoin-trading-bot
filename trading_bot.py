import ccxt
import os
import pandas as pd
import talib
from dotenv import load_dotenv
import numpy as np
import asyncio
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import telegram
import math  # For rounding down the spend

# Load environment variables from .env file
load_dotenv()

# Initialize the KuCoin exchange
exchange = ccxt.kucoin({
    'apiKey': os.getenv('KUCOIN_API_KEY'),
    'secret': os.getenv('KUCOIN_SECRET_KEY'),
    'password': os.getenv('KUCOIN_PASSWORD'),
})

tg_api = os.getenv('TG_BOT_SUGI')
tg_channel = os.getenv('TG_ID')
tg_bot = telegram.Bot(tg_api)

PAIRS = os.getenv('PAIRS').replace('"', '').split(',')

# Feature columns used for training and prediction
FEATURE_COLUMNS = ['ema50', 'ema200', 'macd', 'macdsignal', 'rsi', 'psar', 'atr', 'bollinger_upper', 'bollinger_lower', 'vwap']

# Function to send Telegram message
async def send_telegram_message(bot, message, chat_id, emoji):
    try:
        tgb_msg = f"{emoji} {message}"
        await bot.send_message(chat_id=chat_id, text=tgb_msg)
        print(f"Telegram message sent: {tgb_msg}")
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")

# Fetch the minimum order size for a specific pair
def get_min_order_size(pair):
    market_info = exchange.load_markets()
    if pair in market_info:
        return market_info[pair]['limits']['amount']['min']
    return None

# Fetch historical data for training the model
def fetch_historical_data(pair, timeframe='1h', since=None):
    try:
        # Fetch historical OHLCV data
        ohlcv = exchange.fetch_ohlcv(pair, timeframe=timeframe, since=since, limit=500)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Error fetching historical data for {pair}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

# Fetch the latest data for live predictions
def fetch_latest_data(pair, timeframe='1h'):
    try:
        # Fetch the latest OHLCV data
        ohlcv = exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=500)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Error fetching latest data for {pair}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

# Buy and Sell Orders Logic with minimum order size check
async def create_order(pair, side, usdt_amount=None, token_amount=None):
    try:
        print(f"Attempting to create {side} order for {pair}")

        # Fetch the current price of the token
        ticker = exchange.fetch_ticker(pair)
        current_price = ticker['last']

        if current_price is None:
            print(f"Error: Could not fetch current price for {pair}")
            await send_telegram_message(tg_bot, f"Error: Could not fetch current price for {pair}.", tg_channel, "❌")
            return

        print(f"Current price for {pair}: {current_price}")

        if side == 'buy':
            if usdt_amount is None:
                print("Error: No USDT amount specified for the buy order.")
                return

            # Calculate the amount of token we can buy
            token_amount = usdt_amount / current_price
            print(f"Calculated token amount for {pair}: {token_amount} (based on {usdt_amount} USDT)")

            # Check for minimum order size
            min_order_size = get_min_order_size(pair)
            if min_order_size and token_amount < min_order_size:
                print(f"Order amount {token_amount} is below the minimum order size of {min_order_size} for {pair}")
                await send_telegram_message(tg_bot, f"Order amount {token_amount} below minimum size for {pair}. No order created.", tg_channel, "⚠️")
                return

            # Create a market buy order
            order = exchange.create_order(pair, 'market', 'buy', token_amount)
            print(f"Buy order created: {order}")
            await send_telegram_message(tg_bot, f"Buy order created for {pair}: {token_amount} (worth {usdt_amount} USDT)", tg_channel, "🔵")

        elif side == 'sell':
            if token_amount is None or token_amount <= 0:
                print("Error: No token amount specified for the sell order.")
                return

            print(f"Selling entire balance for {pair}: {token_amount}")

            # Check for minimum order size
            min_order_size = get_min_order_size(pair)
            if min_order_size and token_amount < min_order_size:
                print(f"Order amount {token_amount} is below the minimum order size of {min_order_size} for {pair}")
                await send_telegram_message(tg_bot, f"Order amount {token_amount} below minimum size for {pair}. No order created.", tg_channel, "⚠️")
                return

            # Create a market sell order
            order = exchange.create_order(pair, 'market', 'sell', token_amount)
            print(f"Sell order created: {order}")
            await send_telegram_message(tg_bot, f"Sell order created for {pair}: {token_amount} tokens", tg_channel, "🟢")

    except Exception as e:
        print(f"Order creation failed: {e}")
        await send_telegram_message(tg_bot, f"Order creation failed: {e}", tg_channel, "❌")

# Load the trained Random Forest model, or train it if no model exists
def load_model():
    try:
        model = joblib.load("random_forest_model.pkl")
        print("Model loaded from file.")
        return model
    except FileNotFoundError:
        print("Model file not found. Training a new model.")
        return train_and_save_model()

# Train the model using historical data
def train_and_save_model():
    historical_data = fetch_historical_data()  # Fetch historical data using real API
    X, y = prepare_data(historical_data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    print("Model trained.")
    joblib.dump(model, "random_forest_model.pkl")
    print("Model saved to random_forest_model.pkl.")

    return model

# Prepare features and target for training the model
def prepare_data(df):
    df['ema50'] = talib.EMA(df['close'], timeperiod=50)
    df['ema200'] = talib.EMA(df['close'], timeperiod=200)
    df['macd'], df['macdsignal'], _ = talib.MACD(df['close'])
    df['rsi'] = talib.RSI(df['close'])
    df['psar'] = talib.SAR(df['high'], df['low'])
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    df['bollinger_upper'], _, df['bollinger_lower'] = talib.BBANDS(df['close'], timeperiod=20)

    df['target'] = df['close'].shift(-1) > df['close']
    df['target'] = df['target'].apply(lambda x: 1 if x else -1)
    df = df.dropna()

    X = df[FEATURE_COLUMNS]
    y = df['target']
    return X, y

# Make predictions using the trained model
def make_prediction_with_confluence(model, latest_data):
    features = {'ema50': None, 'ema200': None, 'macd': None, 'macdsignal': None, 'rsi': None, 'psar': None, 'atr': None, 'bollinger_upper': None, 'bollinger_lower': None, 'vwap': None}

    if len(latest_data['close']) >= 50:
        ema50 = talib.EMA(latest_data['close'], timeperiod=50)
        if len(ema50) > 0:
            features['ema50'] = ema50.iloc[-1]

    if len(latest_data['close']) >= 200:
        ema200 = talib.EMA(latest_data['close'], timeperiod=200)
        if len(ema200) > 0:
            features['ema200'] = ema200.iloc[-1]

    if len(latest_data['close']) >= 26:
        macd, macdsignal, _ = talib.MACD(latest_data['close'])
        if len(macd) > 0 and len(macdsignal) > 0:
            features['macd'] = macd.iloc[-1]
            features['macdsignal'] = macdsignal.iloc[-1]

    if len(latest_data['close']) >= 14:
        rsi = talib.RSI(latest_data['close'])
        if len(rsi) > 0:
            features['rsi'] = rsi.iloc[-1]

    if len(latest_data['high']) >= 2 and len(latest_data['low']) >= 2:
        psar = talib.SAR(latest_data['high'], latest_data['low'])
        if len(psar) > 0:
            features['psar'] = psar.iloc[-1]

    if len(latest_data['high']) >= 14:
        atr = talib.ATR(latest_data['high'], latest_data['low'], latest_data['close'], timeperiod=14)
        if len(atr) > 0:
            features['atr'] = atr.iloc[-1]

    if len(latest_data['volume']) >= 1:
        vwap = (latest_data['volume'] * (latest_data['high'] + latest_data['low'] + latest_data['close']) / 3).cumsum() / latest_data['volume'].cumsum()
        features['vwap'] = vwap.iloc[-1]

    if len(latest_data['close']) >= 20:
        bollinger_upper, _, bollinger_lower = talib.BBANDS(latest_data['close'], timeperiod=20)
        if len(bollinger_upper) > 0 and len(bollinger_lower) > 0:
            features['bollinger_upper'] = bollinger_upper.iloc[-1]
            features['bollinger_lower'] = bollinger_lower.iloc[-1]

    features_df = pd.DataFrame([features])
    features_df.columns = FEATURE_COLUMNS
    features_df = features_df.fillna(0)

    prediction_proba = model.predict_proba(features_df)
    buy_proba = prediction_proba[0][1]
    sell_proba = prediction_proba[0][0]

    return buy_proba, sell_proba, features_df

# Main logic loop to handle all tokens
async def process_data(pair, model, allocated_amount_per_token):
    if not pair.endswith('/USDT'):
        print(f"Skipping {pair} because it doesn't trade with USDT.")
        await send_telegram_message(tg_bot, f"Skipping {pair}: Not a USDT pair.", tg_channel, "⚠️")
        return

    latest_data = fetch_latest_data(pair)
    await send_telegram_message(tg_bot, f"Fetching data for {pair}...", tg_channel, "📊")

    buy_proba, sell_proba, _ = make_prediction_with_confluence(model, latest_data)

    token = pair.split('/')[0]
    available_token_balance = exchange.fetch_balance().get('free', {}).get(token, 0)

    spend = allocated_amount_per_token.get(pair, 0)

    print(f"Prediction probabilities: Sell: {sell_proba:.2f}, Buy: {buy_proba:.2f}")

    if buy_proba > 0.65 and spend > 0 and available_token_balance == 0:
        print(f"Buy signal detected based on model prediction.")
        await send_telegram_message(tg_bot, f"Buy signal detected for {pair}.", tg_channel, "🔵")
        await create_order(pair, 'buy', spend)
    elif sell_proba > 0.65 and available_token_balance > 0:
        print(f"Sell signal detected based on model prediction.")
        await send_telegram_message(tg_bot, f"Sell signal detected for {pair}.", tg_channel, "🟢")
        await create_order(pair, 'sell', token_amount=available_token_balance)
    else:
        print(f"No clear signal for {pair}")
        await send_telegram_message(tg_bot, f"No clear signal for {pair}.", tg_channel, "🔘")

# Main loop to handle each pair
async def main():
    model = load_model()

    try:
        available_usdt_balance = exchange.fetch_balance()['USDT']['free']
    except Exception as e:
        print(f"Error fetching USDT balance: {e}")
        available_usdt_balance = 0

    print(f"Total available USDT: {available_usdt_balance}")
    await send_telegram_message(tg_bot, f"Bot starting with {available_usdt_balance} USDT available.", tg_channel, "🚀")

    tokens_with_zero_balance = []
    allocated_amount_per_token = {}

    for pair in PAIRS:
        token = pair.split('/')[0]
        try:
            token_balance = exchange.fetch_balance().get('free', {}).get(token, 0)
        except Exception as e:
            print(f"Error fetching balance for {token}: {e}")
            token_balance = 0

        if token_balance == 0:
            tokens_with_zero_balance.append(pair)

    if tokens_with_zero_balance:
        allocated_amount_per_token = {
            pair: math.floor(available_usdt_balance / len(tokens_with_zero_balance)) for pair in tokens_with_zero_balance
        }

    print(f"Allocated amounts per token: {allocated_amount_per_token}")

    for pair in PAIRS:
        print(f"Starting process for {pair}...")
        await send_telegram_message(tg_bot, f"Processing {pair}...", tg_channel, "🔄")
        await process_data(pair, model, allocated_amount_per_token)
        await send_telegram_message(tg_bot, f"Process complete for {pair}.", tg_channel, "✅")

# Run the main loop asynchronously
if __name__ == "__main__":
    asyncio.run(main())
