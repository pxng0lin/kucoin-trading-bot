# PART OF MY NOTADEV SERIES

## !! Disclaimer !!
This bot is provided for **educational purposes only**. I am not a financial advisor or an expert in trading. The bot and its functionality should be used with caution, and **I take no responsibility for any financial losses that may occur as a result of using this bot**. It is important to thoroughly understand the risks involved in trading before using any automated trading software. **Use at your own risk**. Always consult with a qualified financial professional before making any investment decisions.

# Crypto Trading Bot with KuCoin Integration

This repository contains a comprehensive crypto trading bot that works with the **KuCoin** exchange, using **machine learning** to make buy and sell predictions based on historical market data. Additionally, a **portfolio summary** script is included to provide updates on recent trades and the current state of your portfolio.

## Features

### 1. **Trading Bot**:
   - **Buy and Sell Signals**: The bot predicts buy and sell signals using a trained **Random Forest Classifier**.
   - **Technical Indicators**: Uses technical indicators such as EMA, MACD, RSI, ATR, and Bollinger Bands for predictions.
   - **Automated Trading**: Executes buy/sell orders on KuCoin based on the model’s predictions.
   - **Telegram Notifications**: Sends updates and results to a specified Telegram chat.

### 2. **Model Training Script**:
   - **Data Collection**: Fetches historical market data from KuCoin.
   - **Feature Engineering**: Computes various technical indicators used to train the model.
   - **Random Forest Model**: Trains a machine learning model based on the collected data.
   - **Dynamic Data Range**: Attempts to collect up to 5 years of historical data, reducing the range if data is unavailable.

### 3. **Summary Script**:
   - **Transaction Summary**: Fetches and displays the last 10 transactions for each trading pair.
   - **Profit/Loss Calculation**: Calculates profit or loss for each pair using a **First-In, First-Out (FIFO)** approach.
   - **Portfolio Summary**: Shows the current balance and status of your portfolio.
   - **Telegram Notifications**: Sends the summary report to a Telegram chat.

## Installation Instructions

### 1. **Clone the Repository**:
```bash
https://github.com/pxng0lin/kucoin-trading-bot.git
cd kucoin-trading-bot
```

### 2. **Install Required Dependencies**:
```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:
```txt
ccxt
python-dotenv
telegram
pandas
asyncio
numpy
scikit-learn
joblib
talib
```

### 3. **Setup `.env` File**:
Create a `.env` file in the root of your project and provide the following environment variables:

```env
KUCOIN_API_KEY=your_kucoin_api_key
KUCOIN_SECRET_KEY=your_kucoin_secret_key
KUCOIN_PASSWORD=your_kucoin_password
TG_BOT_SUGI=your_telegram_bot_api_key
TG_ID=your_telegram_chat_id
PAIRS="" e.g. "AVAX/USDT,BNB/USDT"
```

### Environment Variables Explanation:
- `KUCOIN_API_KEY`, `KUCOIN_SECRET_KEY`, `KUCOIN_PASSWORD`: Your KuCoin API credentials.
- `TG_BOT`: Your Telegram bot API key.
- `TG_ID`: The chat ID where the bot will send its updates.
- `PAIRS`: A comma-separated list of trading pairs you want to trade (e.g., `AVAX/USDT`, `SOL/USDT`, `BNB/USDT`).

### 4. **Run the Scripts**

#### Trading Bot:
The bot uses a trained model to execute buy and sell orders based on market conditions.

```bash
python trading_bot.py
```

#### Model Training Script:
To fetch historical data and train the machine learning model:
```bash
python model_training.py
```

The model will be saved as `random_forest_model.pkl` and will be used by the trading bot.

#### Summary Script:
To get a portfolio summary and profit/loss report:
```bash
python summary_script.py
```

This script will send a formatted portfolio summary and recent trade performance to your Telegram.

## Script Breakdown

### 1. **Trading Bot (`trading_bot.py`)**

This script automates the trading process on KuCoin using a machine learning model that predicts buy and sell signals based on technical indicators.

**Key Features**:
- **Automated Trading**: Buys and sells tokens automatically based on the model’s predictions.
- **Dynamic Allocation**: Splits your USDT balance across multiple tokens and executes trades accordingly.
- **Technical Analysis**: Uses EMA, MACD, RSI, Bollinger Bands, ATR, and VWAP to generate prediction features.
- **Model Integration**: Uses a trained Random Forest model to predict the probability of price movements.
- **Telegram Alerts**: Notifies you of trades and results via Telegram.

**How It Works**:
- The bot fetches the latest market data for each trading pair specified in the `.env` file.
- It calculates various technical indicators and passes them to the machine learning model.
- Based on the prediction probabilities, it decides whether to execute a buy or sell order.

### 2. **Model Training Script (`model_training.py`)**

This script fetches historical market data for the specified trading pairs and trains a machine learning model using technical indicators.

**Key Features**:
- **Data Collection**: Fetches historical OHLCV data (open, high, low, close, volume) from KuCoin.
- **Technical Indicators**: Computes a variety of indicators (EMA, MACD, RSI, ATR, etc.) to use as model features.
- **Training**: Trains a Random Forest model to predict future price movements.
- **Data Management**: Attempts to fetch up to 5 years of data, adjusting the date range if data is unavailable.

**How It Works**:
- Fetches historical data for the trading pairs in the `.env` file.
- Trains a Random Forest classifier using the technical indicators as features.
- Saves the trained model to `random_forest_model.pkl` for future use by the trading bot.

### 3. **Summary Script (`summary_script.py`)**

This script summarizes recent trades and portfolio status, sending the report via Telegram.

**Key Features**:
- **Transaction Summary**: Fetches the last 10 transactions for each trading pair and calculates profit or loss.
- **Portfolio Summary**: Displays the current balance and holdings for all traded tokens.
- **Telegram Alerts**: Sends the portfolio summary to your specified Telegram chat.

**How It Works**:
- Fetches the last 10 transactions for each pair and computes the profit/loss using a FIFO approach.
- Fetches the current token balances from KuCoin.
- Sends a summary of the profit/loss and token balances to your Telegram chat.

## Conclusion

This crypto trading bot is a comprehensive solution for trading on KuCoin using machine learning. The model training script enables you to train your own model based on historical market data, while the trading bot uses this model to make live predictions and trade accordingly. Additionally, the summary script provides a detailed report of your portfolio and recent trades.

With the included Telegram integration, you’ll be able to monitor all trading activities and portfolio updates in real-time.
