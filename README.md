
# Trading Bot for KuCoin Exchange

This is a trading bot for the KuCoin exchange that automatically detects buy and sell signals using machine learning models and technical indicators. The bot monitors selected cryptocurrency pairs, analyzes historical and real-time data, and executes trades based on a combination of prediction probabilities and market conditions.

## Features

- **Machine Learning-Based Predictions**: The bot uses a trained Random Forest classifier to predict buy or sell signals based on historical data and real-time market analysis.
- **Technical Indicators**: The bot leverages key technical indicators like EMA (50 and 200), MACD, RSI, ATR, PSAR, Bollinger Bands, and VWAP to generate and confirm signals.
- **Confluence Approach**: Signals are generated when a confluence of indicators aligns with the prediction probabilities from the model. This ensures a higher level of confidence in each trade decision.
- **Automated Trading**: When a buy or sell signal is detected, the bot places market orders on KuCoin using available USDT or token balances.
- **Balance Management**: The bot allocates available USDT to different trading pairs based on the tokens with zero balance. This ensures efficient use of available capital.
- **Telegram Notifications**: The bot sends real-time updates and notifications via Telegram for every buy, sell, or decision to skip a trade due to existing balances.

## Logic Overview

1. **Data Collection**: 
   - The bot fetches historical and real-time OHLCV (open, high, low, close, volume) data for each trading pair.
   - This data is used to calculate various technical indicators.

2. **Model Prediction**:
   - A Random Forest model, trained on historical data, generates probabilities for buy and sell signals.
   - The model uses features such as EMA, MACD, RSI, PSAR, ATR, VWAP, and Bollinger Bands.

3. **Signal Generation**:
   - If the model predicts a buy probability above a certain threshold (e.g., 0.65) and the token has no existing balance, the bot triggers a buy order.
   - If the model predicts a sell probability above a certain threshold and the token has a balance, the bot triggers a sell order.
   - No trade is made if the bot detects that a signal exists but the conditions for a buy or sell are not met (e.g., token already held or insufficient USDT).

4. **Execution**:
   - The bot executes market buy/sell orders through the KuCoin exchange using `ccxt`.

## Installation

To install and run the bot, follow these steps:

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/kucoin-trading-bot.git
cd kucoin-trading-bot
```

### 2. Set up environment variables

Create a `.env` file in the root directory of the project with the following content:

```plaintext
KUCOIN_API_KEY=your_kucoin_api_key
KUCOIN_SECRET_KEY=your_kucoin_secret_key
KUCOIN_PASSWORD=your_kucoin_password
TG_BOT=your_telegram_bot_api_key
TG_ID=your_telegram_chat_id
PAIRS="PAIR1/USDT,PAIR2/USDT..."  # Comma-separated list of trading pairs
```

Make sure to replace the values with your actual KuCoin API credentials and Telegram bot credentials.

### 3. Install dependencies

The project requires Python 3.9 or above. Install the necessary packages using the following command:

```bash
pip install -r requirements.txt
```

Here is a list of the required packages:

- `ccxt`: A cryptocurrency trading library to interact with KuCoin and other exchanges.
- `pandas`: For data manipulation and analysis.
- `talib`: A library for technical analysis indicators.
- `python-telegram-bot`: To send notifications via Telegram.
- `joblib`: For saving and loading the trained machine learning model.
- `scikit-learn`: For training and predicting using the Random Forest classifier.
- `python-dotenv`: For managing environment variables from a `.env` file.
- `numpy`: For numerical computations.

### 4. Running the Bot

Once the environment is set up and dependencies are installed, you can start the bot using:

```bash
python trading_bot.py
```

The bot will load the trained machine learning model (`random_forest_model.pkl`) and start monitoring the specified trading pairs. If the model does not exist, it will automatically train a new model using historical data.

### 5. Training a New Model (Optional)

If you need to retrain the model manually, run the model training script:

```bash
python model_script.py
```

This script will fetch historical data for each pair, calculate technical indicators, and train the Random Forest model. The trained model will be saved to `random_forest_model.pkl`.
