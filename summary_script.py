import ccxt
import os
import asyncio
from dotenv import load_dotenv
import telegram
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Initialize the KuCoin exchange
exchange = ccxt.kucoin({
    'apiKey': os.getenv('KUCOIN_API_KEY'),
    'secret': os.getenv('KUCOIN_SECRET_KEY'),
    'password': os.getenv('KUCOIN_PASSWORD'),
})

# Telegram bot setup (TG_BOT_SUGI)
tg_api = os.getenv('TG_BOT_SUGI')
tg_channel = os.getenv('TG_ID')
tg_bot = telegram.Bot(tg_api)

# Fetch the last 10 buy and sell transactions for a token pair
def fetch_last_transactions(pair, limit=10):
    trades = exchange.fetch_my_trades(pair, limit=limit)
    return trades

# Calculate profit or loss based on the last 10 transactions
def calculate_profit_loss(trades):
    total_profit_loss = 0
    buy_prices = []
    sell_prices = []

    for trade in trades:
        if trade['side'] == 'buy':
            buy_prices.append(trade['price'])
        elif trade['side'] == 'sell':
            sell_prices.append(trade['price'])

    # Match buys and sells (FIFO approach)
    for buy_price, sell_price in zip(buy_prices, sell_prices):
        profit_loss = sell_price - buy_price
        total_profit_loss += profit_loss

    return total_profit_loss

# Fetch the current balance for the traded tokens in the portfolio
def fetch_portfolio_summary(traded_tokens):
    balance = exchange.fetch_balance()
    token_portfolio = {}

    for token in traded_tokens:
        if balance['total'].get(token, 0) > 0:
            token_portfolio[token] = {
                'available': balance['free'][token],
                'total': balance['total'][token]
            }

    return token_portfolio

# Format the summary as a Telegram message
def format_summary_message(portfolio, profit_loss_summary):
    message = f"📊 Portfolio Summary as of {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}:\n"

    for token, info in portfolio.items():
        message += f"💼 {token}: {info['available']} available, {info['total']} total\n"

    message += "\n💰 Profit/Loss Summary (last 10 transactions):\n"
    for pair, profit_loss in profit_loss_summary.items():
        if profit_loss > 0:
            status = "Profit"
        elif profit_loss < 0:
            status = "Loss"
        else:
            status = "Even"
        message += f"🔸 {pair}: {status}: {profit_loss:.2f} USDT\n"

    return message

# Main function to send portfolio summary
async def send_portfolio_summary():
    pairs = os.getenv('PAIRS').replace('"', '').split(',')
    profit_loss_summary = {}
    traded_tokens = set(pair.split('/')[0] for pair in pairs)  # Extract base tokens (e.g., BTC from BTC/USDT)

    for pair in pairs:
        print(f"Fetching data for {pair}...")
        trades = fetch_last_transactions(pair)
        profit_loss = calculate_profit_loss(trades)
        profit_loss_summary[pair] = profit_loss

    # Fetch the current portfolio balance for the traded tokens
    portfolio = fetch_portfolio_summary(traded_tokens)

    # Format the message
    summary_message = format_summary_message(portfolio, profit_loss_summary)

    # Send the summary via Telegram
    await send_telegram_message(tg_bot, summary_message, tg_channel)

# Function to send a message to Telegram
async def send_telegram_message(bot, message, chat_id):
    try:
        await bot.send_message(chat_id=chat_id, text=message)
        print(f"Telegram message sent: {message}")
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")

# Run the portfolio summary function
if __name__ == "__main__":
    asyncio.run(send_portfolio_summary())
