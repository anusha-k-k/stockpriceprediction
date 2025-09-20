#minor project
#stock price prediction
#project by Anusha K K


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def generate_future_stock_prices_wide(
    tickers, start_prices=None, n_days=30, drift=0.0005, volatility=0.02, seed=None
):
    if seed is not None:
        np.random.seed(seed)

    if start_prices is None:
        start_prices = {ticker: 100 for ticker in tickers}

    start_datetime = datetime.now()
    future_dates = [start_datetime + timedelta(days=i) for i in range(1, n_days + 1)]

    price_data = {"Date": future_dates}

    for ticker in tickers:
        daily_returns = np.random.normal(loc=drift, scale=volatility, size=n_days)
        prices = start_prices.get(ticker, 100) * np.exp(np.cumsum(daily_returns))
        price_data[ticker] = prices

    return pd.DataFrame(price_data)

if __name__ == "__main__":
    # Define stock tickers and starting prices
    tickers = ['AAPL', 'MSFT', 'META', 'NFLX', 'JPM', 'TSLA']
    start_prices = {
        'AAPL': 190,
        'MSFT': 350,
        'META': 320,
        'NFLX': 430,
        'JPM': 160,
        'TSLA': 700,
    }

    n_days = 30

    # Generate predictions
    predictions_df = generate_future_stock_prices_wide(
        tickers=tickers,
        start_prices=start_prices,
        n_days=n_days,
        seed=42
    )

    # Optional: round numbers for cleaner output
    predictions_df = predictions_df.round(2)

    # ✅ Print a "CSV" to terminal with visible column gaps (fixed-width style)
    print(predictions_df.to_string(index=False, justify='left'))

    # 📈 Plotting
    predictions_df.set_index("Date", inplace=True)

    plt.figure(figsize=(14, 7))
    for ticker in tickers:
        plt.plot(predictions_df.index, predictions_df[ticker], label=ticker)

    plt.title("Stock Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
