import yfinance as yf
import pandas as pd
import numpy as np

ASSET_UNIVERSE = {
    "XLK": "Tech",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLY": "Consumer Disc.",
    "XLP": "Consumer Staples",
    "SPY": "S&P 500",
    "EFA": "Intl Developed",
    "EEM": "Emerging Markets",
    "TLT": "Long-Term Treasury",
    "IEF": "Mid-Term Treasury",
    "LQD": "Corp Bonds",
    "HYG": "High Yield",
    "VNQ": "Real Estate (REIT)",
    "GLD": "Gold",
    "USO": "Oil",
}

TICKERS = list(ASSET_UNIVERSE.keys())


def download_prices(tickers=None, start="2015-01-01", end="2024-12-31"):
    if tickers is None:
        tickers = TICKERS
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"]
    threshold = 0.05 * len(prices)
    prices = prices.dropna(axis=1, thresh=len(prices) - threshold)
    prices = prices.dropna()
    return prices


def compute_log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()


def compute_annual_stats(log_returns):
    mean_returns = log_returns.mean() * 252
    volatilities = log_returns.std() * np.sqrt(252)
    return mean_returns, volatilities


def compute_correlation_matrix(log_returns):
    return log_returns.corr()


def save_data(prices, log_returns, path="data/"):
    prices.to_csv(f"{path}prices.csv")
    log_returns.to_csv(f"{path}log_returns.csv")
    print(f"Saved to {path}")


def load_data(path="data/"):
    prices = pd.read_csv(f"{path}prices.csv", index_col=0, parse_dates=True)
    log_returns = pd.read_csv(f"{path}log_returns.csv", index_col=0, parse_dates=True)
    return prices, log_returns