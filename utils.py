import pandas as pd
import numpy as np
import networkx as nx
import yfinance as yf


SP500_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "JPM", "JNJ", "V", "UNH", "XOM", "PG", "MA", "HD", "CVX", "MRK",
    "ABBV", "PEP", "KO", "AVGO", "COST", "MCD", "WMT", "BAC", "CRM",
    "ACN", "LLY", "TMO", "CSCO", "ABT", "NKE", "NEE", "DHR", "TXN",
    "ORCL", "PM", "MS", "RTX", "AMGN", "HON", "UPS", "QCOM", "IBM",
    "GS", "CAT", "SBUX", "BA", "GE"
]

CRYPTO_TICKERS = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD"]

ALL_TICKERS = SP500_TICKERS + CRYPTO_TICKERS


def load_prices(start="2019-01-01", end="2024-12-31"):
    raw = yf.download(ALL_TICKERS, start=start, end=end, auto_adjust=True)
    prices = raw.xs("Close", axis=1, level="Price")
    prices = prices.dropna(how="all")
    prices = prices.dropna(axis=1, thresh=int(0.5 * len(prices)))
    return prices


def compute_returns(prices):
    return np.log(prices / prices.shift(1)).dropna(how="all")


def build_network(returns, threshold=0.5):
    corr_matrix = returns.corr()
    G = nx.Graph()
    G.add_nodes_from(corr_matrix.columns)
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            asset_i = corr_matrix.columns[i]
            asset_j = corr_matrix.columns[j]
            correlation = corr_matrix.loc[asset_i, asset_j]
            if correlation > threshold:
                G.add_edge(asset_i, asset_j, weight=correlation)
    return G, corr_matrix