import numpy as np
import pandas as pd
from data import DataLoaderYfinance as yf_data

class Basket:
    def __init__(self, name, weights, tickers, benchmark_ticker):
        if len(tickers) != len(weights):
            raise ValueError("Tickers and weights must be the same length!")
        if not np.isclose(sum(weights), 1):
            raise ValueError("Weights must sum to 1!")
        self.name = name
        self.tickers = tickers
        self.weights = np.array(weights)
        self.benchmark_ticker = benchmark_ticker
    
    def basket_info(self):
        return {
            'Name': self.name,
            'Tickers': self.tickers,
            'Weights': self.weights,
            'Benchmark': self.benchmark_ticker
        }

class PortfolioManager:
    def __init__(self, baskets, start_date, end_date):
        self.baskets = baskets
        self.start = start_date
        self.end = end_date
        self.data_daily = None
        self.data_monthly = None
        self._fetch_all_data()

    def _fetch_all_data(self):
        # Collect all unique tickers and benchmarks
        all_tickers = list(set([t for b in self.baskets for t in b.tickers]))
        all_benchmarks = list(set([b.benchmark_ticker for b in self.baskets] + ['SPY']))
        full_list = all_tickers + all_benchmarks

        # Download data monthly and daily
        self.data_daily, self.data_monthly = yf_data.get_daily_monthly_yf(full_list, self.start, self.end)

    def calculate_basket_metrics(self):
        results = []
        for b in self.baskets:
            # Daily stats
            daily_rets = self.data_daily[b.tickers].pct_change().dropna()
            basket_daily_rets = daily_rets @ b.weights
            
            # Volatility
            cov = daily_rets.cov()
            vol_daily = np.sqrt(b.weights.T @ cov @ b.weights)
            
            # Beta against its specific benchmark
            bench_rets = self.data_daily[b.benchmark_ticker].pct_change().dropna()
            combined = pd.concat([basket_daily_rets, bench_rets], axis=1).dropna()
            beta = np.cov(combined.iloc[:,0], combined.iloc[:,1])[0,1] / np.var(combined.iloc[:,1])

            # Monthly stats
            m_rets = self.data_monthly[b.tickers].pct_change().dropna()
            basket_monthly_rets = m_rets @ b.weights

            results.append({
                'Name': b.name,
                'Daily Rets': basket_daily_rets,
                'Monthly Rets': basket_monthly_rets,
                'Ann Return': basket_daily_rets.mean() * 252,
                'Ann Vol': vol_daily * np.sqrt(252),
                'Beta': beta,
                'Benchmark': b.benchmark_ticker
            })
        return results
