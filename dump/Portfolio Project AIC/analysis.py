import pandas as pd
import numpy as np
import statsmodels.api as sm
from optimiser import Optimiser

class Analysis:
    def __init__(self, manager, risk_aversion=0.5, conf=0.05,max_beta=1.0):
        self.manager = manager
        self.risk_aversion = risk_aversion
        self.conf = conf
        self.max_beta = max_beta
    @staticmethod
    def colour_grading(column):
        if column.name == "Sharpe Ratio":
            return ['background-color: green' if v > 1 else 'background-color: orange' if v > 0.5 else 'background-color: red' for v in column]
        elif column.name == "Sortino Ratio":
            return ['background-color: green' if v > 2 else 'background-color: orange' if v > 1 else 'background-color: red' for v in column]
        elif column.name == "R-Squared (Composite)":
            return ['background-color: green' if v > 0.8 else 'background-color: orange' if v > 0.5 else 'background-color: red' for v in column]
        elif column.name == "Calmar Ratio":
            return ['background-color: green' if v > 1 else 'background-color: orange' if v > 0.5 else 'background-color: red' for v in column]
        elif column.name == "VaR (Monthly)":
            return ['background-color: red' if v < -0.05 else 'background-color: orange' if v < 0 else 'background-color: green' for v in column]
        elif column.name == "CVaR (Monthly)":
            return ['background-color: red' if v < -0.07 else 'background-color: orange' if v < 0 else 'background-color: green' for v in column]
        elif column.name == "Max Drawdown Duration (days)":   
            return ['background-color: red' if v > 126 else 'background-color: orange' if v > 63 else 'background-color: green' for v in column]
        elif column.name == "Max Drawdown Duration (months)":   
            return ['background-color: red' if v > 6 else 'background-color: orange' if v > 3 else 'background-color: green' for v in column]
        elif column.name == "Skewness":
            return ['background-color: green' if v > 0 else 'background-color: orange' if v > -0.5 else 'background-color: red' for v in column]
        elif column.name == "Kurtosis":
            return ['background-color: green' if v < 3 else 'background-color: orange' if v < 5 else 'background-color: red' for v in column]
        elif column.name == "Max Drawdown":
            return ['background-color: red' if v < -0.2 else 'background-color: orange' if v < -0.1 else 'background-color: green' for v in column]
        elif column.name == "Upside Capture %":
            return ['background-color: green' if v > 100 else 'background-color: orange' if v > 50 else 'background-color: red' for v in column]
        elif column.name == "Downside Capture %":
            return ['background-color: red' if v > 100 else 'background-color: orange' if v > 50 else 'background-color: green' for v in column]
        else:
            return [''] * len(column)
    
    def generate_report(self):
        stats = self.manager.calculate_basket_metrics()
        weights = Optimiser.get_weights(stats, self.risk_aversion)
        
        # 1. Composite Performance
        port_monthly_rets = pd.concat([b['Monthly Rets'] for b in stats], axis=1).dropna() @ weights
        port_daily_rets = pd.concat([b['Daily Rets'] for b in stats], axis=1).dropna() @ weights
        

        bench_monthly_df = pd.concat([self.manager.data_monthly[b['Benchmark']].pct_change() for b in stats], axis=1).dropna()
        comp_bench_rets = bench_monthly_df @ weights

        ann_ret = port_daily_rets.mean() * 252
        ann_vol = port_daily_rets.std() * np.sqrt(252)
        
        # 3. Ratios
        # Sharpe
        sharpe = ann_ret / ann_vol
        
        # Sortino
        downside_std = port_monthly_rets[port_monthly_rets < 0].std() * np.sqrt(12)
        sortino = (port_monthly_rets.mean() * 12) / downside_std

        # R-Squared
        X = sm.add_constant(comp_bench_rets.values)
        r2 = sm.OLS(port_monthly_rets.values, X).fit().rsquared

        # 4. Drawdown
        wealth = (1 + port_daily_rets).cumprod()
        peaks = wealth.expanding().max()
        dd = (wealth - peaks) / peaks
        mdd = dd.min()
        mdd_dur = (dd < 0).groupby((dd == 0).cumsum()).sum().max()

        # 5. Risk
        var = np.percentile(port_monthly_rets, self.conf * 100)
        cvar = port_monthly_rets[port_monthly_rets <= var].mean()

        # 6. Captures
        up = comp_bench_rets > 0
        dn = comp_bench_rets < 0
        up_cap = (port_monthly_rets[up].mean() / comp_bench_rets[up].mean()) * 100
        dn_cap = (port_monthly_rets[dn].mean() / comp_bench_rets[dn].mean()) * 100

        metrics = {
            'Annualized Return': ann_ret,
            'Annualized Volatility': ann_vol,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'R-Squared (Composite)': r2,
            'Max Drawdown': mdd,
            'Max Drawdown Duration (days)': mdd_dur,
            'Max Drawdown Duration (months)': mdd_dur / 21,  # Approximate trading days in a month
            'Calmar Ratio': ann_ret / abs(mdd),
            'Upside Capture %': up_cap,
            'Downside Capture %': dn_cap,
            'VaR (Monthly)': var,
            'CVaR (Monthly)': cvar,
            'Skewness': port_monthly_rets.skew(),
            'Kurtosis': port_monthly_rets.kurtosis()
        }

        report = pd.Series(metrics, name="Portfolio Metrics")
        weights_series = pd.Series(weights, index=[b['Name'] for b in stats], name="Optimal Weights")
        
        return report, weights_series, port_monthly_rets, port_daily_rets
    
