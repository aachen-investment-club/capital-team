'''
This is my testing document, where I will be testing for statistical significance of stationarity and volatility clustering in the data.
stationarity means that the statistical properties of the data do not change over time
each security has a 'true' mean and the data fluctuates around this

1. Test if returns are (weakly) stationary
2. Test for ARCH effects (volatility clustering)

At the end of the project, I can test the residuals of the model for ARCH effects, to see if the model has captured all the volatility clustering in the data.

I have tested according to the following: 


Augmented Dicky Fuller test (ADF test):

Test for a unit root. This means that the data suffers from a shock that does not revert.
This implies non-stationarity, as there is not 'true' underlying trend. 

H0: the data is non-stationary
H1: the data is stationary


KPSS test:

Test for stationarity around a deterministic trend, using unit root. This means that the data has a 'true' underlying trend, but it is not affected by shocks.

H0: the data is stationary
H1: the data is non-stationary


Phillips-Perron test (PP test):

Test for a unit root, but it is more robust to heteroskedasticity and autocorrelation than the ADF test.

H0: the data is non-stationary
H1: the data is stationary

Ljung-Box test on squared returns:

Test for autocorrelation in the squared returns. This does not explicitly show ARCH effects, but it is a good indicator of volatility clustering.

H0: no autocorrelation in squared returns (white noise)
H1: autocorrelation in squared returns

Engle's ARCH-LM test:

Test for ARCH effects by regressing the squared returns on their own lags. This is a more direct test for volatility clustering.

H0: no ARCH effects(homoskedasticity=constant variance)
H1: ARCH effects present(heteroskedasticity=changing variance)

-------------------------------------
ISSUES TO LOOK OUT FOR:

Structural breaks: sudden market changes that affect stationarity
Sample-size sensitivity: small samples may not capture true properties, especially for ADF and PP

--------------------------------------

The data inputted will come from the loader.py file, which will be a series of daily closing prices for a single security

'''

import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from arch.unitroot import PhillipsPerron
import matplotlib.pyplot as plt


class Testing:
    def __init__(self):
        pass

    @staticmethod
    def _clean(series):
        """Drop NaN/inf so the statsmodels tests don't crash."""
        s = series.replace([np.inf, -np.inf], np.nan).dropna()
        return s

    @staticmethod
    def test_adf(returns):
        s = Testing._clean(returns)
        result = adfuller(s)
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
        verdict = "stationary" if result[1] < 0.05 else "non-stationary"
        print(f"ADF Verdict: {verdict}")

    @staticmethod
    def test_kpss(returns):
        s = Testing._clean(returns)
        result = kpss(s, regression='c')
        print('KPSS Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[3].items():
            print('\t%s: %.3f' % (key, value))
        verdict = "stationary" if result[1] > 0.05 else "non-stationary"
        print(f"KPSS Verdict: {verdict}")

    @staticmethod
    def test_pp(returns):
        s = Testing._clean(returns)
        result = PhillipsPerron(s)
        print('PP Statistic: %f' % result.stat)
        print('p-value: %f' % result.pvalue)
        print('Critical Values:')
        for key, value in result.critical_values.items():
            print('\t%s: %.3f' % (key, value))
        verdict = "non-stationary" if result.pvalue > 0.05 else "stationary"
        print(f"PP Verdict: {verdict}")

    @staticmethod
    def plot_data(returns):
        """Plot a returns series with rolling 60-day mean and std-dev overlays."""
        r = Testing._clean(returns)

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

        r.plot(ax=axes[0], title="Log Returns (%)")
        axes[0].axhline(0, color="black", linewidth=0.5)

        r.rolling(60).mean().plot(ax=axes[1], title="Rolling 60-day Mean")
        axes[1].axhline(r.mean(), color="red", linestyle="--", linewidth=0.8)

        r.rolling(60).std().plot(ax=axes[2], title="Rolling 60-day Std Dev")
        axes[2].axhline(r.std(), color="red", linestyle="--", linewidth=0.8)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def test_arch(prices, lags=[5, 10, 22]):
        """Takes PRICES and computes returns internally."""
        log_returns = (np.log(prices / prices.shift(1)) * 100).replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        # Ljung-Box on squared returns
        print("=== Ljung-Box on Squared Returns ===")
        lb = acorr_ljungbox(log_returns ** 2, lags=lags, return_df=True)
        print(lb)
        print()

        # Engle's ARCH-LM at the maximum lag
        print(f"=== Engle's ARCH-LM (lags = {max(lags)}) ===")
        stat, p_value, _, _ = het_arch(log_returns, nlags=max(lags))
        print(f"LM Statistic: {stat:.4f}")
        print(f"p-value: {p_value:.6f}")

        if p_value < 0.05:
            print("\nVerdict: ARCH effects detected — GARCH is justified")
        else:
            print("\nVerdict: No significant ARCH effects — variance may be constant")

    @staticmethod
    def test_stationarity(returns):
        """Run ADF, KPSS, PP and plot the returns series."""
        print("Running ADF Test:")
        Testing.test_adf(returns)
        print("\nRunning KPSS Test:")
        Testing.test_kpss(returns)
        print("\nRunning PP Test:")
        Testing.test_pp(returns)
        print("\nPlotting Data:")
        Testing.plot_data(returns)