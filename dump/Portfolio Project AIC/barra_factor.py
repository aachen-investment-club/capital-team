'''R = X*f + e
R: returns of the assets
X: factor loadings (exposures)
f: factor returns
e: idiosyncratic returns (residuals)

we can determine the factors based off of style or industry:

for style factors, we chose: value, growth, momentum, volatility, and liquidity
value: book-to-market ratio
growth: earnings growth rate
momentum: return in past 12 months, excluding last month
volatility: standard deviation of returns
liquidity: share turnover

for industry factors, we can use the 11 GICS sectors:
Consumer Discretionary
Consumer Staples
Energy
Financials
Health Care
Industrials
Information Technology
Materials
Real Estate
Telecommunication Services
Utilities



'''
import pandas as pd
import numpy as np
import statsmodels.api as sm
import datetime


class BarraFactor:
    def __init__(self, fundamental_df):
        self.raw_data = fundamental_df.copy()
        self.exposure_matrix = None
        self.factor_results = None
        self.factor_returns = None
        self.omega_annual = None

    def calculate_factor_returns(self):
        cols_to_fix = [
            'TR.PriceToBVPerShare', 'TR.PriceClose(SDate=0)',
            'TR.PriceClose(SDate=-1M)', 'TR.PriceClose(SDate=-1Y)',
            'TR.Volume', 'TR.SharesOutstanding', 'TR.CompanyMarketCap',
            'TR.TotalReturn(SDate=-2Y, EDate=0, Frq=D)'
        ]
        for col in cols_to_fix:
            if col in self.raw_data.columns:
                self.raw_data[col] = pd.to_numeric(self.raw_data[col], errors='coerce')

        industry_dummies = pd.get_dummies(
            self.raw_data['TR.GICSSector'], prefix='Industry'
        ).astype(float)

        value_raw      = 1 / (self.raw_data['TR.PriceToBVPerShare'] + 1e-8)
        growth_raw     = self.raw_data['TR.PriceClose(SDate=0)'] / self.raw_data['TR.PriceClose(SDate=-1Y)'] - 1
        momentum_raw   = self.raw_data['TR.PriceClose(SDate=-1M)'] / self.raw_data['TR.PriceClose(SDate=-1Y)'] - 1
        volatility_raw = (
            self.raw_data['TR.PriceClose(SDate=0)'] - self.raw_data['TR.PriceClose(SDate=-1Y)']
        ).abs() / (self.raw_data['TR.PriceClose(SDate=-1Y)'] + 1e-8)
        liquidity_raw  = self.raw_data['TR.Volume'] / self.raw_data['TR.SharesOutstanding']

        def winsorize(series, lower_quantile=0.01, upper_quantile=0.99):
            return series.clip(lower=series.quantile(lower_quantile),
                               upper=series.quantile(upper_quantile))

        def get_z_score(series):
            return (series - series.mean()) / (series.std() + 1e-8)

        X_style = pd.DataFrame({
            'Value':      get_z_score(winsorize(value_raw)),
            'Growth':     get_z_score(winsorize(growth_raw)),
            'Momentum':   get_z_score(winsorize(momentum_raw)),
            'Volatility': get_z_score(winsorize(volatility_raw)),
            'Liquidity':  get_z_score(winsorize(liquidity_raw)),
        }, index=self.raw_data.index).astype(float)

        X_combined = pd.concat([industry_dummies, X_style], axis=1)
        self.exposure_matrix = X_combined

        y           = pd.to_numeric(self.raw_data['TR.TotalReturn(SDate=-2Y, EDate=0, Frq=D)'], errors='coerce')
        market_caps = pd.to_numeric(self.raw_data['TR.CompanyMarketCap'], errors='coerce')
        w           = np.sqrt(market_caps)

        full_data = pd.concat([y, self.exposure_matrix, w.rename('Weights')], axis=1).dropna()

        # Force everything to float64 before passing to statsmodels
        y_clean = full_data['TR.TotalReturn(SDate=-2Y, EDate=0, Frq=D)'].astype(float)
        X_clean = full_data[self.exposure_matrix.columns].astype(float)
        w_clean = full_data['Weights'].astype(float)

        model = sm.WLS(y_clean, X_clean, weights=w_clean)
        self.factor_results = model.fit()
        self.factor_returns = self.factor_results.params
        return self.factor_returns

    def match_to_baskets(self, baskets):
        attribution_report = {}
        for basket in baskets:
            w_series      = pd.Series(basket.weights, index=basket.tickers)
            valid_tickers = self.exposure_matrix.index.intersection(basket.tickers)
            if len(valid_tickers) == 0:
                continue
            basket_exposures      = self.exposure_matrix.loc[valid_tickers]
            weights_aligned       = w_series.loc[valid_tickers]
            weights_aligned       = weights_aligned / weights_aligned.sum()
            basket_total_exposure = basket_exposures.T.dot(weights_aligned)
            contributions         = basket_total_exposure * self.factor_returns
            attribution_report[basket.name] = {
                'Total_Return':     contributions.sum(),
                'Factor_Breakdown': contributions,
                'Total_Exposures':  basket_total_exposure
            }
        return attribution_report

    def factor_return_history(self):
        if self.factor_results is None:
            raise RuntimeError("Call calculate_factor_returns() first.")
        history = pd.DataFrame(
            [self.factor_returns],
            index=[pd.Timestamp(datetime.date.today())]
        )
        return history

    def calculate_ewma_risk_matrix(self, factor_return_history, halflife=60):
        # With a single cross-sectional snapshot we cannot compute a true EWMA.
        # Use the OLS parameter covariance matrix as a proxy for factor risk.
        if factor_return_history.shape[0] < 2:
            cov = pd.DataFrame(
                self.factor_results.cov_params(),
                index=self.factor_returns.index,
                columns=self.factor_returns.index
            ).astype(float)
            self.omega_annual = cov * 252
        else:
            ewma_cov_df        = factor_return_history.ewm(halflife=halflife).cov()
            latest_date        = factor_return_history.index[-1]
            latest_omega_daily = ewma_cov_df.xs(latest_date)
            self.omega_annual  = latest_omega_daily * 252
        return self.omega_annual

    def _compute_daily_factors(self, daily_df):
        raise NotImplementedError(
            "Pass time-series fundamental data (one row per instrument per date) "
            "to use this method. Currently get_barra_fundamentals_lseg returns a snapshot."
        )


    def risk_results(self, baskets, benchmark_exposure=None):
        risk_results = {}
        report = self.match_to_baskets(baskets)
        for basket_name, data in report.items():
            h = data['Total_Exposures']
            if benchmark_exposure is not None:
                h = h - benchmark_exposure
            systematic_var        = h.dot(self.omega_annual).dot(h)
            systematic_vol        = np.sqrt(systematic_var)
            marginal_contribution = self.omega_annual.dot(h)
            risk_contribution     = h * marginal_contribution / systematic_vol
            risk_results[basket_name] = {
                'Systematic_Vol':    systematic_vol,
                'Risk_Contribution': risk_contribution
            }
        return risk_results