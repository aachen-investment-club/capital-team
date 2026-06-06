import pandas as pd
import yfinance as yf
import lseg.data as ld
import logging


class DataLoaderYfinance:
    @staticmethod
    def fetch_prices_yf(tickers, start, end, interval='1d'):
        data = yf.download(tickers, start=start, end=end, interval=interval, progress=False)
        if 'Adj Close' in data.columns:
            prices = data['Adj Close']
        else:
            prices = data['Close']
        return prices.ffill().dropna()

    @staticmethod
    def get_daily_monthly_yf(tickers, start, end):
        daily = DataLoaderYfinance.fetch_prices_yf(tickers, start, end, interval='1d')
        monthly = DataLoaderYfinance.fetch_prices_yf(tickers, start, end, interval='1mo')
        return daily, monthly


class DataLoaderLSEG:
    def __init__(self, config_path="./lseg-data.config.json"):
        self.config_path = config_path
        self.session = None

    def open_session_lseg(self):
        try:
            self.session = ld.open_session()
            logging.info("LSEG Session opened successfully.")
        except Exception as e:
            logging.error(f"Failed to open session: {e}")
            raise

    def get_historical_data_lseg(self, universe, fields, start, end, interval="daily"):
        try:
            df = ld.get_history(
                universe=universe,
                fields=fields,
                start=start,
                end=end,
                interval=interval
            )
            if df is None or df.empty:
                logging.warning(f"No data returned for {universe}")
                return pd.DataFrame()
            return df
        except Exception as e:
            logging.error(f"Error fetching data for {universe}: {e}")
            return pd.DataFrame()

    def _fetch_fundamentals(self, universe):
        try:
            df = ld.get_data(
                universe=universe,
                fields=[
                    "TR.GICSSector",
                    "TR.PriceToBVPerShare",
                    "TR.Volume",
                    "TR.SharesOutstanding",
                    "TR.CompanyMarketCap",
                ]
            )
            if df is not None and not df.empty and 'Instrument' in df.columns:
                df.set_index('Instrument', inplace=True)
                df.index = df.index.str.replace(r'\.[A-Z]+$', '', regex=True)
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            logging.error(f"Fundamentals fetch failed for {universe}: {e}")
            return pd.DataFrame()

    def get_barra_fundamentals_lseg(self, universe, nasdaq, nyse):
        from datetime import datetime, timedelta

        today         = datetime.today().strftime('%Y-%m-%d')
        two_years_ago = (datetime.today() - timedelta(days=730)).strftime('%Y-%m-%d')
        one_year_ago  = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
        one_month_ago = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')

        try:
            df_nasdaq       = self._fetch_fundamentals(nasdaq)
            df_nyse         = self._fetch_fundamentals(nyse)
            df_fundamentals = pd.concat([df_nasdaq, df_nyse])

            df_returns   = ld.get_data(
                universe=universe,
                fields=[f"TR.TotalReturn(SDate={two_years_ago}, EDate={today}, Frq=D)"]
            )
            df_price_now = ld.get_data(universe=universe, fields=[f"TR.PriceClose(SDate={today})"])
            df_price_1m  = ld.get_data(universe=universe, fields=[f"TR.PriceClose(SDate={one_month_ago})"])
            df_price_1y  = ld.get_data(universe=universe, fields=[f"TR.PriceClose(SDate={one_year_ago})"])

            for df in [df_returns, df_price_now, df_price_1m, df_price_1y]:
                if df is None or df.empty:
                    logging.error("One or more LSEG price/return fetches returned empty.")
                    return pd.DataFrame()
                if 'Instrument' in df.columns:
                    df.set_index('Instrument', inplace=True)
                df.index = df.index.str.replace(r'\.[A-Z]+$', '', regex=True)

            price_col_now = [c for c in df_price_now.columns if 'Price' in c or 'Close' in c]
            price_col_1m  = [c for c in df_price_1m.columns  if 'Price' in c or 'Close' in c]
            price_col_1y  = [c for c in df_price_1y.columns  if 'Price' in c or 'Close' in c]

            if price_col_now: df_price_now.rename(columns={price_col_now[0]: 'TR.PriceClose(SDate=0)'},   inplace=True)
            if price_col_1m:  df_price_1m.rename( columns={price_col_1m[0]:  'TR.PriceClose(SDate=-1M)'}, inplace=True)
            if price_col_1y:  df_price_1y.rename( columns={price_col_1y[0]:  'TR.PriceClose(SDate=-1Y)'}, inplace=True)

            df = df_fundamentals.join([df_returns, df_price_now, df_price_1m, df_price_1y], how='left')

            col_rename = {}
            for col in df.columns:
                cl = col.lower()
                if 'gics' in cl or 'sector' in cl:
                    col_rename[col] = 'TR.GICSSector'
                elif 'book' in cl or 'pricetobv' in cl:
                    col_rename[col] = 'TR.PriceToBVPerShare'
                elif 'volume' in cl:
                    col_rename[col] = 'TR.Volume'
                elif 'outstanding' in cl or 'shares out' in cl:
                    col_rename[col] = 'TR.SharesOutstanding'
                elif 'market cap' in cl or 'marketcap' in cl:
                    col_rename[col] = 'TR.CompanyMarketCap'
                elif 'total return' in cl or 'totalreturn' in cl:
                    col_rename[col] = 'TR.TotalReturn(SDate=-2Y, EDate=0, Frq=D)'

            df.rename(columns=col_rename, inplace=True)

            if df.index.duplicated().any():
                df = df[~df.index.duplicated(keep='last')]

            return df

        except Exception as e:
            logging.error(f"Failed to fetch fundamentals: {e}")
            return pd.DataFrame()

    def close_session_lseg(self):
        if self.session:
            ld.close_session()


class DataProcessor:
    def __init__(self):
        pass

    def process_lseg(self, raw_data):
        raw_data.to_csv("lseg_raw_data.csv", index=True)
        print("LSEG raw data saved to lseg_raw_data.csv")
        return raw_data