'''this is the data loader
I will be taking data from the LSEG refinitiv workspace
login details are called from an external .json file (lseg-data.config.json)
'''
import pandas as pd
import lseg.data as ld
import yfinance as yf

class DataLoader:
    def __init__(self, config_path="./lseg-data.config.json", open_session=True):
        self.config_path = config_path
        if open_session:
            ld.open_session()

    def clean_up(self, data):
        data = data.copy()                                  # avoid mutating the original
        data.index = pd.to_datetime(data.index)             # ensure index is datetime
        data = data.sort_index()                            # chronological order
        data = data[~data.index.duplicated(keep="last")]    # de-duplicate
        data = data.apply(pd.to_numeric, errors="coerce")   # non-numerics -> NaN
        data = data.dropna(how="all")                       # drop fully-NaN rows
        return data

    def _to_series(self, data):
        """Convert single-column DataFrame from get_history into a 1-D Series."""
        clean = self.clean_up(data)
        if isinstance(clean, pd.DataFrame):
            clean = clean.iloc[:, 0]
        return clean

    def load_daily_close(self, ticker, start, end):
        data = ld.get_history(
            universe=[ticker],
            fields=["TR.PriceClose"],
            parameters={"Adjusted": 1},
            start=start,
            end=end,
            interval="1D",
        )
        return self._to_series(data)

    def load_intraday(self, ticker, start, end):
        data = ld.get_history(
            universe=[ticker],
            fields=["TR.PriceClose"],
            parameters={"Adjusted": 1},
            start=start,
            end=end,
            interval="5min",
        )
        return self._to_series(data)
    
    def load_vix(self, start, end):
        data = yf.download("^VIX", start=start, end=end)["Close"]
        return self._to_series(data)