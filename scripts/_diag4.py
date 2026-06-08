import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from lib.data import get_security_master, get_daily_weightings_history

sm = get_security_master()
print("Security master asset_types:")
print(sm[["ticker","asset_type"]].to_string())

wt = get_daily_weightings_history()
if not wt.empty:
    latest = wt[wt["date"] == wt["date"].max()]
    latest = latest[~latest["symbol"].str.startswith("CASH_")]
    print("\nLatest portfolio symbols:", latest["symbol"].tolist())
