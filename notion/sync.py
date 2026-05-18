from dotenv import load_dotenv
load_dotenv() 

import yfinance as yf
import requests
from datetime import datetime
import os
import time

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
TICKERS_DB_ID = os.getenv("TICKERS_DB_ID")
PRICES_DB_ID = os.getenv("PRICES_DB_ID")

HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28",   # stable version, no breaking changes
}

def get_active_tickers() -> list[str]:
    url = f"https://api.notion.com/v1/databases/{TICKERS_DB_ID}/query"
    payload = {"filter": {"property": "Active", "checkbox": {"equals": True}}}
    response = requests.post(url, headers=HEADERS, json=payload)
    response.raise_for_status()
    tickers = []
    for page in response.json()["results"]:
        title = page["properties"]["Name"]["title"]
        if title:
            tickers.append(title[0]["plain_text"].upper())
    return tickers

def fetch_stock_data(ticker: str) -> dict:
    stock = yf.Ticker(ticker)
    info = stock.info
    current_year = datetime.today().year
    hist = stock.history(start=f"{current_year}-01-01")
    if len(hist) >= 2:
        ytd_return = round((hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100, 2)
    else:
        ytd_return = None
    return {
        "ticker": ticker,
        "market_cap": info.get("marketCap"),
        "revenue_ttm": info.get("totalRevenue"),
        "ytd_return": ytd_return,
    }

def push_to_notion(data: dict):
    url = "https://api.notion.com/v1/pages"
    payload = {
        "parent": {"database_id": PRICES_DB_ID},
        "properties": {
            "Ticker":       {"title":  [{"text": {"content": data["ticker"]}}]},
            "Market Cap":   {"number": data["market_cap"]},
            "Revenue TTM":  {"number": data["revenue_ttm"]},
            "Return YTD":   {"number": data["ytd_return"]},
            "Date":         {"date":   {"start": datetime.today().strftime("%Y-%m-%d")}},
        }
    }
    response = requests.post(url, headers=HEADERS, json=payload)
    response.raise_for_status()
    print(f"✅ {data['ticker']} pushed to Notion")

tickers = get_active_tickers()
print(f"Found {len(tickers)} active tickers: {tickers}")

for ticker in tickers:
    try:
        data = fetch_stock_data(ticker)
        push_to_notion(data)
    except Exception as e:
        print(f"❌ Failed for {ticker}: {e}")
    time.sleep(0.5)