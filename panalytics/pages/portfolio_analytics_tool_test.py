import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Portfolio Analytics",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
[data-testid="stMetricValue"] {
    font-size: 1.5rem;
}
</style>
""", unsafe_allow_html=True)


# ===== DATA LOADING =====

trades = pd.read_csv("data/portfolio.csv", parse_dates=["date"])
close_prices = pd.read_csv("data/timeseries.csv", index_col=0, parse_dates=True)

st.write("### Trades")
st.write(trades)
#st.write("### Close Prices")
#st.write(close_prices)

# conversion from wide to long format
prices_long = close_prices.stack().reset_index()
prices_long.columns = ["date", "ticker", "close"]

# st.write("### Close prices in long format")
# st.write(prices_long)


# ===== TRADES =====

st.write("### Trades with purchase prices based on close price at trade date")
trades = trades.merge(
    prices_long,
    left_on=["date", "ticker"],
    right_on=["date", "ticker"],
)
trades["purchase_price"] = trades["units"] * trades["close"]
st.write(trades)

total_purchase_price = trades["purchase_price"].sum()


# ===== PORTFOLIO VALUE =====

# drop unnecessary columns (especially purchase_price)
portfolio = trades.drop(columns=["date", "purchase_price"])
# aggregate by ticker and units -> get total units per ticker
portfolio = portfolio.groupby(["ticker"])[["units"]].sum().reset_index()
# look up current price for each ticker and calculate current value of positions
portfolio["current_price"] = portfolio["ticker"].map(close_prices.iloc[-1])
portfolio["current_value"] = portfolio["units"] * portfolio["current_price"]

current_portfolio_value = portfolio["current_value"].sum()

portfolio_return = current_portfolio_value / total_purchase_price - 1


# =====  =====

trade_shares = (
    trades
    .pivot_table(index="date", columns="ticker",
                 values="units", aggfunc="sum")
    .reindex(close_prices.index) # alle Handelstage
    .fillna(0)
    .cumsum() # ← laufende Stückzahl
)

st.write(trade_shares)

trade_cashflows = (
    trades
    .assign(cashflow=lambda x: x["units"] * x["close"])
    .groupby("date")["cashflow"]
    .sum()
    .reindex(close_prices.index)
    .fillna(0)
)

portfolio_value = (close_prices * trade_shares).sum(axis=1)
st.write(portfolio_value)

def twr_returns(pv, cf):
    rets = []
    for i in range(1, len(pv)):
        # Wert heute OHNE heutigen Kapitalzufluss
        val_ex_cf = pv.iloc[i] - cf.iloc[i]
        r = val_ex_cf / pv.iloc[i-1] - 1
        rets.append(r)
    return pd.Series(rets, index=pv.index[1:])

portfolio_returns = twr_returns(portfolio_value, trade_cashflows)
portfolio_returns.fillna(0)
st.write(portfolio_returns)

rf = 0.035 # risikofreier Zins

vol = portfolio_returns.std() * np.sqrt(252)
sharpe = (portfolio_returns.mean()*252 - rf) / vol
cum = (1 + portfolio_returns).cumprod()
mdd = ((cum - cum.cummax()) / cum.cummax()).min()
twr_total = cum.iloc[-1] - 1
cagr = (cum.iloc[-1] ** (252/len(cum))) - 1

# Downside-Volatilität für Sortino
downside = portfolio_returns[portfolio_returns < 0]
sortino = (portfolio_returns.mean()*252 - rf) / (downside.std()*np.sqrt(252))


# ===== DESIGN =====

st.title("Portfolio Analytics Tool")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Purchase Price", f"€{total_purchase_price:,.2f}")
with col2:
    st.metric("Current Portfolio Value", f"€{current_portfolio_value:,.2f}")
with col3:
    st.metric("Portfolio Return", f"{portfolio_return:.2%}")

st.write(f"TWR gesamt: {twr_total:.2%}")
st.write(f"CAGR: {cagr:.2%}")
st.write(f"Volatilität: {vol:.2%}")
st.write(f"Sharpe: {sharpe:.2f}")
st.write(f"Sortino: {sortino:.2f}")
st.write(f"Max DD: {mdd:.2%}")