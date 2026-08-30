"""
COPY-PASTE PAGE TEMPLATE, the whole teammate-facing pattern.

1. Copy this file to capital/dashboard/pages/my_page.py (no leading underscore, underscore files are not registered).
2. Adjust register_page(): path, name (navbar label), order (navbar position),
   description (home-page card).
3. Build the layout from dmc components; get data ONLY via capital.data.loaders
   (never import boto3/duckdb in a page) and math via capital.analytics.
4. Callbacks take small inputs (ticker, dates) and return figures, never move
   DataFrames through the browser. Loaders are cached server-side.
5. Heavy math, chosen by how heavy:
   - Seconds: run it synchronously in the callback (see pages/volatility.py).
   - Tens of seconds and up: submit it to the job queue (capital.jobs) and poll,
     as pages/factor_screen.py does. Work runs in a subprocess, so it cannot
     make the dashboard sluggish, results survive a reload and are visible to
     everyone, and a crash cannot take the app down.
   Do NOT use background=True: dash-mantine-components 2.8.0's typed props
   aren't picklable by dill, so DiskcacheManager crashes on any callback
   returning a dmc component. The job queue sidesteps this entirely, because
   jobs move plain JSON and never components.

Charts pick up the "capital" template automatically (capital.theme is imported
by the app). Wrap graphs with components.graph() for the shared PNG export.
"""
import dash
import dash_mantine_components as dmc
import plotly.express as px
from dash import Input, Output, callback

from capital.dashboard import components as ui
from capital.data import loaders

dash.register_page(
    __name__,
    path="/my-page",
    name="My Page",
    order=99,
    description="One line about what this page shows.",
)


def layout():
    master = loaders.get_security_master()
    return dmc.Stack([
        ui.page_title("My Page", "What this page shows and how to read it."),
        dmc.Select(
            id="my-page-ticker",
            label="Security",
            data=sorted(master["ticker"]),
            value=master["ticker"].iloc[0] if len(master) else None,
            searchable=True, w=280,
        ),
        ui.graph("my-page-chart"),
    ])


@callback(Output("my-page-chart", "figure"), Input("my-page-ticker", "value"))
def update_chart(ticker):
    master = loaders.get_security_master()
    row = master[master["ticker"] == ticker]
    sid = row["security_id"].iloc[0]
    df = loaders.get_eod_prices(sid)
    return px.line(df, x="date", y="adj_close", title=f"{ticker}, adjusted close")
