"""
Universe - what securities we cover and what data we actually hold for them.

Replaces the old per-security Equities page. That page answered "show me this
stock"; this one also answers "what is in the universe, and where are the holes",
which is the question that matters once the master grows past a hundred
hand-curated names into a thousand index constituents.

Three tabs: Composition (what the universe is made of), Data coverage (what the
store holds, and what is missing), and Security (the per-name price detail the
old page provided).
"""

import math
from datetime import date, timedelta

import dash
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dash_table, dcc, html
from plotly.subplots import make_subplots

from capital.dashboard import components as ui
from capital.data import loaders
from capital.theme import BORDER, GRAPH_CONFIG, NAVY, TEXT_MUTED

dash.register_page(
    __name__,
    path="/universe",
    name="Universe",
    order=2,
    description="Every security we cover, what data we hold for each, "
    "and per-security price history.",
)

PERIODS = ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "All"]

#: Fixed categorical order, never cycled (validated for colour-vision separation).
SERIES = ["#0C1E40", "#B8962E", "#2E7D6B", "#8B2E4A", "#3A6B9C"]
POS, NEG = "#B8962E", "#3A6B9C"
UP, DOWN = "#2E7D6B", "#8B2E4A"

_GROUPINGS = {
    "Sector": "gics_sector",
    "Country": "country",
    "Currency": "currency",
    "Asset type": "asset_type",
}

#: Descriptor columns, in the order the factor model blends them into styles.
_DESCRIPTOR_LABELS = {
    "market_cap": "Market cap",
    "pb_ratio": "Price / book",
    "pe_ratio": "Price / earnings",
    "ps_ratio": "Price / sales",
    "pcf_ratio": "Price / cash flow",
    "dividend_yield": "Dividend yield",
    "roe": "Return on equity",
    "roa": "Return on assets",
    "gross_margin": "Gross margin",
    "debt_to_ev": "Debt / enterprise value",
    "revenue_ttm": "Revenue",
    "eps_ttm": "Earnings per share",
    "shares_outstanding": "Shares outstanding",
}


# ── Figures ───────────────────────────────────────────────────────────────────


def _fig(
    fig: go.Figure,
    title: str,
    subtitle: str = "",
    height: int = 400,
    left: int = 60,
    bottom: int = 52,
) -> go.Figure:
    text = f"<b>{title}</b>"
    if subtitle:
        text += f"<br><span style='font-size:12px;color:{TEXT_MUTED}'>{subtitle}</span>"
    fig.update_layout(
        template="capital",
        height=height,
        title=dict(text=text, x=0, xanchor="left", pad=dict(b=10)),
        margin=dict(l=left, r=40, t=80 if subtitle else 62, b=bottom),
    )
    # The brand template ticks at 15px; ui.axis_margin budgets for
    # ui.AXIS_FONT_SIZE. Pin both axes to that size, or long category labels get
    # clipped by exactly the difference.
    fig.update_yaxes(tickfont=dict(size=ui.AXIS_FONT_SIZE))
    fig.update_xaxes(tickfont=dict(size=ui.AXIS_FONT_SIZE))
    return fig


def _count_bar(counts: pd.Series, title: str, subtitle: str) -> go.Figure:
    s = counts.sort_values()
    fig = go.Figure(
        go.Bar(
            y=s.index.tolist(),
            x=s.values,
            orientation="h",
            marker=dict(color=SERIES[0], line=dict(width=2, color="#FFFFFF")),
            text=[f"{v:,}" for v in s.values],
            textposition="outside",
            textfont=dict(size=12, color="#334155"),
            cliponaxis=False,
            hovertemplate="%{y}: %{x:,} securities<extra></extra>",
        )
    )
    fig = _fig(
        fig, title, subtitle, max(320, 26 * len(s) + 150), left=ui.axis_margin(s.index)
    )
    fig.update_layout(bargap=0.28, xaxis=dict(title="Securities"))
    fig.update_xaxes(range=[0, float(s.max()) * 1.18])
    return fig


def _coverage_bar(pct: pd.Series, title: str, subtitle: str) -> go.Figure:
    s = pct.sort_values()
    colours = [POS if v >= 0.75 else ("#C08A1E" if v >= 0.4 else NEG) for v in s.values]
    fig = go.Figure(
        go.Bar(
            y=s.index.tolist(),
            x=s.values,
            orientation="h",
            marker=dict(color=colours, line=dict(width=2, color="#FFFFFF")),
            text=[f"{v:.0%}" for v in s.values],
            textposition="outside",
            textfont=dict(size=12, color="#334155"),
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.1%} of securities<extra></extra>",
        )
    )
    fig = _fig(
        fig, title, subtitle, max(320, 26 * len(s) + 150), left=ui.axis_margin(s.index)
    )
    fig.update_layout(
        bargap=0.28,
        xaxis=dict(
            title="Share of securities with data", tickformat=".0%", range=[0, 1.16]
        ),
    )
    return fig


def _history_hist(years: pd.Series) -> go.Figure:
    vals = years[years > 0]
    fig = go.Figure(
        go.Histogram(
            x=vals,
            nbinsx=24,
            marker=dict(color=SERIES[0], line=dict(width=1, color="#FFFFFF")),
            hovertemplate="%{y} securities with ~%{x:.1f}y<extra></extra>",
        )
    )
    fig = _fig(
        fig,
        "Depth of price history",
        "Securities with no history at all are excluded from this chart",
        380,
    )
    fig.update_layout(
        xaxis=dict(title="Years of daily history"),
        yaxis=dict(title="Securities"),
        bargap=0.06,
    )
    return fig


def _table(df: pd.DataFrame, id: str, page_size: int = 20, height: int = 560) -> html.Div:
    numeric = {c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    # Column ids stay snake_case for callbacks; only the visible name is prettied.
    headers = {"gics_sector": "Sector", "px_days": "Price Days",
               "px_years": "Years Of History", "px_first": "Data From",
               "px_last": "Data Through", "fund_days": "Fundamental Days",
               "asset_type": "Asset Type", "with_prices": "With Prices",
               "median_years": "Median Years"}
    return html.Div(
        dash_table.DataTable(
            id=id,
            data=df.to_dict("records"),
            columns=[
                {
                    "name": headers.get(c, c.replace("_", " ").title()),
                    "id": c,
                    "type": "numeric" if c in numeric else "text",
                }
                for c in df.columns
            ],
            sort_action="native",
            filter_action="native",
            page_size=page_size,
            style_table={"overflowX": "auto", "maxHeight": height, "overflowY": "auto"},
            style_cell={
                "fontFamily": "Inter, system-ui, sans-serif",
                "fontSize": "13px",
                "padding": "6px 10px",
                "border": f"1px solid {BORDER}",
            },
            style_header={
                "backgroundColor": "#EFF6FF",
                "fontWeight": 600,
                "color": NAVY,
                "border": f"1px solid {BORDER}",
            },
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#F8FAFC"}
            ],
        ),
        className="factor-table-wrap",
    )

    # ── Layout ────────────────────────────────────────────────────────────────────


def layout():
    master = loaders.get_security_master()
    if master.empty:
        return ui.alert(
            "No securities are configured yet.",
            "yellow",
        )
    return dmc.Stack(
        [
            ui.page_title(
                "Universe",
                "Every security we cover, what data we hold for each, and "
                "per-security price history.",
            ),
            html.Div(id="uni-headline"),
            dmc.Tabs(
                id="uni-tabs",
                value="composition",
                mt="md",
                children=[
                    dmc.TabsList(
                        [
                            dmc.TabsTab("Composition", value="composition"),
                            dmc.TabsTab("Data coverage", value="coverage"),
                            dmc.TabsTab("Search", value="search"),
                            dmc.TabsTab("Security", value="security"),
                        ]
                    ),
                    dmc.TabsPanel(
                        dmc.Stack(
                            [
                                dmc.SegmentedControl(
                                    id="uni-group",
                                    value="Sector",
                                    data=list(_GROUPINGS),
                                    mt="sm",
                                ),
                                dcc.Loading(
                                    html.Div(
                                        id="uni-composition", style={"minHeight": "320px"}
                                    ),
                                    delay_show=300,
                                    overlay_style={
                                        "visibility": "visible",
                                        "opacity": 0.45,
                                    },
                                ),
                            ],
                            gap="sm",
                            mt="md",
                        ),
                        value="composition",
                    ),
                    dmc.TabsPanel(
                        dmc.Stack(
                            [
                                dcc.Loading(
                                    html.Div(
                                        id="uni-coverage", style={"minHeight": "320px"}
                                    ),
                                    delay_show=300,
                                    overlay_style={
                                        "visibility": "visible",
                                        "opacity": 0.45,
                                    },
                                ),
                            ],
                            gap="sm",
                            mt="md",
                        ),
                        value="coverage",
                    ),
                    dmc.TabsPanel(
                        dmc.Stack(
                            [
                                dmc.Group(
                                    [
                                        dmc.TextInput(
                                            id="uni-search",
                                            label="Search",
                                            placeholder="Ticker, name, sector, country or currency",
                                            leftSection=ui.SEARCH_ICON,
                                            w=460,
                                        ),
                                        dmc.MultiSelect(
                                            id="uni-search-type",
                                            label="Asset type",
                                            data=["COMMON", "ETF", "INDEX"],
                                            placeholder="All types",
                                            w=240,
                                        ),
                                        dmc.Switch(
                                            id="uni-search-hasdata",
                                            label="Only securities with data",
                                            checked=False,
                                            mt=30,
                                        ),
                                    ],
                                    align="start",
                                    gap="md",
                                ),
                                dcc.Loading(
                                    html.Div(
                                        id="uni-search-results",
                                        style={"minHeight": "320px"},
                                    ),
                                    delay_show=300,
                                    overlay_style={"visibility": "visible", "opacity": 0.45},
                                ),
                            ],
                            gap="sm",
                            mt="md",
                        ),
                        value="search",
                    ),
                    dmc.TabsPanel(
                        dmc.Stack(
                            [
                                dmc.Group(
                                    [
                                        ui.security_select("uni-security", master, w=380),
                                        dmc.SegmentedControl(
                                            id="uni-period", data=PERIODS, value="1Y", mt=24
                                        ),
                                        dmc.Checkbox(
                                            id="uni-volume",
                                            label="Volume",
                                            checked=True,
                                            mt=30,
                                        ),
                                    ],
                                    align="start",
                                    gap="md",
                                ),
                                dmc.Text(id="uni-history-note", size="sm", c="dimmed"),
                                html.Div(id="uni-metrics"),
                                ui.graph("uni-chart"),
                            ],
                            gap="sm",
                            mt="md",
                        ),
                        value="security",
                    ),
                ],
            ),
        ],
        gap="xs",
    )

    # ── Headline tiles ────────────────────────────────────────────────────────────


@callback(Output("uni-headline", "children"), Input("uni-tabs", "value"))
def _headline(_tab):
    cov = loaders.get_data_coverage()
    if cov.empty:
        return ui.alert("No securities are configured yet.", "yellow")
    n = len(cov)
    priced = int((cov.get("px_days", pd.Series(dtype=int)) > 0).sum())
    with_fund = int((cov.get("fund_days", pd.Series(dtype=int)) > 0).sum())
    years = cov.get("px_years", pd.Series(dtype=float))
    median_years = float(years[years > 0].median()) if (years > 0).any() else 0.0
    first = (
        cov["px_first"].min()
        if "px_first" in cov and cov["px_first"].notna().any()
        else None
    )
    last = (
        cov["px_last"].max() if "px_last" in cov and cov["px_last"].notna().any() else None
    )

    tiles = ui.metric_row(
        [
            ("Securities", f"{n:,}", "covered by the dashboard"),
            (
                "With price history",
                f"{priced:,} / {n:,}",
                f"{priced / n:.0%} of the universe" if n else "",
            ),
            (
                "With fundamentals",
                f"{with_fund:,} / {n:,}",
                f"{with_fund / n:.0%} of the universe" if n else "",
            ),
            ("Median history", f"{median_years:.1f}y", "across securities that have any"),
            (
                "History spans",
                f"{first:%Y}" + (f" to {last:%Y}" if last is not None else "")
                if first is not None
                else "no data",
                "daily price observations",
            ),
        ]
    )
    if priced < n:
        return dmc.Stack(
            [
                tiles,
                ui.alert(
                    f"{n - priced:,} of {n:,} securities have no price history "
                    f"loaded yet, so the factor model cannot see them. They become "
                    f"available once the next data load covers them.",
                    "yellow",
                ),
            ],
            gap="sm",
        )
    return tiles

    # ── Composition tab ───────────────────────────────────────────────────────────


@callback(
    Output("uni-composition", "children"),
    Input("uni-tabs", "value"),
    Input("uni-group", "value"),
)
def _composition(tab, grouping):
    if tab != "composition":
        return dash.no_update
    cov = loaders.get_data_coverage()
    col = _GROUPINGS.get(grouping, "gics_sector")
    if col not in cov.columns:
        return ui.alert(f"No {grouping.lower()} information available.", "yellow")
    counts = cov[col].fillna("Unknown").replace("", "Unknown").value_counts()

    priced = cov[cov.get("px_days", 0) > 0]
    by_group = (
        pd.DataFrame(
            {
                "Covered": counts,
                "With data": priced[col]
                .fillna("Unknown")
                .replace("", "Unknown")
                .value_counts(),
            }
        )
        .fillna(0)
        .astype(int)
        .sort_values("Covered")
    )

    fig = go.Figure()
    for i, colname in enumerate(["Covered", "With data"]):
        fig.add_trace(
            go.Bar(
                y=by_group.index.tolist(),
                x=by_group[colname],
                orientation="h",
                name=colname,
                marker=dict(color=SERIES[i], line=dict(width=2, color="#FFFFFF")),
                hovertemplate=f"{colname} · %{{y}}: %{{x:,}}<extra></extra>",
            )
        )
    fig = _fig(
        fig,
        f"Universe by {grouping.lower()}",
        "Securities we cover against those we hold data for",
        max(360, 30 * len(by_group) + 190),
        left=ui.axis_margin(by_group.index),
    )
    fig.update_layout(
        barmode="group",
        bargap=0.26,
        bargroupgap=0.08,
        xaxis=dict(title="Securities"),
        legend=dict(orientation="h", yanchor="top", y=-0.14, x=0),
    )
    fig.update_layout(margin=dict(b=fig.layout.margin.b + 44))

    table = (
        cov.groupby(cov[col].fillna("Unknown").replace("", "Unknown"))
        .agg(
            securities=("security_id", "count"),
            with_prices=("px_days", lambda s: int((s > 0).sum())),
            median_years=(
                "px_years",
                lambda s: round(float(s[s > 0].median()), 1) if (s > 0).any() else 0.0,
            ),
        )
        .reset_index()
        .rename(columns={col: grouping.lower()})
        .sort_values("securities", ascending=False)
    )

    return dmc.Stack(
        [
            dcc.Graph(figure=fig, config=GRAPH_CONFIG, id="uni-fig-composition"),
            dcc.Graph(
                figure=_history_hist(cov.get("px_years", pd.Series(dtype=float))),
                config=GRAPH_CONFIG,
                id="uni-fig-history",
            ),
            ui.section("Breakdown"),
            _table(table, "uni-composition-table", page_size=15, height=420),
        ],
        gap="sm",
    )

    # ── Coverage tab ──────────────────────────────────────────────────────────────


@callback(Output("uni-coverage", "children"), Input("uni-tabs", "value"))
def _coverage(tab):
    if tab != "coverage":
        return dash.no_update
    cov = loaders.get_data_coverage()
    if cov.empty:
        return ui.alert("No securities are configured yet.", "yellow")

    has_cols = [c for c in cov.columns if c.startswith("has_")]
    n = len(cov)
    pct = pd.Series(
        {
            _DESCRIPTOR_LABELS.get(c[4:], c[4:].replace("_", " ").title()): float(
                (cov[c] > 0).sum()
            )
            / n
            for c in has_cols
        }
    )
    # A retired column that is present in the schema but never populated is noise.
    pct = pct[pct > 0] if (pct > 0).any() else pct

    detail = cov[
        [
            "ticker",
            "name",
            "asset_type",
            "gics_sector",
            "country",
            "px_days",
            "px_years",
            "px_first",
            "px_last",
            "fund_days",
        ]
    ].copy()
    for c in ("px_first", "px_last"):
        detail[c] = pd.to_datetime(detail[c]).dt.date.astype(str).replace("NaT", "")
    detail["status"] = np.where(
        detail["px_days"] == 0,
        "no price data",
        np.where(cov["fund_days"] == 0, "prices only", "prices + fundamentals"),
    )
    detail = detail.sort_values(["px_days", "ticker"])

    status_counts = detail["status"].value_counts()

    return dmc.Stack(
        [
            dmc.SimpleGrid(
                [
                    dcc.Graph(
                        figure=_count_bar(
                            status_counts, "Data status", "Every security we cover"
                        ),
                        config=GRAPH_CONFIG,
                        id="uni-fig-status",
                    ),
                    dcc.Graph(
                        figure=_coverage_bar(
                            pct,
                            "Fundamentals coverage",
                            "Share of securities with at least one "
                            "observation of each descriptor",
                        ),
                        config=GRAPH_CONFIG,
                        id="uni-fig-coverage",
                    ),
                ],
                cols={"base": 1, "lg": 2},
            ),
            ui.section("Per-security coverage"),
            dmc.Text(
                "Sorted so securities with the least data appear first. "
                "Filter the status column to isolate what is still missing data.",
                size="sm",
                c="dimmed",
            ),
            _table(detail, "uni-coverage-table", page_size=25, height=620),
        ],
        gap="sm",
    )

    # ── Security tab ──────────────────────────────────────────────────────────────


def _filter_period(df: pd.DataFrame, p: str) -> pd.DataFrame:
    if p == "All" or df.empty:
        return df
    today = date.today()
    cutoff: date = {
        "1M": today - timedelta(days=30),
        "3M": today - timedelta(days=91),
        "6M": today - timedelta(days=182),
        "YTD": date(today.year, 1, 1),
        "1Y": today - timedelta(days=365),
        "3Y": today - timedelta(days=3 * 365),
        "5Y": today - timedelta(days=5 * 365),
    }[p]
    return df[df["date"] >= cutoff].reset_index(drop=True)


def _empty_fig(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="capital",
        height=380,
        annotations=[
            dict(text=message, showarrow=False, font=dict(size=14, color=TEXT_MUTED))
        ],
    )
    return fig


@callback(
    Output("uni-chart", "figure"),
    Output("uni-metrics", "children"),
    Output("uni-history-note", "children"),
    Input("uni-security", "value"),
    Input("uni-period", "value"),
    Input("uni-volume", "checked"),
)
def _security(security_id, period, show_volume):
    if not security_id:
        return _empty_fig("Pick a security."), [], ""
    master = loaders.get_security_master()
    row = master[master["security_id"] == security_id]
    ticker = row["ticker"].iloc[0] if len(row) else security_id
    name = row["name"].iloc[0] if len(row) else ""

    df = loaders.get_eod_prices(security_id)
    if df.empty:
        return (
            _empty_fig(
                "No price history for this security yet. It will appear "
                "once the next data load covers it."
            ),
            [],
            "",
        )

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    view = _filter_period(df, period)
    note = (
        f"{ticker} · {name} · history {df['date'].min()} to "
        f"{df['date'].max()} ({len(df):,} trading days)"
    )
    if view.empty:
        return _empty_fig(f"No data in the selected period ({period})."), [], note

    all_returns = df["close"].pct_change(fill_method=None).dropna()
    days = max((df["date"].iloc[-1] - df["date"].iloc[0]).days, 1)
    total_ret = df["close"].iloc[-1] / df["close"].iloc[0] - 1
    ann_return = (1 + total_ret) ** (365.25 / days) - 1
    ann_vol = all_returns.std() * math.sqrt(252) if len(all_returns) > 1 else None
    period_ret = view["close"].iloc[-1] / view["close"].iloc[0] - 1
    metrics = ui.metric_row(
        [
            ("Latest close", f"{view['close'].iloc[-1]:,.2f}", str(view["date"].iloc[-1])),
            (f"{period} return", f"{period_ret * 100:+.1f}%", "selected period"),
            ("Ann. return", f"{ann_return * 100:+.1f}%", "full history"),
            (
                "Ann. volatility",
                f"{ann_vol * 100:.1f}%" if ann_vol is not None else "n/a",
                "full history",
            ),
        ],
        cols={"base": 2, "sm": 4},
    )

    if show_volume:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.78, 0.22],
            vertical_spacing=0.03,
        )
    else:
        fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Candlestick(
            x=view["date"],
            open=view["open"],
            high=view["high"],
            low=view["low"],
            close=view["close"],
            name=ticker,
            increasing_line_color=UP,
            decreasing_line_color=DOWN,
            increasing_fillcolor=UP,
            decreasing_fillcolor=DOWN,
        ),
        row=1,
        col=1,
    )
    if show_volume:
        colours = [UP if c >= o else DOWN for c, o in zip(view["close"], view["open"])]
        fig.add_trace(
            go.Bar(
                x=view["date"],
                y=view["volume"],
                name="Volume",
                marker_color=colours,
                marker_opacity=0.55,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="", showticklabels=False, row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig = _fig(fig, f"{ticker}", name, 540 if show_volume else 420)
    fig.update_layout(
        xaxis_rangeslider_visible=False, hovermode="x unified", showlegend=False
    )
    return fig, metrics, note


# ── Search tab ────────────────────────────────────────────────────────────────

_SEARCH_FIELDS = ["ticker", "name", "gics_sector", "country", "currency", "ric", "isin"]


@callback(
    Output("uni-search-results", "children"),
    Input("uni-tabs", "value"),
    Input("uni-search", "value"),
    Input("uni-search-type", "value"),
    Input("uni-search-hasdata", "checked"),
)
def _search(tab, query, asset_types, only_with_data):
    """Free-text lookup across the whole universe.

    Matches every term independently and across all fields, so "uk bank" and
    "bank uk" both work and neither needs to know which column holds which fact.
    """
    if tab != "search":
        return dash.no_update
    cov = loaders.get_data_coverage()
    if cov.empty:
        return ui.alert("No securities are configured yet.", "yellow")

    hits = cov
    if asset_types:
        hits = hits[hits["asset_type"].isin(asset_types)]
    if only_with_data:
        hits = hits[hits["px_days"] > 0]

    terms = [t for t in str(query or "").lower().split() if t]
    if terms:
        haystack = (hits[[c for c in _SEARCH_FIELDS if c in hits.columns]]
                    .fillna("").astype(str).agg(" ".join, axis=1).str.lower())
        for term in terms:
            hits = hits[haystack.loc[hits.index].str.contains(term, regex=False)]

    if hits.empty:
        return ui.alert(f"Nothing matches “{query}”. Try a ticker, a company "
                        f"name, a sector or a country code.", "gray")

    table = hits[["ticker", "name", "asset_type", "gics_sector", "country",
                  "currency", "px_years", "px_last"]].copy()
    table["px_last"] = pd.to_datetime(table["px_last"]).dt.date.astype(str).replace("NaT", "")
    table = table.rename(columns={"px_years": "years_of_history",
                                  "px_last": "data_through"})
    table = table.sort_values(["years_of_history", "ticker"], ascending=[False, True])

    caption = f"{len(hits):,} of {len(cov):,} securities match"
    if terms:
        caption += f" “{query}”"
    return dmc.Stack([
        dmc.Text(caption, size="sm", c="dimmed"),
        _table(table, "uni-search-table", page_size=25, height=620),
    ], gap="xs")
