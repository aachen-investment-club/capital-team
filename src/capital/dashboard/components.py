"""Shared building blocks for dashboard pages."""
import pandas as pd
import dash_mantine_components as dmc
from dash import dcc, html

from capital.theme import GRAPH_CONFIG, NAVY, TEXT_MUTED

_IFRAME_BASE = (
    "<style>@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');"
    "body{margin:0;font-family:'Inter',system-ui,sans-serif;}</style>"
)


def page_title(title: str, info: str) -> dmc.Group:
    """Page H1 with the ℹ help popover (replaces st.title + st.popover)."""
    return dmc.Group(
        [
            dmc.Title(title, order=1, c=NAVY),
            dmc.HoverCard(
                [
                    dmc.HoverCardTarget(dmc.ThemeIcon("ℹ", variant="light", radius="xl", size="md")),
                    dmc.HoverCardDropdown(dmc.Text(info, size="sm", maw=420)),
                ],
                position="right", shadow="md",
            ),
        ],
        gap="sm", align="center", mb="md",
    )


def section(title: str) -> dmc.Title:
    return dmc.Title(title, order=3, c=NAVY, mt="xl", mb="sm")


def graph(id: str, **kwargs) -> dcc.Loading:
    """dcc.Graph with the shared PNG-export config, wrapped in a loading spinner."""
    return dcc.Loading(dcc.Graph(id=id, config=GRAPH_CONFIG, **kwargs), delay_show=300)


def raw_html(src: str, height: int) -> html.Iframe:
    """Render a self-contained styled HTML fragment (legacy *_html helpers)."""
    return html.Iframe(
        srcDoc=_IFRAME_BASE + src,
        style={"width": "100%", "height": f"{height}px", "border": "none", "display": "block"},
    )


def df_table(df: pd.DataFrame, striped: bool = True) -> dmc.Table:
    """A plain Mantine table from a (already formatted) DataFrame."""
    return dmc.Table(
        data={"head": list(df.columns), "body": df.values.tolist()},
        striped=striped, highlightOnHover=True, withTableBorder=True,
        verticalSpacing=6, horizontalSpacing="md", fz="sm",
    )


#: Order asset types are grouped in every securities picker: the things people
#: search for most, first.
_ASSET_GROUPS = [("COMMON", "Equities"), ("ETF", "ETFs & funds"), ("INDEX", "Indices")]


def security_options(master: pd.DataFrame, group: bool = True,
                     value_col: str = "security_id") -> list:
    """Options for a securities picker, grouped by asset type.

    One formatting rule, applied everywhere: the ticker is the identifier people
    type and scan for, so it leads; the company name follows after a spacer for
    recognition. Both remain searchable because Mantine matches on the whole
    label string.
    """
    if master is None or master.empty:
        return []
    df = master.dropna(subset=[value_col]).copy()
    df["_label"] = df["ticker"].astype(str).str.strip() + "\u2003" + df["name"].astype(str).str.strip()
    df = df.sort_values("ticker")
    if not group or "asset_type" not in df.columns:
        return [{"value": v, "label": l} for v, l in zip(df[value_col], df["_label"])]

    groups, seen = [], set()
    for key, title in _ASSET_GROUPS:
        block = df[df["asset_type"] == key]
        if block.empty:
            continue
        seen.add(key)
        groups.append({"group": f"{title} ({len(block)})",
                       "items": [{"value": v, "label": l}
                                 for v, l in zip(block[value_col], block["_label"])]})
    rest = df[~df["asset_type"].isin(seen)]
    if not rest.empty:
        groups.append({"group": f"Other ({len(rest)})",
                       "items": [{"value": v, "label": l}
                                 for v, l in zip(rest[value_col], rest["_label"])]})
    return groups


def security_select(id: str, master: pd.DataFrame, label: str = "Security",
                    value: str | None = None, w: int | str = 360,
                    group: bool = True, clearable: bool = False,
                    description: str | None = None, value_col: str = "security_id",
                    extra: list | None = None, **kwargs) -> dmc.Select:
    """The securities picker, identical on every page that has one.

    Consistency here is the point: someone who learns to find a security on one
    page should not have to relearn it on the next. Search is capped and
    scrollable so a 1,000+ name universe stays responsive in the dropdown.
    """
    options = security_options(master, group=group, value_col=value_col)
    if extra:
        options = [{"group": "Portfolio", "items": list(extra)}, *options] if group \
            else [*extra, *options]
    if value is None and options:
        first = options[0]
        value = first["items"][0]["value"] if "items" in first else first["value"]
    return dmc.Select(
        id=id, label=label, description=description, data=options, value=value,
        searchable=True, clearable=clearable, w=w,
        placeholder="Search by ticker or name",
        nothingFoundMessage="No matching security",
        maxDropdownHeight=340, limit=100,
        comboboxProps={"shadow": "md", "transitionProps": {"duration": 120}},
        leftSection=SEARCH_ICON,
        **kwargs,
    )


#: Magnifier icon as a self-contained data URI. An image rather than a glyph so
#: it renders identically on every platform without pulling in an icon library.
_SEARCH_ICON_URI = (
    "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxNiAxNiIgd2lkdGg9IjE1IiBoZWlnaHQ9IjE1IiBmaWxsPSJub25lIiBzdHJva2U9IiM2NDc0OEIiIHN0cm9rZS13aWR0aD0iMS42IiBzdHJva2UtbGluZWNhcD0icm91bmQiPjxjaXJjbGUgY3g9IjciIGN5PSI3IiByPSI1Ii8+PHBhdGggZD0iTTEwLjggMTAuOCBMMTUgMTUiLz48L3N2Zz4="
)
SEARCH_ICON = html.Img(src=_SEARCH_ICON_URI, width=15, height=15,
                       style={"display": "block", "opacity": 0.75})


# Approximate advance width per character as a fraction of the font size, for
# Inter. Split three ways because a proportional font makes "Communication
# Services" far wider than "Utilities" at the same character count.
_NARROW_CHARS = set("iljtfrI.,:;'\"|!()[]{} ")
_WIDE_CHARS = set("ABCDEFGHJKLMNOPQRSTUVWXYZmwMW@%&")

#: Tick font for categorical axes. Set explicitly wherever axis_margin is used:
#: the brand template defaults to 15px, and a margin computed for one size while
#: the axis renders at another is exactly how labels get clipped.
AXIS_FONT_SIZE = 12


def text_width_px(text: str, font_size: int = AXIS_FONT_SIZE) -> float:
    """Estimated rendered width of a string, in pixels."""
    total = 0.0
    for ch in str(text):
        if ch in _NARROW_CHARS:
            total += 0.31
        elif ch in _WIDE_CHARS:
            total += 0.72
        else:
            total += 0.55
    return total * font_size


def axis_margin(labels, font_size: int = AXIS_FONT_SIZE, pad: int = 26,
                floor: int = 60, cap: int = 300) -> int:
    """Left margin wide enough for the longest category label, plus tick padding.

    Plotly reserves no space for tick text, so a fixed left margin silently
    truncates long labels ("Communication Services" becomes "ommunication
    Services"). Every categorical axis in the dashboard sizes its own margin from
    the actual label text and the actual tick font size.
    """
    widest = max((text_width_px(x, font_size) for x in labels), default=0.0)
    return int(min(cap, max(floor, widest + pad)))


def explain(title: str, body: str, opened: bool = False) -> dmc.Accordion:
    """A collapsed "how this works" panel that expands on click.

    The dashboard is a teaching tool as much as a monitor, so every non-obvious
    chart carries one of these. Collapsed by default: someone who already knows
    what an information coefficient is should not have to scroll past a
    paragraph explaining it, and someone who doesn't should never have to leave
    the page to find out. `body` is markdown and may contain $...$ / $$...$$
    maths, which MathJax renders.
    """
    return dmc.Accordion(
        value=title if opened else None,
        chevronPosition="left", variant="subtle", radius="sm",
        styles={"control": {"paddingLeft": 0}, "content": {"paddingLeft": 28}},
        children=[dmc.AccordionItem(value=title, children=[
            dmc.AccordionControl(dmc.Text(title, size="sm", c=TEXT_MUTED, fw=500)),
            dmc.AccordionPanel(
                dcc.Markdown(body, mathjax=True, className="explain-body",
                             dangerously_allow_html=False)),
        ])],
    )


def metric(label: str, value: str, hint: str = "") -> dmc.Paper:
    """One KPI tile using the shared .metric-card styling."""
    children = [html.Div(label, className="metric-label"),
                html.Div(value, className="metric-value")]
    if hint:
        children.append(dmc.Text(hint, size="xs", c=TEXT_MUTED, mt=2))
    return dmc.Paper(children, className="metric-card")


def metric_row(items: list[tuple[str, str, str]], cols: dict | None = None) -> dmc.SimpleGrid:
    return dmc.SimpleGrid(
        [metric(label, value, hint) for label, value, hint in items],
        cols=cols or {"base": 2, "sm": 3, "lg": 5}, spacing="sm",
    )


def alert(message: str, color: str = "blue", title: str | None = None) -> dmc.Alert:
    return dmc.Alert(message, color=color, variant="light", title=title)


def export_button(id: str, label: str = "Export Figures") -> html.Div:
    """Ghost-style export button + its dcc.Download sink."""
    return html.Div([
        dmc.Button(label, id=id, variant="subtle", color="gray", size="xs"),
        dcc.Download(id=f"{id}-download"),
    ])
