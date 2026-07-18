"""Shared building blocks for dashboard pages."""
import pandas as pd
import dash_mantine_components as dmc
from dash import dcc, html

from capital.theme import GRAPH_CONFIG, NAVY

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


def export_button(id: str, label: str = "Export Figures") -> html.Div:
    """Ghost-style export button + its dcc.Download sink."""
    return html.Div([
        dmc.Button(label, id=id, variant="subtle", color="gray", size="xs"),
        dcc.Download(id=f"{id}-download"),
    ])
