"""AppShell: brand header + navbar generated from the Dash page registry.

Adding a page file with dash.register_page() automatically adds its nav entry —
there is no central page list to maintain.
"""
import dash
import dash_mantine_components as dmc
from dash import ALL, Input, Output, callback, dcc, html

from capital.theme import MANTINE_THEME, NAVY


def _nav_links() -> list:
    pages = sorted(dash.page_registry.values(), key=lambda p: p.get("order", 99))
    return [
        dmc.NavLink(
            id={"type": "nav-link", "path": p["path"]},
            label=p["name"],
            href=p["path"],
            active=False,
            styles={
                "label": {"color": "#CBD5E1", "fontSize": "14px"},
                "root": {"borderRadius": "6px"},
            },
        )
        for p in pages
    ]


def build_shell():
    header = dmc.AppShellHeader(
        dmc.Group(
            [
                html.Img(src="/assets/logo-white.png", style={"height": "34px"}),
                dmc.Title("Capital Dashboard", order=4, c="white"),
            ],
            h="100%", px="md", gap="sm",
        ),
        style={"backgroundColor": NAVY, "border": "none"},
    )

    navbar = dmc.AppShellNavbar(
        dmc.ScrollArea(dmc.Stack(_nav_links(), gap=4, p="sm"), h="100%"),
        style={"backgroundColor": NAVY, "border": "none"},
    )

    return dmc.MantineProvider(
        theme=MANTINE_THEME,
        children=dmc.AppShell(
            [
                dcc.Location(id="shell-url"),
                header,
                navbar,
                dmc.AppShellMain(dash.page_container),
            ],
            header={"height": 56},
            navbar={"width": 230, "breakpoint": "sm", "collapsed": {"mobile": True}},
            padding="lg",
        ),
    )


@callback(
    Output({"type": "nav-link", "path": ALL}, "active"),
    Input("shell-url", "pathname"),
)
def _highlight_active(pathname):
    # ctx.outputs_list carries the pattern-matched ids in order
    return [o["id"]["path"] == (pathname or "/") for o in dash.ctx.outputs_list]
