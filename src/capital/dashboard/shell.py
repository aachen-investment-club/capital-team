"""AppShell: brand header + navbar generated from the Dash page registry.

Adding a page file with dash.register_page() automatically adds its nav entry —
there is no central page list to maintain. Home is the one exception: it lives
behind the header logo instead of the nav list (see _nav_links).
"""
import dash
import dash_mantine_components as dmc
from dash import ALL, Input, Output, callback, dcc, html

from capital.theme import MANTINE_THEME, NAVY, SIDEBAR_BG

HEADER_HEIGHT = 72
NAVBAR_WIDTH = 230
WEBSITE_URL = "https://www.aachen-investment-club.de/teams/capital"


def _nav_links() -> list:
    pages = sorted(dash.page_registry.values(), key=lambda p: p.get("order", 99))
    return [
        dmc.NavLink(
            id={"type": "nav-link", "path": p["path"]},
            label=p["name"],
            href=p["path"],
            active=False,
            styles={
                "label": {"color": NAVY, "fontSize": "14px"},
                "root": {"borderRadius": "6px"},
            },
        )
        for p in pages
        if p["path"] != "/"  # Home is reached via the header logo, not the list
    ]


def _navbar_state(opened: bool) -> dict:
    return {
        "width": NAVBAR_WIDTH,
        "breakpoint": "sm",
        "collapsed": {"mobile": not opened, "desktop": not opened},
    }


def build_shell():
    header = dmc.AppShellHeader(
        html.Div(
            [
                dmc.Group(
                    [
                        dmc.Burger(id="navbar-burger", opened=True, size="sm", color="white"),
                        dmc.Title("Capital Team Dashboard", order=4, c="white"),
                    ],
                    gap="sm",
                    style={"zIndex": 1},
                ),
                dmc.Anchor(
                    html.Img(src="/assets/logo-white.png", style={"height": "52px", "display": "block"}),
                    href="/",
                    style={
                        "position": "absolute", "left": "50%", "top": "50%",
                        "transform": "translate(-50%, -50%)",
                    },
                ),
                dmc.Anchor(
                    dmc.Group(
                        [dmc.Text("Club Website", size="sm", c="#CBD5E1"), dmc.Text("↗", c="#CBD5E1")],
                        gap=4,
                    ),
                    href=WEBSITE_URL, target="_blank", underline="never",
                    style={"zIndex": 1},
                ),
            ],
            style={
                "position": "relative", "height": "100%",
                "display": "flex", "alignItems": "center", "justifyContent": "space-between",
                "padding": "0 20px",
            },
        ),
        style={"backgroundColor": NAVY, "border": "none"},
    )

    navbar = dmc.AppShellNavbar(
        dmc.ScrollArea(dmc.Stack(_nav_links(), gap=4, p="sm"), h="100%"),
        style={"backgroundColor": SIDEBAR_BG, "border": "none"},
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
            header={"height": HEADER_HEIGHT},
            navbar=_navbar_state(True),
            padding="lg",
            id="app-shell",
        ),
    )


@callback(
    Output({"type": "nav-link", "path": ALL}, "active"),
    Input("shell-url", "pathname"),
)
def _highlight_active(pathname):
    # ctx.outputs_list carries the pattern-matched ids in order
    return [o["id"]["path"] == (pathname or "/") for o in dash.ctx.outputs_list]


@callback(
    Output("app-shell", "navbar"),
    Input("navbar-burger", "opened"),
)
def _toggle_navbar(opened):
    return _navbar_state(opened)
