"""Landing page — cards linking to every registered page."""
import dash
import dash_mantine_components as dmc

from capital.theme import NAVY

dash.register_page(__name__, path="/", name="Home", order=0)


def layout():
    cards = [
        dmc.Anchor(
            dmc.Paper(
                [
                    dmc.Text(p["name"], fw=600, c=NAVY),
                    dmc.Text(p.get("description") or "", size="sm", c="dimmed", mt=4),
                ],
                withBorder=True, radius="md", p="lg",
                className="home-card",
            ),
            href=p["path"], underline="never",
        )
        for p in sorted(dash.page_registry.values(), key=lambda p: p.get("order", 99))
        if p["path"] != "/"
    ]
    return dmc.Stack(
        [
            dmc.Title("AIC Capital Dashboard", order=1, c=NAVY),
            dmc.Text("Portfolio analytics for the Aachen Investment Club — "
                     "returns, risk, factor models and market monitors.", c="dimmed"),
            dmc.SimpleGrid(cards, cols={"base": 1, "sm": 2, "lg": 3}, mt="lg"),
        ],
        gap="xs",
    )
