"""Landing page — cards linking to every registered page, plus contributors."""
import dash
import dash_mantine_components as dmc

from capital.theme import NAVY

dash.register_page(__name__, path="/", name="Home", order=0)

CONTRIBUTORS = [
    ("Mathis Makarski", "Founder of the Team, creator of the dashboard"),
    ("Nicolas Wellers", "Portfolio Manager"),
]


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

    contributor_cards = [
        dmc.Paper(
            [
                dmc.Text(name, fw=600, c=NAVY),
                dmc.Text(role, size="sm", c="dimmed", mt=4),
            ],
            withBorder=True, radius="md", p="lg",
        )
        for name, role in CONTRIBUTORS
    ]

    return dmc.Stack(
        [
            dmc.Title("Capital Team Dashboard", order=1, c=NAVY),
            dmc.Text("Portfolio analytics for the Aachen Investment Club — "
                     "returns, risk, factor models and market monitors.", c="dimmed"),
            dmc.SimpleGrid(cards, cols={"base": 1, "sm": 2, "lg": 3}, mt="lg"),

            dmc.Divider(mt="xl"),
            dmc.Title("Contributors", order=3, c=NAVY, mt="md"),
            dmc.SimpleGrid(contributor_cards, cols={"base": 1, "sm": 2}, mt="sm"),
        ],
        gap="xs",
    )
