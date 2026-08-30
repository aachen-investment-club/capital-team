"""Landing page, cards linking to every registered page, plus contributors."""
import dash
import dash_mantine_components as dmc

from capital.theme import NAVY

dash.register_page(__name__, path="/", name="Home", order=0)

# name, role, LinkedIn profile URL, email
CONTRIBUTORS = [
    ("Mathis Makarski", "Founder of the Team, creator of the dashboard",
     "https://www.linkedin.com/in/mathis-makarski", "mathis.makarski@aic.rwth-aachen.de"),
    ("Nicolas Wellers", "Portfolio Manager",
     "https://www.linkedin.com/in/nicolas-wellers/", "nicolas.wellers@aic.rwth-aachen.de"),
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
                dmc.Group(
                    [
                        dmc.Anchor("LinkedIn", href=linkedin, target="_blank",
                                   underline="hover", size="sm") if linkedin else None,
                        dmc.Anchor("Email", href=f"mailto:{email}",
                                   underline="hover", size="sm") if email else None,
                    ],
                    gap="md", mt="sm",
                ),
            ],
            withBorder=True, radius="md", p="lg",
        )
        for name, role, linkedin, email in CONTRIBUTORS
    ]

    return dmc.Stack(
        [
            dmc.Title("Capital Team Dashboard", order=1, c=NAVY),
            dmc.Text("Portfolio analytics for the Aachen Investment Club: "
                     "returns, risk, factor models and market monitors.", c="dimmed"),
            dmc.SimpleGrid(cards, cols={"base": 1, "sm": 2, "lg": 3}, mt="lg"),

            dmc.Divider(mt="xl"),
            dmc.Title("Contributors", order=3, c=NAVY, mt="md"),
            dmc.SimpleGrid(contributor_cards, cols={"base": 1, "sm": 2}, mt="sm"),
        ],
        gap="xs",
    )
