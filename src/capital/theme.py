"""Brand theme: palette, the "capital" Plotly template (registered at import),
and the Mantine theme for the Dash app."""
import plotly.graph_objects as go
import plotly.io as pio

# ── Brand palette ────────────────────────────────────────────────────────────
NAVY       = "#0C1E40"   # header, headings
SIDEBAR_BG = "#EFF6FF"   # sidebar — light steel blue-gray; light bg needs dark nav-link text
HOVER      = "#DDE1E7"   # button / interactive hover fill
BLUE_MID   = "#60A5FA"   # secondary lines
BLUE_LIGHT = "#EFF6FF"   # card fills
BLUE_BORDER= "#BFDBFE"   # card borders
WHITE      = "#FFFFFF"
GRAY_BG    = "#F8FAFC"
BORDER     = "#E2E8F0"
TEXT       = "#0F172A"
TEXT_MUTED = "#64748B"

# ── Chart colour palette ─────────────────────────────────────────────────────
# Global fallback palette for plotly charts if not other colors specified in pages
_PALETTE = [
    "#0C1E40",  # navy              — primary series
    "#B8962E",  # antique gold      — 2nd series
    "#2E7D6B",  # deep sage green   — 3rd series
    "#8B2E4A",  # burgundy          — 4th series
    "#3A6B9C",  # steel blue        — 5th series
    "#A0522D",  # sienna copper     — 6th series
    "#4A5568",  # charcoal          — 7th series
    "#1A6B8A",  # petrol            — 8th series
    "#7B6FA0",  # muted violet      — 9th series
    "#94A3B8",  # pale slate        — 10th series
]

# ── Plotly template ──────────────────────────────────────────────────────────
_TEMPLATE = go.layout.Template(
    layout=dict(
        font=dict(family="Inter, system-ui, sans-serif", color=TEXT, size=13),
        paper_bgcolor=WHITE,
        plot_bgcolor=GRAY_BG,
        colorway=_PALETTE,
        title=dict(font=dict(size=15, color=NAVY, family="Inter, system-ui, sans-serif")),
        xaxis=dict(
            gridcolor=BORDER, showgrid=True, zeroline=False,
            linecolor=BORDER, tickcolor=BORDER,
            tickfont=dict(color=TEXT_MUTED, size=15),
            title_font=dict(size=15, color=TEXT_MUTED),
        ),
        yaxis=dict(
            gridcolor=BORDER, showgrid=True, zeroline=False,
            linecolor=BORDER, tickcolor=BORDER,
            tickfont=dict(color=TEXT_MUTED, size=15),
            title_font=dict(size=15, color=TEXT_MUTED),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, borderwidth=1,
            font=dict(color=TEXT_MUTED, size=15),
        ),
        margin=dict(l=60, r=20, t=48, b=48),
        hoverlabel=dict(bgcolor=WHITE, bordercolor=BLUE_BORDER, font_size=13, font_color=TEXT),
    )
)

# Register the template globally — importing capital.theme is enough for any
# figure (Dash callbacks, scripts, notebooks) to pick up the brand template.
pio.templates["capital"] = _TEMPLATE
pio.templates.default = "capital"

# ── Mantine theme (Dash app) ─────────────────────────────────────────────────
# 10-shade scale required by Mantine; index 6 ≈ brand navy (used for filled UI).
_NAVY_SCALE = [
    "#EFF6FF", "#DBEAFE", "#BFDBFE", "#93C5FD", "#3A6B9C",
    "#1E3A5F", "#0C1E40", "#0A1936", "#08142C", "#060F22",
]

MANTINE_THEME = {
    "fontFamily": "Inter, system-ui, sans-serif",
    "primaryColor": "navy",
    "colors": {"navy": _NAVY_SCALE},
    "headings": {
        "fontFamily": "Inter, system-ui, sans-serif",
        "fontWeight": "600",
    },
    "defaultRadius": "md",
}

# Plotly PNG-export config shared by every dcc.Graph
GRAPH_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}
