import pathlib

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

_LOGO = str(pathlib.Path(__file__).parent.parent / "assets" / "logo-icon.png")
FAVICON = _LOGO

# ── Brand palette ────────────────────────────────────────────────────────────
NAVY       = "#0C1E40"   # sidebar, headings
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

# ── CSS injection ────────────────────────────────────────────────────────────
def inject_css() -> None:
    """Call once per page in set_page_config() scope to apply the brand theme."""
    pio.templates["capital"] = _TEMPLATE
    pio.templates.default = "capital"
    st.logo(_LOGO, size='large')
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', system-ui, sans-serif;
    }}

    /* ── Sidebar ─────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background-color: {NAVY};
        border-right: none;
    }}
    [data-testid="stSidebar"] * {{
        color: #CBD5E1 !important;
    }}
    [data-testid="stSidebarNav"] a span {{
        color: #93C5FD !important;
    }}
    [data-testid="stSidebarNav"] a:hover span {{
        color: {WHITE} !important;
    }}
    [data-testid="stSidebarNav"] a[aria-current="page"] span {{
        color: {WHITE} !important;
        font-weight: 600;
    }}
    [data-testid="stSidebarHeader"] {{
        border-bottom: 1px solid rgba(255,255,255,0.08);
        padding-bottom: 0.75rem;
    }}

    /* ── Top bar ─────────────────────────────────────────────────── */
    header[data-testid="stHeader"] {{
        background-color: {WHITE};
        border-bottom: 1px solid {BORDER};
    }}

    /* ── Main area ───────────────────────────────────────────────── */
    .stApp > .main {{
        background-color: {WHITE};
    }}
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
    }}

    /* ── Page title ──────────────────────────────────────────────── */
    h1 {{
        color: {NAVY};
        font-weight: 700;
        letter-spacing: -0.02em;
    }}

    /* ── Section subheadings ─────────────────────────────────────── */
    h2, h3 {{
        color: {NAVY};
        font-weight: 600;
        letter-spacing: -0.01em;
        padding-bottom: 6px;
        margin-top: 1.5rem;
    }}

    /* ── Metric cards ────────────────────────────────────────────── */
    [data-testid="stMetric"] {{
        background-color: {BLUE_LIGHT};
        border: 1px solid {BLUE_BORDER};
        border-radius: 10px;
        padding: 1rem 1.25rem;
    }}
    [data-testid="stMetricLabel"] > div {{
        color: {TEXT_MUTED} !important;
        font-size: 0.78rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}
    [data-testid="stMetricValue"] > div {{
        color: {NAVY} !important;
        font-weight: 700;
    }}
    [data-testid="stMetricDelta"] {{
        font-weight: 500;
    }}

    /* ── Divider ─────────────────────────────────────────────────── */
    hr {{
        border: none;
        border-top: 1px solid {BORDER};
        margin: 1.5rem 0;
    }}

    /* ── Buttons ─────────────────────────────────────────────────── */
    .stButton > button {{
        background-color: {GRAY_BG};
        color: {NAVY};
        border: 1px solid {BORDER};
        border-radius: 6px;
        font-weight: 500;
        padding: 0.4rem 1.1rem;
        transition: background-color 0.15s;
    }}
    .stButton > button:hover {{
        background-color: {HOVER};
        color: {NAVY};
    }}

    /* ── Ghost action button (wrap element in <div class="action-ghost">) ── */
    div.action-ghost .stButton > button {{
        background: transparent;
        color: {TEXT_MUTED};
        border: 1px solid {BORDER};
        border-radius: 6px;
        font-size: 12px;
        font-weight: 400;
        padding: 0.25rem 0.75rem;
        letter-spacing: 0.02em;
        transition: color 0.15s, border-color 0.15s;
    }}
    div.action-ghost .stButton > button:hover {{
        background: {HOVER};
        color: {NAVY};
        border-color: {BORDER};
    }}

    /* ── Multiselect pills ───────────────────────────────────────── */
    .stMultiSelect [data-baseweb="tag"] {{
        background-color: {BLUE_LIGHT};
        border: 1px solid {BLUE_BORDER};
        color: {NAVY};
    }}

    /* ── Caption / muted text ────────────────────────────────────── */
    .stCaption, [data-testid="stCaptionContainer"] {{
        color: {TEXT_MUTED} !important;
    }}

    /* ── Dataframe ───────────────────────────────────────────────── */
    [data-testid="stDataFrameResizable"] {{
        border: 1px solid {BORDER};
        border-radius: 8px;
        overflow: hidden;
    }}
    </style>
    """, unsafe_allow_html=True)
