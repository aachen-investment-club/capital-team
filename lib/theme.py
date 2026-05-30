import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ── Brand palette ────────────────────────────────────────────────────────────
NAVY       = "#0C1E40"   # sidebar, headings
BLUE       = "#2563EB"   # primary accent
BLUE_DARK  = "#1D4ED8"   # hover / borders
BLUE_MID   = "#3B82F6"   # secondary lines
BLUE_LIGHT = "#EFF6FF"   # card fills
BLUE_BORDER= "#BFDBFE"   # card borders
WHITE      = "#FFFFFF"
GRAY_BG    = "#F8FAFC"
BORDER     = "#E2E8F0"
TEXT       = "#0F172A"
TEXT_MUTED = "#64748B"

# Chart categorical sequence: lead with blues, then complementary accents
_PALETTE = [
    BLUE,      # #2563EB
    "#10B981", # emerald  — positive / second series
    "#F59E0B", # amber    — third series
    BLUE_MID,  # #3B82F6
    "#8B5CF6", # violet
    "#EF4444", # red      — risk / negative
    "#06B6D4", # cyan
    "#F97316", # orange
    "#84CC16", # lime
    TEXT_MUTED,
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
            linecolor=BORDER, tickcolor=BORDER, tickfont=dict(color=TEXT_MUTED),
        ),
        yaxis=dict(
            gridcolor=BORDER, showgrid=True, zeroline=False,
            linecolor=BORDER, tickcolor=BORDER, tickfont=dict(color=TEXT_MUTED),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, borderwidth=1,
            font=dict(color=TEXT_MUTED),
        ),
        margin=dict(l=60, r=20, t=48, b=48),
        hoverlabel=dict(bgcolor=WHITE, bordercolor=BLUE_BORDER, font_size=12, font_color=TEXT),
    )
)

pio.templates["capital"] = _TEMPLATE
pio.templates.default = "capital"

# ── PNG export config (1200 × 675 @ 2× — report quality) ────────────────────
PNG_CONFIG = dict(
    toImageButtonOptions=dict(
        format="png",
        filename="chart",
        height=675,
        width=1200,
        scale=2,
    )
)


# ── CSS injection ────────────────────────────────────────────────────────────
def inject_css() -> None:
    """Call once per page in set_page_config() scope to apply the brand theme."""
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
        border-bottom: 2px solid {BLUE};
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
        background-color: {BLUE};
        color: {WHITE};
        border: none;
        border-radius: 6px;
        font-weight: 500;
        padding: 0.4rem 1.1rem;
        transition: background-color 0.15s;
    }}
    .stButton > button:hover {{
        background-color: {BLUE_DARK};
        color: {WHITE};
    }}

    /* ── Multiselect pills ───────────────────────────────────────── */
    .stMultiSelect [data-baseweb="tag"] {{
        background-color: {BLUE_LIGHT};
        border: 1px solid {BLUE_BORDER};
        color: {BLUE_DARK};
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
