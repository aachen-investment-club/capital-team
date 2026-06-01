import pathlib

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

_LOGO = str(pathlib.Path(__file__).parent.parent / "assets" / "logo-icon.png")
FAVICON = _LOGO

# ── Brand palette ────────────────────────────────────────────────────────────
NAVY       = "#0C1E40"   # sidebar, headings
BLUE       = "#3A8AC4"   # primary accent (medium blue)
BLUE_DARK  = "#2563EB"   # hover / borders
BLUE_MID   = "#60A5FA"   # secondary lines
BLUE_LIGHT = "#EFF6FF"   # card fills
BLUE_BORDER= "#BFDBFE"   # card borders
WHITE      = "#FFFFFF"
GRAY_BG    = "#F8FAFC"
BORDER     = "#E2E8F0"
TEXT       = "#0F172A"
TEXT_MUTED = "#64748B"

# ── Chart colour palette ─────────────────────────────────────────────────────
# Blues anchor the brand; slates/greys give breathing room between series;
# teal and indigo provide distinct accents without clashing.
# Alternating warm and cool greys keeps adjacent series readable.
_PALETTE = [
    "#5A80D2",  # cobalt blue       — portfolio / primary series
    "#64748B",  # slate             — 2nd series / benchmarks
    "#0EA5E9",  # sky blue          — 3rd series
    "#334155",  # dark slate        — 4th series
    "#0D9488",  # teal              — 5th series
    "#94A3B8",  # blue-grey         — 6th series
    "#6366F1",  # indigo            — 7th series
    "#475569",  # slate-700         — 8th series
    "#0891B2",  # ocean blue        — 9th series
    "#CBD5E1",  # pale slate        — 10th series
]

# ── Stable colour maps ───────────────────────────────────────────────────────
# Hardcode every known series so colour never depends on what else is visible.

BENCHMARK_COLORS: dict[str, str] = {
    "PORTFOLIO":   _PALETTE[0],  # cobalt blue  — always the primary line
    "SPX":         _PALETTE[1],  # slate
    "MSCI_WORLD":  _PALETTE[2],  # sky blue
    "MSCI_EUROPE": _PALETTE[3],  # dark slate
    "60_40":       _PALETTE[4],  # teal
}

# Human-readable labels for ticker codes — used in chart legends and exports.
DISPLAY_NAMES: dict[str, str] = {
    "PORTFOLIO":   "AIC Portfolio",
    "60_40":       "60/40 Balanced",
    "SPX":         "S&P 500",
    "MSCI_WORLD":  "MSCI World",
    "MSCI_EUROPE": "MSCI Europe",
}


def position_color_map(symbols: list[str]) -> dict[str, str]:
    """
    Return a stable symbol → colour mapping.
    Colours are assigned by alphabetical order of the full symbol set so that
    adding or hiding one series never shifts the colours of the others.
    """
    return {s: _PALETTE[i % len(_PALETTE)] for i, s in enumerate(sorted(set(symbols)))}


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

# ── PNG export ───────────────────────────────────────────────────────────────
# Toolbar button (interactive — same styles as screen, scale doubles resolution)
PNG_CONFIG = dict(
    toImageButtonOptions=dict(format="png", filename="chart",
                              height=800, width=800, scale=2,
                              setBackground="transparent"),
)

# Export style settings — tweak here, applied by download_png()
EXPORT = dict(
    font_size  = 22,    # axis ticks, labels, legend
    line_width = 3,     # line traces (px)
    width      = 1400,  # output px
    height     = 800,   # output px
)


def apply_export_style(fig):
    """
    Return a deep copy of fig with print-ready styling:
    larger fonts, thicker lines, and a clean white background.
    Edit EXPORT above to tune all three.
    """
    import copy
    ef = copy.deepcopy(fig)
    fs = EXPORT["font_size"]
    ef.update_layout(
        paper_bgcolor=WHITE,
        plot_bgcolor=GRAY_BG,
        font=dict(size=fs),
        xaxis=dict(tickfont=dict(size=fs), title_font=dict(size=fs)),
        yaxis=dict(tickfont=dict(size=fs), title_font=dict(size=fs)),
        legend=dict(font=dict(size=fs)),
    )
    # Thicken every line trace
    ef.for_each_trace(
        lambda t: t.update(line=dict(width=EXPORT["line_width"]))
        if t.type in ("scatter", "scattergl") and t.mode and "lines" in t.mode
        else None
    )
    return ef


def download_png(fig, filename: str = "chart") -> None:
    """
    Streamlit download button that renders fig with print-ready styling via kaleido.
    Requires Chrome — run once in your terminal: uv run plotly_get_chrome
    """
    import plotly.io as _pio
    ef = apply_export_style(fig)
    try:
        png = _pio.to_image(
            ef, format="png",
            width=EXPORT["width"], height=EXPORT["height"], scale=1,
        )
        st.download_button(
            label="↓ Export PNG",
            data=png,
            file_name=f"{filename}.png",
            mime="image/png",
            key=f"dl_{filename}",
        )
    except Exception:
        st.caption("⚠ Export requires Chrome — run `uv run plotly_get_chrome` once in your terminal.")


# ── CSS injection ────────────────────────────────────────────────────────────
def inject_css() -> None:
    """Call once per page in set_page_config() scope to apply the brand theme."""
    # Re-register on every render so font/colour changes in this file take effect
    # immediately without restarting the Streamlit server.
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
        background: transparent;
        color: {BLUE};
        border-color: {BLUE};
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
