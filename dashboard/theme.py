import plotly.graph_objects as go
import plotly.io as pio

GOOGLE_FONTS = (
    "https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,300;8..60,400;8..60,500;8..60,600;8..60,700"
    "&family=IBM+Plex+Mono:wght@400;500;600&display=swap"
)

# Light editorial palette: warm off-white paper, near-black ink, one restrained
# forest-green accent. Names are kept stable so imports across callbacks/figures
# don't break; only the values change.
BG = "#FAF8F4"          # warm off-white "paper"
BG_STRONG = "#F4F1EA"   # subtle warm panel (masthead / control strips)
SURFACE = "#FFFFFF"
SURFACE_2 = "#F4F1EA"
SURFACE_3 = "#FBFAF7"
BORDER = "#E2DDD3"       # hairline
BORDER_STRONG = "#C9C2B5"
ACCENT = "#1E5C4F"       # deep forest green
MENS_COLOUR = "#2F6E8F"  # muted steel blue
WOMENS_COLOUR = "#7E5A86"  # muted plum
CTRL_COLOUR = "#9A9182"  # warm gray

# Per-arm display metadata: callbacks across tabs all repeat the same
# (label, colour) lookup for the two treatment arms. Single source here.
ARM_META = {
    "mens": ("Men's Email", MENS_COLOUR),
    "womens": ("Women's Email", WOMENS_COLOUR),
}


def hex_to_rgba(hex_color, alpha=1.0):
    """`#RRGGBB` → `rgba(r,g,b,alpha)` string for Plotly fillcolor / shape fills."""
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
TEXT = "#1F1B16"        # warm near-black ink
MUTED = "#6B655C"
SUCCESS = "#2E6B4F"      # muted green (positive delta)
WARNING = "#B07A2E"      # muted ocher
DANGER = "#9B3D2E"       # muted brick (negative delta)

SERIF = "Source Serif 4, Georgia, serif"
MONO = "IBM Plex Mono, monospace"


def register_plotly_template():
    pio.templates["causal_light"] = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor=SURFACE,
            plot_bgcolor=SURFACE,
            font=dict(family=SERIF, color=TEXT, size=12),
            colorway=[MENS_COLOUR, WOMENS_COLOUR, CTRL_COLOUR, ACCENT, WARNING, "#6B655C"],
            xaxis=dict(
                gridcolor=BORDER,
                linecolor=BORDER_STRONG,
                zerolinecolor=BORDER_STRONG,
                tickfont=dict(family=MONO, color=MUTED, size=11),
                title_font=dict(family=SERIF, color=MUTED, size=12)
            ),
            yaxis=dict(
                gridcolor=BORDER,
                linecolor=BORDER_STRONG,
                zerolinecolor=BORDER_STRONG,
                tickfont=dict(family=MONO, color=MUTED, size=11),
                title_font=dict(family=SERIF, color=MUTED, size=12)
            ),
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)",
                font=dict(family=SERIF, color=MUTED)
            ),
            title=dict(
                font=dict(family=SERIF, color=TEXT, size=14),
                pad=dict(l=0)
            ),
        )
    )


PLOTLY_TEMPLATE = "causal_light"
GRAPH_CONFIG = {
    "displaylogo": False,
    "toImageButtonOptions": {"format": "png", "scale": 3},
}

FIGURE_MARGIN = dict(t=50, b=30)       # standard
FIGURE_MARGIN_WIDE = dict(t=50, b=70)  # room for bottom legend

# Cards are no longer filled-and-bordered boxes; they read as quiet surfaces
# separated from the paper by a single hairline. The left-accent-border motif
# is gone.
CARD_STYLE = {
    "backgroundColor": SURFACE,
    "border": f"1px solid {BORDER}",
    "borderRadius": "2px",
}
KPI_LABEL_STYLE = {
    "fontSize": "0.8rem",
    "color": MUTED,
    "fontFamily": SERIF,
    "fontWeight": "500",
    "marginBottom": "0.2rem",
}
KPI_VALUE_STYLE = {
    "fontSize": "1.7rem",
    "fontWeight": "600",
    "fontFamily": MONO,
    "fontVariantNumeric": "tabular-nums",
    "letterSpacing": "-0.01em",
    "lineHeight": "1.1",
    "marginBottom": "0.25rem",
    "color": TEXT,
}
KPI_DELTA_STYLE = {
    "fontSize": "0.82rem",
    "fontFamily": MONO,
    "marginBottom": "0",
}
# Editorial subhead: a kicker in small-caps serif over a thin rule.
SECTION_HEADER_STYLE = {
    "fontFamily": SERIF,
    "fontWeight": "600",
    "fontSize": "1.02rem",
    "color": TEXT,
    "borderBottom": f"1px solid {BORDER_STRONG}",
    "paddingBottom": "0.4rem",
    "marginBottom": "1.35rem",
}

TABLE_CELL = {
    "backgroundColor": SURFACE,
    "color": TEXT,
    "border": "none",
    "borderBottom": f"1px solid {BORDER}",
    "textAlign": "left",
    "padding": "9px 12px",
    "fontFamily": MONO,
    "fontSize": "0.84rem",
}
TABLE_HEADER = {
    "backgroundColor": SURFACE,
    "fontWeight": "600",
    "color": MUTED,
    "fontFamily": SERIF,
    "fontSize": "0.76rem",
    "textTransform": "uppercase",
    "letterSpacing": "0.04em",
    "border": "none",
    "borderBottom": f"1.5px solid {BORDER_STRONG}",
}
TABLE_SELECTED = [
    {"if": {"state": "active"}, "backgroundColor": SURFACE_2, "border": f"1px solid {ACCENT}"},
    {"if": {"state": "selected"}, "backgroundColor": SURFACE_2, "border": f"1px solid {BORDER}"},
]

COVARIATE_LABELS = {
    "recency": "Recency (months)",
    "history": "History ($)",
    "mens": "Mens catalogue",
    "womens": "Womens catalogue",
    "zip_suburban": "Zip: Suburban",
    "zip_rural": "Zip: Rural",
    "channel_web": "Channel: Web",
    "channel_multichannel": "Channel: Multichannel",
    "newbie": "New customer",
}
