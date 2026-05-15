import plotly.graph_objects as go
import plotly.io as pio

GOOGLE_FONTS = (
    "https://fonts.googleapis.com/css2?family=Ubuntu:wght@300;400;500;700"
    "&family=Ubuntu+Mono:wght@400;700&display=swap"
)

BG = "#041818"
BG_STRONG = "#020E0E"
SURFACE = "#072C2C"
SURFACE_2 = "#0D3535"
SURFACE_3 = "#051F1F"
BORDER = "#1A4040"
BORDER_STRONG = "#2A5050"
ACCENT = "#E86F2A"
MENS_COLOUR = "#7DD3FC"
WOMENS_COLOUR = "#A78BFA"
CTRL_COLOUR = "#C6C6C6"


def hex_to_rgba(hex_color, alpha=1.0):
    """`#RRGGBB` → `rgba(r,g,b,alpha)` string for Plotly fillcolor / shape fills."""
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
TEXT = "#E2F0EF"
MUTED = "#8FB5B5"
SUCCESS = "#16A34A"
WARNING = "#D97706"
DANGER = "#DC2626"


def register_plotly_template():
    pio.templates["causal_dark"] = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor=SURFACE,
            plot_bgcolor=SURFACE_2,
            font=dict(family="Ubuntu, sans-serif", color=TEXT, size=12),
            colorway=[MENS_COLOUR, WOMENS_COLOUR, CTRL_COLOUR, ACCENT, "#5EEAD4", "#FBBF24"],
            xaxis=dict(
                gridcolor=BORDER,
                linecolor=BORDER,
                zerolinecolor=BORDER_STRONG,
                tickfont=dict(color=MUTED),
                title_font=dict(color=MUTED, size=11)
            ),
            yaxis=dict(
                gridcolor=BORDER,
                linecolor=BORDER,
                zerolinecolor=BORDER_STRONG,
                tickfont=dict(color=MUTED),
                title_font=dict(color=MUTED, size=11)
            ),
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)",
                font=dict(color=MUTED)
            ),
            title=dict(
                font=dict(family="Ubuntu, sans-serif", color=TEXT, size=14),
                pad=dict(l=0)
            ),
        )
    )


PLOTLY_TEMPLATE = "causal_dark"
GRAPH_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 3}}

CARD_STYLE = {
    "backgroundColor": SURFACE,
    "border": f"1px solid {BORDER}",
    "borderRadius": "4px",
}
KPI_LABEL_STYLE = {
    "fontSize": "0.78rem",
    "color": MUTED,
    "fontFamily": "Ubuntu, sans-serif",
    "fontWeight": "500",
    "marginBottom": "0.15rem",
}
KPI_VALUE_STYLE = {
    "fontSize": "1.7rem",
    "fontWeight": "700",
    "fontFamily": "Ubuntu, sans-serif",
    "letterSpacing": "-0.02em",
    "lineHeight": "1.1",
    "marginBottom": "0.25rem",
    "color": TEXT,
}
KPI_DELTA_STYLE = {
    "fontSize": "0.8rem",
    "fontFamily": "Ubuntu, sans-serif",
    "marginBottom": "0",
}
SECTION_HEADER_STYLE = {
    "fontFamily": "Ubuntu, sans-serif",
    "fontWeight": "600",
    "fontSize": "0.86rem",
    "color": MUTED,
    "borderBottom": f"1px solid {BORDER}",
    "paddingBottom": "0.5rem",
    "marginBottom": "1.35rem",
}

TABLE_CELL = {
    "backgroundColor": SURFACE,
    "color": TEXT,
    "border": f"1px solid {BORDER}",
    "textAlign": "left",
    "padding": "8px 12px",
    "fontFamily": "Ubuntu, sans-serif",
    "fontSize": "0.85rem",
}
TABLE_HEADER = {
    "backgroundColor": BG,
    "fontWeight": "600",
    "color": MUTED,
    "fontFamily": "Ubuntu, sans-serif",
    "fontSize": "0.74rem",
    "border": f"1px solid {BORDER}",
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
