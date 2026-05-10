import plotly.graph_objects as go
import plotly.io as pio

GOOGLE_FONTS = (
    "https://fonts.googleapis.com/css2?family=Ubuntu:wght@300;400;500;700"
    "&family=Oswald:wght@400;500;600;700"
    "&family=Ubuntu+Mono:wght@400;700&display=swap"
)

BG = "#041818"
SURFACE = "#072C2C"
SURFACE_2 = "#0D3535"
BORDER = "#1A4040"
ACCENT = "#FF5F03"
MENS_COLOUR = "#22D3EE"
WOMENS_COLOUR = "#B1C17E"
CTRL_COLOUR = "#C6C6C6"
TEXT = "#E2F0EF"
MUTED = "#6B9090"
SUCCESS = "#16A34A"
WARNING = "#D97706"
DANGER = "#DC2626"


def register_plotly_template():
    pio.templates["enterprise_dark"] = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor=SURFACE,
            plot_bgcolor=SURFACE_2,
            font=dict(family="Ubuntu, sans-serif", color=TEXT, size=12),
            colorway=[MENS_COLOUR, WOMENS_COLOUR, CTRL_COLOUR, ACCENT, "#A78BFA", "#FBBF24"],
            xaxis=dict(
                gridcolor=BORDER,
                linecolor=BORDER,
                zerolinecolor="#2A5050",
                tickfont=dict(color=MUTED),
                title_font=dict(color=MUTED, size=11)
            ),
            yaxis=dict(
                gridcolor=BORDER,
                linecolor=BORDER,
                zerolinecolor="#2A5050",
                tickfont=dict(color=MUTED),
                title_font=dict(color=MUTED, size=11)
            ),
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)",
                font=dict(color=MUTED)
            ),
            title=dict(
                font=dict(family="Oswald, sans-serif", color=TEXT, size=14),
                pad=dict(l=0)
            ),
        )
    )


PLOTLY_TEMPLATE = "enterprise_dark"
GRAPH_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 3}}

CARD_STYLE = {
    "backgroundColor": SURFACE,
    "border": f"1px solid {BORDER}",
    "borderRadius": "4px",
}
KPI_LABEL_STYLE = {
    "fontSize": "0.8rem",
    "color": MUTED,
    "textTransform": "uppercase",
    "letterSpacing": "0.1em",
    "fontFamily": "Ubuntu Mono, monospace",
    "marginBottom": "0.15rem",
}
KPI_VALUE_STYLE = {
    "fontSize": "1.7rem",
    "fontWeight": "700",
    "fontFamily": "Oswald, sans-serif",
    "letterSpacing": "0.02em",
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
    "fontFamily": "Oswald, sans-serif",
    "fontWeight": "500",
    "fontSize": "0.78rem",
    "letterSpacing": "0.1em",
    "textTransform": "uppercase",
    "color": MUTED,
    "borderBottom": f"1px solid {BORDER}",
    "paddingBottom": "0.5rem",
    "marginBottom": "1rem",
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
    "letterSpacing": "0.01em",
    "textTransform": "none",
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
