"""
app.py
------
Causal Inference Dashboard - Hillstrom Email Campaign Dataset
6-tab Dash app with enterprise dark theme. Heavy computation is cached to disk.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc

import causal_utils as cu

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------

GOOGLE_FONTS = (
    "https://fonts.googleapis.com/css2?family=Ubuntu:wght@300;400;500;700"
    "&family=Oswald:wght@400;500;600;700"
    "&family=Ubuntu+Mono:wght@400;700&display=swap"
)

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG, GOOGLE_FONTS],
    suppress_callback_exceptions=True,
    title="Causal Inference Dashboard",
)
server = app.server

print("=" * 60)
print("Causal Inference Dashboard")
print("=" * 60)
RESULTS = cu.load_or_build_cache()
DF = RESULTS["df"]
PSM = RESULTS["psm"]
BAYESIAN = RESULTS["bayesian"]
UPLIFT = RESULTS["uplift"]
OLS = RESULTS["ols"]
print('Dashboard ready!')

# ---------------------------------------------------------------------------
# Enterprise design tokens
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Custom Plotly template
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

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
SECTION_HEADER = SECTION_HEADER_STYLE

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
    "fontFamily": "Ubuntu Mono, monospace",
    "fontSize": "0.7rem",
    "letterSpacing": "0.06em",
    "textTransform": "uppercase",
    "border": f"1px solid {BORDER}",
}

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

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            *, *::before, *::after { box-sizing: border-box; }

            body {
                background-color: #041818 !important;
                font-family: "Ubuntu", sans-serif !important;
                color: #E2F0EF !important;
                -webkit-font-smoothing: antialiased;
                margin: 0;
            }

            ::-webkit-scrollbar { width: 5px; height: 5px; }
            ::-webkit-scrollbar-track { background: #041818; }
            ::-webkit-scrollbar-thumb { background: #1A4040; border-radius: 3px; }
            ::-webkit-scrollbar-thumb:hover { background: #FF5F03; }

            /* Navbar */
            .enterprise-navbar {
                background: #020E0E !important;
                border-bottom: 2px solid #FF5F03 !important;
                padding: 0.55rem 0 !important;
            }

            /* Cards */
            .card {
                background-color: #072C2C !important;
                border: 1px solid #1A4040 !important;
                border-radius: 4px !important;
            }
            .card-body { padding: 0.9rem 1rem !important; }
            .card-header {
                background-color: #051F1F !important;
                border-bottom: 1px solid #1A4040 !important;
                color: #E2F0EF !important;
                font-family: "Oswald", sans-serif !important;
                font-weight: 500 !important;
                font-size: 0.78rem !important;
                letter-spacing: 0.1em !important;
                text-transform: uppercase !important;
                padding: 0.6rem 1rem !important;
            }

            /* Tab nav */
            .nav-tabs {
                border-bottom: 1px solid #1A4040 !important;
                padding: 0 0.5rem;
                background-color: #020E0E !important;
            }
            .nav-tabs .nav-link {
                color: #6B9090 !important;
                background: transparent !important;
                border: none !important;
                border-bottom: 2px solid transparent !important;
                border-radius: 0 !important;
                padding: 0.65rem 1.1rem !important;
                font-family: "Oswald", sans-serif !important;
                font-weight: 400 !important;
                font-size: 0.78rem !important;
                letter-spacing: 0.08em !important;
                text-transform: uppercase !important;
                transition: color 0.15s, border-color 0.15s;
                white-space: nowrap;
            }
            .nav-tabs .nav-link:hover {
                color: #E2F0EF !important;
                border-bottom-color: #2A5050 !important;
            }
            .nav-tabs .nav-link.active {
                color: #FF5F03 !important;
                background: transparent !important;
                border-bottom: 2px solid #FF5F03 !important;
            }
            .tab-content { background-color: #041818 !important; }

            /* Accordion */
            .accordion-button {
                background-color: #072C2C !important;
                color: #E2F0EF !important;
                font-family: "Ubuntu", sans-serif !important;
                font-size: 0.87rem !important;
                box-shadow: none !important;
            }
            .accordion-button:not(.collapsed) {
                background-color: #0D3535 !important;
                color: #FF5F03 !important;
            }
            .accordion-body {
                background-color: #072C2C !important;
                color: #9ABABA !important;
                font-size: 0.87rem !important;
                line-height: 1.7 !important;
                border-top: 1px solid #1A4040 !important;
            }
            .accordion-item {
                background-color: #072C2C !important;
                border: 1px solid #1A4040 !important;
                margin-bottom: 4px !important;
            }

            /* Methodology button */
            .btn-methodology {
                background: transparent !important;
                border: 1px solid #1A4040 !important;
                color: #6B9090 !important;
                font-family: "Ubuntu Mono", monospace !important;
                font-size: 0.72rem !important;
                letter-spacing: 0.06em !important;
                padding: 0.3rem 0.85rem !important;
                border-radius: 3px !important;
                cursor: pointer;
                transition: all 0.15s;
            }
            .btn-methodology:hover,
            .btn-methodology:focus {
                border-color: #FF5F03 !important;
                color: #FF5F03 !important;
                background: rgba(255,95,3,0.06) !important;
                outline: none !important;
                box-shadow: none !important;
            }

            /* Forms */
            .form-check-input:checked {
                background-color: #FF5F03 !important;
                border-color: #FF5F03 !important;
            }
            .form-check-input:focus {
                box-shadow: 0 0 0 0.2rem rgba(255,95,3,0.25) !important;
            }
            .form-control, .form-select {
                background-color: #0D3535 !important;
                border: 1px solid #1A4040 !important;
                color: #E2F0EF !important;
            }
            .form-control:focus, .form-select:focus {
                background-color: #0D3535 !important;
                border-color: #FF5F03 !important;
                color: #E2F0EF !important;
                box-shadow: 0 0 0 0.15rem rgba(255,95,3,0.2) !important;
            }
            .input-group-text {
                background-color: #0D3535 !important;
                border: 1px solid #1A4040 !important;
                color: #6B9090 !important;
                font-family: "Ubuntu Mono", monospace !important;
                font-size: 0.82rem !important;
            }
            .form-check-label {
                font-size: 0.85rem !important;
                color: #9ABABA !important;
            }
            label.small { color: #6B9090 !important; }

            /* Methodology card */
            .methodology-content .card-body {
                border-left: 2px solid #1A4040 !important;
                background-color: #051F1F !important;
            }

            .text-muted { color: #6B9090 !important; }

            /* Tooltips (ⓘ info icons) */
            .tooltip .tooltip-inner {
                background-color: rgba(13, 53, 53, 0.92) !important;
                color: #C8E0DF !important;
                border: 1px solid #1A4040 !important;
                border-radius: 4px !important;
                font-family: "Ubuntu", sans-serif !important;
                font-size: 0.78rem !important;
                line-height: 1.5 !important;
                max-width: 260px !important;
                padding: 0.45rem 0.7rem !important;
                backdrop-filter: blur(4px);
                text-align: left !important;
                box-shadow: 0 4px 16px rgba(0,0,0,0.45) !important;
            }
            .tooltip .tooltip-arrow::before {
                border-right-color: #1A4040 !important;
            }

            /* DataTable export button */
            .export {
                font-size: 0 !important;
                background: transparent !important;
                border: 1px solid #1A4040 !important;
                color: #6B9090 !important;
                font-family: "Ubuntu Mono", monospace !important;
                letter-spacing: 0.06em !important;
                padding: 0.3rem 0.85rem !important;
                border-radius: 3px !important;
                cursor: pointer !important;
                transition: all 0.15s !important;
                text-transform: uppercase !important;
                margin-bottom: 0.5rem !important;
            }
            .export::after {
                content: "Export to CSV";
                font-size: 0.72rem;
                font-family: "Ubuntu Mono", monospace;
                letter-spacing: 0.06em;
                text-transform: uppercase;
            }
            .export:hover, .export:focus {
                border-color: #FF5F03 !important;
                color: #FF5F03 !important;
                background: rgba(255, 95, 3, 0.06) !important;
                outline: none !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""

# ---------------------------------------------------------------------------
# Component helpers
# ---------------------------------------------------------------------------


def kpi_card(
    value,
    label,
    delta=None,
    delta_positive=None,
    color=TEXT,
    accent=None,
    info=None,
    info_id=None,
    pct_change=None,
):
    delta_color = (
        SUCCESS if delta_positive else DANGER if delta_positive is False else MUTED
    )
    left_border = accent if accent else BORDER

    label_children = [label]
    extra = []
    if info and info_id:
        label_children += [
            html.Span(
                " ⓘ",
                id=info_id,
                style={
                    "cursor": "help",
                    "color": MUTED,
                    "fontSize": "0.85em",
                    "marginLeft": "3px"
                },
            ),
        ]
        extra = [
            dbc.Tooltip(
                info,
                target=info_id,
                placement="right",
                style={"fontFamily": "Ubuntu, sans-serif", "fontSize": "0.8rem"}
            )
        ]

    left_block = html.Div(
        [
            html.P(label_children, style=KPI_LABEL_STYLE),
            html.P(value, style={**KPI_VALUE_STYLE, "color": color}),
            html.P(delta or " ", style={**KPI_DELTA_STYLE, "color": delta_color}),
        ],
        style={"flex": "1"},
    )

    if pct_change is not None:
        pct_color = SUCCESS if pct_change >= 0 else DANGER
        arrow = "↑" if pct_change >= 0 else "↓"
        right_block = html.Div(
            [
                html.Span(arrow, style={"fontSize": "1.6rem", "lineHeight": "1"}),
                html.Div(
                    f"{abs(pct_change):.1f}%",
                    style={"fontSize": "0.85rem", "fontWeight": "700", "marginTop": "2px"},
                ),
            ],
            style={
                "color": pct_color,
                "textAlign": "center",
                "alignSelf": "center",
                "paddingLeft": "14px",
                "fontFamily": "Ubuntu Mono, monospace",
                "lineHeight": "1.2",
                "minWidth": "52px",
            },
        )
        body_children = [html.Div([left_block, right_block], style={"display": "flex"}), *extra]
    else:
        body_children = [left_block, *extra]

    return dbc.Card(
        dbc.CardBody(body_children),
        style={**CARD_STYLE, "borderLeft": f"3px solid {left_border}"},
        className="mb-2"
    )


def section_header(text):
    return html.H5(text, style=SECTION_HEADER_STYLE)


def methodology_collapse(tab_id, content):
    return html.Div(
        [
            html.Button(
                "▸ Methodology & Assumptions",
                id=f"method-btn-{tab_id}",
                className="btn-methodology mb-2",
                n_clicks=0
            ),
            dbc.Collapse(
                dbc.Card(
                    dbc.CardBody(
                        content,
                        style={
                            "fontSize": "0.85rem",
                            "color": "#9ABABA",
                            "lineHeight": "1.7"
                        },
                    ),
                    className="methodology-content",
                    style={**CARD_STYLE, "marginTop": "4px"}
                ),
                id=f"method-collapse-{tab_id}",
                is_open=False
            ),
        ],
        className="mt-3"
    )


# ---------------------------------------------------------------------------
# Tab layouts
# ---------------------------------------------------------------------------

# ── Tab 1: Overview ──────────────────────────────────────────────────────────


def tab1_layout():
    seg_counts = DF["segment"].value_counts()
    n_mens = seg_counts.get("Mens E-Mail", 0)
    n_womens = seg_counts.get("Womens E-Mail", 0)
    n_control = seg_counts.get("No E-Mail", 0)

    conv_mens = DF[DF["segment"] == "Mens E-Mail"]["conversion"].mean() * 100
    conv_womens = DF[DF["segment"] == "Womens E-Mail"]["conversion"].mean() * 100
    conv_control = DF[DF["segment"] == "No E-Mail"]["conversion"].mean() * 100

    spend_mens = DF[DF["segment"] == "Mens E-Mail"]["spend"]
    spend_womens = DF[DF["segment"] == "Womens E-Mail"]["spend"]
    spend_control = DF[DF["segment"] == "No E-Mail"]["spend"]
    avg_mens, avg_womens, avg_control = spend_mens.mean(), spend_womens.mean(), spend_control.mean()
    lift_mens = avg_mens - avg_control
    lift_womens = avg_womens - avg_control

    def _ci95(a, b, n_boot=2000, seed=cu.RANDOM_SEED):
        rng = np.random.default_rng(seed)
        diffs = np.array([
            rng.choice(a, size=len(a), replace=True).mean() -
            rng.choice(b, size=len(b), replace=True).mean()
            for _ in range(n_boot)
        ])
        return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

    mens_lo, mens_hi = _ci95(spend_mens, spend_control)
    wom_lo, wom_hi = _ci95(spend_womens, spend_control)
    mens_sig = mens_lo > 0
    wom_sig = wom_lo > 0
    proj_mens = lift_mens * n_mens
    proj_womens = lift_womens * n_womens
    proj_mens_lo, proj_mens_hi = mens_lo * n_mens, mens_hi * n_mens
    proj_wom_lo, proj_wom_hi = wom_lo * n_womens, wom_hi * n_womens

    if wom_sig and not mens_sig:
        headline = (
            f"The Women's campaign drove a statistically meaningful lift of "
            f"${lift_womens:.2f} per recipient (bootstrap 95% CI ${wom_lo:.2f}-${wom_hi:.2f}), "
            f"worth roughly ${proj_womens:,.0f} across the {n_womens:,} customers mailed "
            f"(projection CI: ${proj_wom_lo:,.0f}-${proj_wom_hi:,.0f}). "
            f"The Men's campaign shows a smaller lift (${lift_mens:.2f}/recipient) that is not "
            f"distinguishable from zero at 95% confidence — treat as inconclusive."
        )
        headline_color = SUCCESS
    elif mens_sig and wom_sig:
        headline = (
            f"Both campaigns drove statistically meaningful lifts. "
            f"Women's: ${lift_womens:.2f}/recipient (95% CI ${wom_lo:.2f}-${wom_hi:.2f}, "
            f"projected ${proj_womens:,.0f} [${proj_wom_lo:,.0f}-${proj_wom_hi:,.0f}]). "
            f"Men's: ${lift_mens:.2f}/recipient (95% CI ${mens_lo:.2f}-${mens_hi:.2f}, "
            f"projected ${proj_mens:,.0f} [${proj_mens_lo:,.0f}-${proj_mens_hi:,.0f}])."
        )
        headline_color = SUCCESS
    else:
        headline = (
            f"Women's lift: ${lift_womens:.2f}/recipient (bootstrap 95% CI ${wom_lo:.2f}-${wom_hi:.2f}, "
            f"projected ${proj_womens:,.0f} [${proj_wom_lo:,.0f}-${proj_wom_hi:,.0f}] across {n_womens:,} mailed). "
            f"Men's lift: ${lift_mens:.2f}/recipient (bootstrap 95% CI ${mens_lo:.2f}-${mens_hi:.2f}). "
            f"Results warrant further review across methods (see Tab 6)."
        )
        headline_color = WARNING

    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        kpi_card(
                            f"{n_mens:,}",
                            "Men's Email (treated)",
                            color=MENS_COLOUR,
                            accent=MENS_COLOUR
                        ),
                        md=3
                    ),
                    dbc.Col(
                        kpi_card(
                            f"{n_womens:,}",
                            "Women's Email (treated)",
                            color=WOMENS_COLOUR,
                            accent=WOMENS_COLOUR
                        ),
                        md=3
                    ),
                    dbc.Col(
                        kpi_card(
                            f"{n_control:,}",
                            "Control (no email)",
                            color=CTRL_COLOUR,
                            accent=CTRL_COLOUR
                        ),
                        md=3
                    ),
                    dbc.Col(
                        kpi_card(
                            f"{len(DF):,}",
                            "Total customers in test",
                            color=ACCENT,
                            accent=ACCENT
                        ),
                        md=3
                    ),
                ],
                className="mb-3"
            ),
            dbc.Row(
                [
                    dbc.Col(
                        kpi_card(
                            f"${avg_mens:.2f}",
                            "Revenue/recipient: Men",
                            f"+${lift_mens:.2f} vs control"
                            + (" (sig.)" if mens_sig else " (n.s.)"),
                            mens_sig,
                            color=MENS_COLOUR,
                            accent=MENS_COLOUR,
                            info="Avg spend across ALL recipients (incl. non-spenders). This is the metric that drives campaign ROI. 'sig.' means the bootstrap 95% CI excludes zero.",
                            info_id="ov-info-rev-mens",
                            pct_change=(lift_mens / avg_control * 100) if avg_control else None,
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        kpi_card(
                            f"${avg_womens:.2f}",
                            "Revenue/recipient: Women",
                            f"+${lift_womens:.2f} vs control"
                            + (" (sig.)" if wom_sig else " (n.s.)"),
                            wom_sig,
                            color=WOMENS_COLOUR,
                            accent=WOMENS_COLOUR,
                            info="Avg spend across ALL recipients (incl. non-spenders). This is the metric that drives campaign ROI. 'sig.' means the bootstrap 95% CI excludes zero.",
                            info_id="ov-info-rev-womens",
                            pct_change=(lift_womens / avg_control * 100) if avg_control else None,
                        ),
                        md=4
                    ),
                    dbc.Col(
                        kpi_card(
                            f"${avg_control:.2f}",
                            "Revenue/recipient: Control",
                            "baseline",
                            accent=CTRL_COLOUR
                        ),
                        md=4
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        kpi_card(
                            f"{conv_mens:.2f}%",
                            "Conversion Rate: Men",
                            f"+{conv_mens - conv_control:.2f}pp vs control",
                            conv_mens > conv_control,
                            color=MENS_COLOUR,
                            accent=MENS_COLOUR
                        ),
                        md=4
                    ),
                    dbc.Col(
                        kpi_card(
                            f"{conv_womens:.2f}%",
                            "Conversion Rate: Women",
                            f"+{conv_womens - conv_control:.2f}pp vs control",
                            conv_womens > conv_control,
                            color=WOMENS_COLOUR,
                            accent=WOMENS_COLOUR
                        ),
                        md=4
                    ),
                    dbc.Col(
                        kpi_card(
                            f"{conv_control:.2f}%",
                            "Conversion Rate: Control",
                            "baseline",
                            accent=CTRL_COLOUR
                        ),
                        md=4
                    ),
                ],
                className="mb-4"
            ),
            dbc.Card(
                [
                    dbc.CardHeader("Headline Finding"),
                    dbc.CardBody(
                        [
                            html.P(headline, className="mb-3"),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Div(
                                            [
                                                html.Div(
                                                    f"${proj_mens:,.0f}",
                                                    style={
                                                        "fontSize": "1.45rem",
                                                        "fontFamily": "Oswald, sans-serif",
                                                        "fontWeight": "500",
                                                        "color": MENS_COLOUR if mens_sig else MUTED,
                                                        "lineHeight": "1.1",
                                                    }
                                                ),
                                                html.Div(
                                                    [
                                                        "Men's total incremental revenue ",
                                                        html.Span(
                                                            "significant" if mens_sig else "not significant",
                                                            style={
                                                                "fontSize": "0.65rem",
                                                                "color": SUCCESS if mens_sig else DANGER,
                                                                "fontFamily": "Ubuntu Mono, monospace",
                                                                "textTransform": "uppercase",
                                                                "letterSpacing": "0.05em",
                                                            }
                                                        ),
                                                    ],
                                                    style={"fontSize": "0.72rem", "color": MUTED, "marginTop": "3px"}
                                                ),
                                            ],
                                            style={
                                                "borderLeft": f"3px solid {MENS_COLOUR if mens_sig else BORDER}",
                                                "paddingLeft": "0.75rem",
                                            }
                                        ),
                                        md=4
                                    ),
                                    dbc.Col(
                                        html.Div(
                                            [
                                                html.Div(
                                                    f"${proj_womens:,.0f}",
                                                    style={
                                                        "fontSize": "1.45rem",
                                                        "fontFamily": "Oswald, sans-serif",
                                                        "fontWeight": "500",
                                                        "color": WOMENS_COLOUR if wom_sig else MUTED,
                                                        "lineHeight": "1.1",
                                                    }
                                                ),
                                                html.Div(
                                                    [
                                                        "Women's total incremental revenue ",
                                                        html.Span(
                                                            "significant" if wom_sig else "not significant",
                                                            style={
                                                                "fontSize": "0.65rem",
                                                                "color": SUCCESS if wom_sig else DANGER,
                                                                "fontFamily": "Ubuntu Mono, monospace",
                                                                "textTransform": "uppercase",
                                                                "letterSpacing": "0.05em",
                                                            }
                                                        ),
                                                    ],
                                                    style={"fontSize": "0.72rem", "color": MUTED, "marginTop": "3px"}
                                                ),
                                            ],
                                            style={
                                                "borderLeft": f"3px solid {WOMENS_COLOUR if wom_sig else BORDER}",
                                                "paddingLeft": "0.75rem",
                                            }
                                        ),
                                        md=4
                                    ),
                                ],
                                className="mb-3"
                            ),
                            html.P(
                                "Difference-in-means from the randomised experiment; confidence intervals "
                                "use a percentile bootstrap (2,000 resamples) to account for the "
                                "zero-inflated spend distribution. Subsequent tabs stress-test this with "
                                "matching, Bayesian, and uplift methods.",
                                className="text-muted small mb-0"
                            ),
                        ]
                    ),
                ],
                style={**CARD_STYLE, "borderLeft": f"3px solid {headline_color}"},
                className="mb-4"
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            section_header("Spend Distribution by Segment (among spenders)"),
                            dcc.Graph(id="tab1-box", figure=_fig_spend_box(), config=GRAPH_CONFIG)
                        ],
                        md=8
                    ),
                    dbc.Col(
                        [
                            section_header("About this Dataset"),
                            html.P(
                                "The Hillstrom MineThatData dataset captures a randomised email marketing "
                                "experiment across 64k US retail customers. Two treatment groups received "
                                "targeted email campaigns (Men's or Women's catalogue), while the control "
                                "group received nothing.",
                                className="small text-muted"
                            ),
                            html.P(
                                "The causal question: does receiving an email cause customers to spend more? "
                                "Even in a randomised experiment, causal analysis adds value by quantifying "
                                "uncertainty, identifying which customers respond most (HTE), and stress-testing "
                                "results across methodologies.",
                                className="small text-muted"
                            ),
                        ],
                        md=4
                    ),
                ],
                className="mb-4"
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            section_header("Did the randomisation work?"),
                            html.P(
                                [
                                    "Each dot compares the treatment and control groups on a customer attribute "
                                    "(recency, spend history, channel, etc). ",
                                    html.Strong("Dots inside the dashed band = groups are balanced"),
                                    ", meaning any difference in outcomes can be attributed to the email itself "
                                    "rather than pre-existing differences between customers. "
                                    "A randomised experiment should produce this pattern."
                                ],
                                className="small text-muted mb-2"
                            ),
                            dcc.Graph(
                                id="tab1-balance", figure=_fig_covariate_balance(), config=GRAPH_CONFIG
                            ),
                        ]
                    ),
                ]
            ),
            methodology_collapse(
                "tab1",
                [
                    html.P(
                        "This tab presents descriptive statistics. Because this is a randomised experiment, "
                        "groups should be broadly balanced on observed covariates, the balance table should confirm this. "
                        "However, minor imbalances can still bias estimates, motivating the matching and regression "
                        "adjustments in subsequent tabs."
                    ),
                ],
            ),
        ],
        fluid=True,
        className="py-4"
    )


def _fig_spend_box():
    spenders = DF[DF["spend"] > 0].copy()

    seg_order = ["Mens E-Mail", "Womens E-Mail", "No E-Mail"]
    color_map = {
        "Mens E-Mail": MENS_COLOUR,
        "Womens E-Mail": WOMENS_COLOUR,
        "No E-Mail": CTRL_COLOUR
    }

    seg_labels = {
        "Mens E-Mail": "Men's Email",
        "Womens E-Mail": "Women's Email",
        "No E-Mail": "Control"
    }

    fig = go.Figure()
    for seg in seg_order:
        vals = spenders[spenders["segment"] == seg]["spend"].values
        q1, med, q3 = np.percentile(vals, [25, 50, 75], method='linear')
        mean_val = vals.mean()
        fill = color_map[seg]
        fig.add_trace(
            go.Box(
                y=vals,
                name=seg_labels[seg],
                width=0.2,
                marker=dict(
                    color=fill,
                    outliercolor='rgba(0,0,0,0.4)',
                    line=dict(outliercolor='rgba(0,0,0,0.4)', outlierwidth=1)
                ),
                line=dict(color='black', width=1.5),
                fillcolor=fill,
                opacity=0.75,
                boxmean=True,
                boxpoints='outliers',
                customdata=[[q1, med, q3, mean_val] for _ in range(len(vals))],
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Median: $%{customdata[1]:.0f}<br>"
                    "Mean: $%{customdata[3]:.0f}<br>"
                    "IQR: $%{customdata[0]:.0f}-$%{customdata[2]:.0f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        showlegend=False,
        margin=dict(t=30, b=20),
        yaxis_title="Spend ($)",
        xaxis_title="",
        boxgap=0.3
    )
    return fig


def _fig_covariate_balance():
    covs = cu.COVARIATES
    control = DF[DF["segment"] == "No E-Mail"]
    arms = {
        "Men's Email": DF[DF["segment"] == "Mens E-Mail"],
        "Women's Email": DF[DF["segment"] == "Womens E-Mail"]
    }
    colors = {"Men's Email": MENS_COLOUR, "Women's Email": WOMENS_COLOUR}
    symbols = {"Men's Email": "diamond", "Women's Email": "circle"}

    fig = go.Figure()
    for arm_label, arm_df in arms.items():
        smds, labels = [], []
        for cov in covs:
            a = arm_df[cov].values
            b = control[cov].values
            pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
            smd = (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0
            smds.append(smd)
            labels.append(COVARIATE_LABELS.get(cov, cov))

        fig.add_trace(
            go.Scatter(
                x=smds,
                y=labels,
                mode="markers",
                name=arm_label,
                marker=dict(
                    color=colors[arm_label], size=11, symbol=symbols[arm_label]
                ),
                hovertemplate="%{y}<br>SMD: %{x:.3f}<extra>%{fullData.name}</extra>"
            )
        )

    fig.add_vline(
        x=0.1,
        line_dash="dash",
        line_color=WARNING
    )
    fig.add_vline(
        x=-0.1,
        line_dash="dash",
        line_color=WARNING
    )
    fig.add_vline(x=0, line_color=BORDER, line_width=1)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        margin=dict(t=30, b=60, l=160, r=40),
        xaxis_title="Standardised Mean Difference (vs Control)",
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        height=340
    )
    return fig


# ── Tab 2: PSM ───────────────────────────────────────────────────────────────


def tab2_layout():
    return dbc.Container(
        [
            dbc.Card(
                dbc.CardBody(
                    [
                        html.P(
                            [
                                html.Strong("What this does: "),
                                "Each customer who received an email is paired with a control customer "
                                "who looked almost identical beforehand (same purchase history, recency, "
                                "channel and demographics). Comparing paired customers isolates the email's "
                                "effect from any pre-existing differences between groups."
                            ],
                            className="mb-2 small"
                        ),
                        html.P(
                            [
                                html.Strong("How customers are matched: "),
                                "A logistic regression is fitted on all 9 observed covariates (recency, history, "
                                "catalogue interest, channel, zip type, newbie flag) to estimate each customer's ",
                                html.Em("propensity score"),
                                " - the probability they would have received the email given their attributes. "
                                "Each treated customer is then paired with the control customer whose propensity "
                                "score is closest (1:1 nearest-neighbour, with replacement). In a randomised trial "
                                "the score should hover around 0.5 for everyone and the two groups should overlap "
                                "almost completely: that's what the distribution chart on the right shows."
                            ],
                            className="mb-2 small"
                        ),
                        html.P(
                            [
                                html.Strong("Why this matters: "),
                                "Matching sharpens estimates by reducing noise from covariate imbalance. "
                                "The Love Plot below shows balance before and after matching (all dots should "
                                "move inside the ±0.1 band). Uncertainty in the ATT is quantified via 500 "
                                "bootstrap resamples of the matched pairs."
                            ],
                            className="mb-0 small text-muted"
                        ),
                    ]
                ),
                style={**CARD_STYLE, "borderLeft": f"3px solid {ACCENT}"},
                className="mb-3"
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Campaign arm:", className="small text-muted"),
                            dbc.RadioItems(
                                id="psm-arm-selector",
                                options=[
                                    {
                                        "label": "Men's Email vs Control",
                                        "value": "mens"
                                    },
                                    {
                                        "label": "Women's Email vs Control",
                                        "value": "womens"
                                    }
                                ],
                                value="mens",
                                inline=True,
                                className="mb-3"
                            ),
                        ]
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(html.Div(id="psm-kpi-cards"), md=4),
                    dbc.Col(dcc.Graph(id="psm-ps-dist", config=GRAPH_CONFIG), md=8)
                ],
                className="mb-3"
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="psm-love-plot", config=GRAPH_CONFIG), md=7),
                    dbc.Col(dcc.Graph(id="psm-stats-chart", config=GRAPH_CONFIG), md=5)
                ],
                className="mb-3"
            ),
            methodology_collapse(
                "tab2",
                [
                    html.P(
                        "Propensity Score Matching (PSM) estimates the Average Treatment Effect on the Treated (ATT) "
                        "by matching each treated customer to a control customer with a similar probability of "
                        "treatment (propensity score). The propensity score is estimated via logistic regression "
                        "on all observed covariates."
                    ),
                    html.P(
                        "Key assumptions: (1) Ignorability (all confounders are observed and included in the "
                        "propensity model). (2) Common support (treated and control overlap in propensity scores). "
                        "(3) SUTVA (no spillover between customers). The Love Plot shows covariate balance "
                        "before and after matching. Standardised mean differences below 0.1 indicate good balance."
                    ),
                    html.P(
                        "Uncertainty is quantified via a causal bootstrap: 200 replicates, each of "
                        "which resamples the combined treated+control pool, re-fits the propensity "
                        "model, re-matches, and recomputes the ATT. This propagates uncertainty from "
                        "the propensity-score estimation and the matching step. A naive pair-level "
                        "bootstrap would be invalid here because 1:1 nearest-neighbour matching "
                        "with replacement creates dependence between pairs that share a control "
                        "(Abadie & Imbens, 2006)."
                    ),
                    html.P(
                        "Common support is reported as a KPI but not enforced — no caliper is applied, "
                        "so every treated unit finds a match. The number of treated units whose "
                        "propensity lies outside the overlap region is shown alongside the ATT."
                    ),
                ],
            ),
        ],
        fluid=True,
        className="py-4"
    )


# ── Tab 3: Bayesian A/B ───────────────────────────────────────────────────────


def tab3_layout():
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Comparison:", className="small text-muted"),
                            dbc.RadioItems(
                                id="bayes-pair-selector",
                                options=[
                                    {
                                        "label": "Men's vs Control",
                                        "value": "mens_vs_control"
                                    },
                                    {
                                        "label": "Women's vs Control",
                                        "value": "womens_vs_control"
                                    },
                                    {
                                        "label": "Men's vs Women's",
                                        "value": "mens_vs_womens"
                                    }
                                ],
                                value="mens_vs_control",
                                inline=True,
                                className="mb-3"
                            ),
                        ]
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(html.Div(id="bayes-kpi-cards"), md=4),
                    dbc.Col(dcc.Graph(id="bayes-posterior-plot", config=GRAPH_CONFIG), md=8)
                ],
                className="mb-3"
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.Span(
                                        "Minimum meaningful effect",
                                        className="small text-muted"
                                    ),
                                    html.Span(
                                        " ⓘ",
                                        id="rope-info-icon",
                                        style={
                                            "cursor": "help",
                                            "color": MUTED,
                                            "fontSize": "0.85em",
                                            "marginLeft": "4px"
                                        }
                                    ),
                                    dbc.Tooltip(
                                        "The Region of Practical Equivalence (ROPE) defines a band of effect sizes "
                                        "that are too small to act on. Posterior probability inside ±$X per customer "
                                        "is treated as 'practically zero'. Set this to the minimum spend lift that "
                                        "would justify running the campaign.",
                                        target="rope-info-icon",
                                        placement="right",
                                        style={
                                            "fontFamily": "Oswald, sans-serif",
                                            "fontSize": "0.75rem"
                                        }
                                    ),
                                ],
                                style={"marginBottom": "6px"}
                            ),
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText("±$"),
                                    dbc.Input(
                                        id="rope-slider",
                                        type="number",
                                        min=0,
                                        max=10,
                                        step=0.5,
                                        value=1,
                                        debounce=True
                                    ),
                                    dbc.InputGroupText("per customer")
                                ],
                                size="sm",
                                style={"maxWidth": "220px"}
                            ),
                        ],
                        md=5,
                    ),
                    dbc.Col(html.Div(id="rope-result-card"), md=7)
                ],
                className="mb-3"
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Button(
                                "▸ Posterior Predictive Check (conditional spend amount)",
                                id="ppc-btn",
                                className="btn-methodology mb-2",
                                n_clicks=0
                            ),
                            dbc.Collapse(
                                dcc.Graph(id="bayes-ppc-plot", config=GRAPH_CONFIG),
                                id="ppc-collapse",
                                is_open=False
                            ),
                        ]
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Button(
                                "▸ Show MCMC Trace Plots",
                                id="trace-btn",
                                className="btn-methodology mb-2",
                                n_clicks=0
                            ),
                            dbc.Collapse(
                                dcc.Graph(id="bayes-trace-plot", config=GRAPH_CONFIG),
                                id="trace-collapse",
                                is_open=False
                            ),
                        ]
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Button(
                                "▸ MCMC Diagnostics",
                                id="diag-btn",
                                className="btn-methodology mb-2",
                                n_clicks=0
                            ),
                            dbc.Collapse(
                                html.Div(id="bayes-diagnostics-table"),
                                id="diag-collapse",
                                is_open=False
                            )
                        ]
                    ),
                ]
            ),
            methodology_collapse(
                "tab3",
                [
                    html.P(
                        "Spend is ~99% zeros with a right-skewed positive tail, so a plain Normal "
                        "likelihood is a severe misspecification. Instead the model uses a "
                        "two-part (hurdle) specification: a Bernoulli on whether the customer "
                        "spends at all, and a LogNormal on the amount among converters. The "
                        "expected per-customer spend is P(convert) · E[amount | convert], and "
                        "delta is the difference in expected spend between the two arms."
                    ),
                    html.P(
                        "Priors: Beta(1, 1) (uniform) on conversion probability; Normal on "
                        "log-mean centred on the pooled log-amount; HalfNormal on log-sigma. "
                        "MCMC is run with PyMC via the nutpie NUTS sampler (2,000 draws, 2 chains) "
                        "on the full arm data — no subsampling."
                    ),
                    html.P(
                        "The 95% Highest Density Interval (HDI) is the shortest interval containing "
                        "95% of the posterior probability — i.e. a 95% probability the true expected "
                        "spend difference lies in this range (given the model and data)."
                    ),
                    html.P(
                        "The ROPE (Region of Practical Equivalence) lets you define a minimum effect "
                        "size that matters for business decisions. The dashboard shows the probability "
                        "mass outside the ROPE."
                    ),
                ],
            ),
        ],
        fluid=True,
        className="py-4"
    )


# ── Tab 4: Uplift / HTE ───────────────────────────────────────────────────────


def tab4_layout():
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Campaign arm:", className="small text-muted"),
                            dbc.RadioItems(
                                id="uplift-arm-selector",
                                options=[
                                    {"label": "Men's Email", "value": "mens"},
                                    {"label": "Women's Email", "value": "womens"}
                                ],
                                value="mens",
                                inline=True,
                                className="mb-2"
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            html.Label("Model:", className="small text-muted"),
                            dbc.RadioItems(
                                id="uplift-model-selector",
                                options=[
                                    {"label": "T-Learner", "value": "t"},
                                    {"label": "S-Learner", "value": "s"}
                                ],
                                value="t",
                                inline=True,
                                className="mb-2"
                            ),
                        ],
                        md=6
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(html.Div(id="uplift-kpi-cards"), md=4),
                    dbc.Col(dcc.Graph(id="uplift-cate-hist", config=GRAPH_CONFIG), md=8)
                ],
                className="mb-3"
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="uplift-feat-imp", config=GRAPH_CONFIG), md=6),
                    dbc.Col(dcc.Graph(id="uplift-decile-chart", config=GRAPH_CONFIG), md=6)
                ],
                className="mb-3"
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="uplift-qini", config=GRAPH_CONFIG), md=6),
                    dbc.Col(dcc.Graph(id="uplift-segment-compare", config=GRAPH_CONFIG), md=6)
                ],
                className="mb-3"
            ),
            methodology_collapse(
                "tab4",
                [
                    html.P(
                        "Uplift modelling estimates Conditional Average Treatment Effects (CATE): the expected "
                        "causal effect for each individual customer, given their characteristics."
                    ),
                    html.P(
                        "T-Learner trains two separate models: one on treated customers & one on control, "
                        "and computes CATE as the difference in predictions. "
                        "S-Learner trains a single model with treatment as a feature and computes CATE "
                        "by differencing predictions under treatment vs non-treatment."
                    ),
                    html.P(
                        "Both models use 5-fold stratified cross-fitting (stratified on treatment, "
                        "so every fold has both arms represented). Each observation's CATE is "
                        "predicted by a model trained on the other four folds. This avoids "
                        "in-sample overfitting and gives honest out-of-sample estimates."
                    ),
                    html.P(
                        "The Qini curve uses the canonical Radcliffe (2007) definition for continuous "
                        "outcomes: at rank k, cumulative net revenue captured equals the cumulative "
                        "treated spend minus the cumulative control spend re-weighted by the "
                        "treated/control ratio. Higher area under the curve means the model ranks "
                        "high-responders well. The decile chart shows actual spend lift for "
                        "customers ranked by predicted uplift — good models show declining lift."
                    ),
                    html.P(
                        "Feature importance is reported as the absolute difference between the "
                        "T-Learner's treated-outcome and control-outcome random-forest importances. "
                        "Features the two models use differently are the ones driving heterogeneity "
                        "in treatment effect — which is the actual object of interest. Raw "
                        "`estimator_trmnt.feature_importances_` would answer a different question "
                        "(what predicts spend in the treated group)."
                    ),
                    html.P(
                        "S-Learner with a RandomForest and treatment×covariate interactions is known "
                        "to shrink CATE toward zero when outcome variance is large relative to the "
                        "treatment signal — a pattern we see here, where the T-Learner's average "
                        "CATE tends to be larger in magnitude than the S-Learner's."
                    ),
                    html.P(
                        "Key assumption: the same ignorability assumption as PSM: all relevant confounders "
                        "are observed. HTE estimates have wider uncertainty than ATE estimates and should be "
                        "treated as directional rather than precise."
                    ),
                ],
            ),
        ],
        fluid=True,
        className="py-4"
    )


# ── Tab 5: Multi-Arm OLS ─────────────────────────────────────────────────────


def tab5_layout():
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            section_header(
                                "OLS Coefficient Plot: Treatment Effects & Interactions"
                            ),
                            dcc.Graph(id="ols-coef-plot", config=GRAPH_CONFIG)
                        ]
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            section_header("Marginal Effects by Subgroup"),
                            html.Div(id="ols-marginal-table")
                        ]
                    ),
                ],
                className="mb-3"
            ),
            dbc.Row(
                [
                    dbc.Col(section_header("Subgroup Heatmap: Predicted Spend Lift"))
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="ols-heatmap-mens", config=GRAPH_CONFIG), md=6),
                    dbc.Col(dcc.Graph(id="ols-heatmap-womens", config=GRAPH_CONFIG), md=6)
                ]
            ),
            methodology_collapse(
                "tab5",
                [
                    html.P(
                        "OLS regression estimates population-average treatment effects while controlling for "
                        "observed covariates. All three arms (Men, Women, Control) are compared simultaneously "
                        "via treatment dummy variables, avoiding the multiple comparison inflation of separate tests."
                    ),
                    html.P(
                        "Interaction terms (treatment x newbie, treatment x channel, treatment x zip code) capture "
                        "treatment effect heterogeneity at the subgroup level, complementing the "
                        "non-parametric uplift models in Tab 4."
                    ),
                    html.P(
                        "Standard errors use the HC3 heteroscedasticity-robust (White) estimator. "
                        "Spend is right-skewed and its variance scales with the treatment means, so "
                        "default OLS SEs would be biased — HC3 is the recommended small-sample "
                        "correction for this kind of outcome."
                    ),
                    html.P(
                        "The subgroup heatmap collapses the 3-way (newbie × channel × zip) marginal "
                        "effects to a 2-way (newbie × channel) view by averaging over zip, weighted "
                        "by the actual customer counts in each (newbie, channel, zip) cell. An "
                        "unweighted mean would over-represent rare zip categories."
                    ),
                    html.P(
                        "Key assumption: linearity of the conditional expectation function. The model is estimated "
                        "by OLS (not IV or 2SLS) and relies on the same ignorability assumption as PSM."
                    ),
                ],
            ),
        ],
        fluid=True,
        className="py-4"
    )


# ── Tab 6: Method Comparison ──────────────────────────────────────────────────


def tab6_layout():
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            section_header("All Methods Summary"),
                            html.Div(id="comparison-table")
                        ]
                    ),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="forest-plot-mens", config=GRAPH_CONFIG), md=6),
                    dbc.Col(dcc.Graph(id="forest-plot-womens", config=GRAPH_CONFIG), md=6)
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(html.Div(id="key-takeaway-card"), md=5),
                    dbc.Col(
                        [
                            dbc.Accordion(
                                [
                                    dbc.AccordionItem(
                                        [
                                            html.P(
                                                "Estimates the ATT via covariate balancing. Best when you believe all confounders are observed. Sensitive to propensity model misspecification."
                                            ),
                                        ],
                                        title="Propensity Score Matching (PSM)"
                                    ),
                                    dbc.AccordionItem(
                                        [
                                            html.P(
                                                "Provides a full posterior distribution over the treatment effect. Best for communicating uncertainty and making probabilistic decisions. Requires distributional assumptions (Normal likelihood here)."
                                            ),
                                        ],
                                        title="Bayesian A/B Test"
                                    ),
                                    dbc.AccordionItem(
                                        [
                                            html.P(
                                                "Estimates individual-level CATEs, ideal for targeting campaigns to the most responsive customers. Higher variance than ATE estimates. Treat as directional."
                                            ),
                                        ],
                                        title="Uplift / HTE (T-Learner, S-Learner)"
                                    ),
                                    dbc.AccordionItem(
                                        [
                                            html.P(
                                                "Efficient parametric estimate with explicit interaction terms. Assumes linearity. Best for understanding which subgroups drive heterogeneity in a transparent, auditable way."
                                            ),
                                        ],
                                        title="Multi-Arm OLS"
                                    ),
                                ],
                                start_collapsed=True
                            ),
                        ],
                        md=7
                    ),
                ]
            ),
        ],
        fluid=True,
        className="py-4"
    )


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

TABS = [
    dbc.Tab(tab1_layout(), label="1 Overview", tab_id="tab-1"),
    dbc.Tab(tab2_layout(), label="2 Matched-Control", tab_id="tab-2"),
    dbc.Tab(tab3_layout(), label="3 Bayesian A/B", tab_id="tab-3"),
    dbc.Tab(tab4_layout(), label="4 Uplift / HTE", tab_id="tab-4"),
    dbc.Tab(tab5_layout(), label="5 Multi-Arm OLS", tab_id="tab-5"),
    dbc.Tab(tab6_layout(), label="6 Method Comparison", tab_id="tab-6")
]

app.layout = html.Div(
    [
        dbc.Navbar(
            dbc.Container(
                [
                    html.Div(
                        [
                            html.Img(
                                src="/assets/jordan_cheney_logo_dark.png",
                                style={
                                    "height": "28px",
                                    "width": "auto",
                                    "marginRight": "0.85rem",
                                    "filter": "brightness(0) invert(1)",
                                    "opacity": "0.88",
                                }
                            ),
                            html.Span(
                                "Causal",
                                style={
                                    "fontFamily": "Oswald, sans-serif",
                                    "fontWeight": "700",
                                    "fontSize": "1.05rem",
                                    "color": ACCENT,
                                    "letterSpacing": "0.12em",
                                    "textTransform": "uppercase"
                                },
                            ),
                            html.Span(
                                " Inference",
                                style={
                                    "fontFamily": "Oswald, sans-serif",
                                    "fontWeight": "300",
                                    "fontSize": "1.05rem",
                                    "color": TEXT,
                                    "letterSpacing": "0.12em",
                                    "textTransform": "uppercase"
                                },
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center"}
                    ),
                    html.Div(
                        [
                            html.Span(
                                "Hillstrom (2008)",
                                style={
                                    "fontFamily": "Ubuntu Mono, monospace",
                                    "fontSize": "0.7rem",
                                    "color": MUTED,
                                    "letterSpacing": "0.04em"
                                },
                            ),
                            html.Span(
                                " · ", style={"color": BORDER, "margin": "0 0.4rem"}
                            ),
                            html.Span(
                                "64k customers · 3 arms · 6 causal methods",
                                style={
                                    "fontFamily": "Ubuntu Mono, monospace",
                                    "fontSize": "0.7rem",
                                    "color": "#2A5050"
                                },
                            ),
                        ],
                        className="ms-auto d-none d-md-flex align-items-center"
                    ),
                ],
                fluid=True
            ),
            className="enterprise-navbar mb-0",
            sticky="top"
        ),
        dbc.Container(
            [
                dbc.Tabs(TABS, id="main-tabs", active_tab="tab-1"),
            ],
            fluid=True
        ),
    ],
    style={"backgroundColor": BG, "minHeight": "100vh"}
)


# ===========================================================================
# CALLBACKS - Tab 2: PSM
# ===========================================================================


@app.callback(
    Output("psm-kpi-cards", "children"),
    Output("psm-ps-dist", "figure"),
    Output("psm-love-plot", "figure"),
    Output("psm-stats-chart", "figure"),
    Input("psm-arm-selector", "value")
)
def update_psm(arm):
    p = PSM[arm]
    arm_label = "Men's Email" if arm == "mens" else "Women's Email"
    arm_color = MENS_COLOUR if arm == "mens" else WOMENS_COLOUR

    ci_str = f"95% CI: ${p['att_ci_lo']:.2f} - ${p['att_ci_hi']:.2f}"
    ps_dist = p.get("avg_ps_distance")
    dist_str = (
        f"Avg Propensity Score distance: {ps_dist:.3f}"
        if ps_dist is not None
        else "1:1 NN, no caliper"
    )
    cs_str = f"{p['cs_lower']:.3f} - {p['cs_upper']:.3f}"

    kpis = html.Div(
        [
            kpi_card(
                f"${p['att_point']:.2f}",
                f"ATT - {arm_label}",
                ci_str,
                p["att_point"] > 0,
                color=arm_color,
                accent=arm_color,
                info=(
                    "Average Treatment Effect on the Treated (ATT): estimated causal effect of "
                    "receiving the comm on spend (for customers who actually received it). "
                    "It's the mean spend difference between each treated customer and "
                    "their matched-control counterpart."
                ),
                info_id="psm-info-att"
            ),
            kpi_card(
                f"{p['n_matched']:,}",
                "Matched pairs",
                dist_str,
                accent=ACCENT,
                info=(
                    "Each treated customer is paired 1:1 with the most similar control customer "
                    "by propensity score (no caliper applied, so all treated units are matched). "
                    "Average PS distance measures match quality, closer to 0 means tighter matches."
                ),
                info_id="psm-info-pairs"
            ),
            kpi_card(
                cs_str,
                "Common support range",
                info=(
                    "The propensity score range where both treated and control groups overlap. "
                    "Causal estimates are most credible within this range. A wide interval "
                    "indicates good overlap. A narrow one suggests the groups are very different "
                    "and matching might be unreliable at the extremes."
                ),
                info_id="psm-info-cs"
            ),
        ]
    )

    # Propensity score distribution
    ps_fig = go.Figure()
    ps_fig.add_trace(
        go.Histogram(
            x=p["propensity_treated"],
            name=arm_label,
            opacity=0.7,
            nbinsx=50,
            marker_color=arm_color
        )
    )
    ps_fig.add_trace(
        go.Histogram(
            x=p["propensity_control"],
            name="Control",
            opacity=0.7,
            nbinsx=50,
            marker_color=CTRL_COLOUR
        )
    )
    ps_fig.update_layout(
        barmode="overlay",
        template=PLOTLY_TEMPLATE,
        title="Propensity Score Distribution",
        xaxis_title="Propensity score",
        yaxis_title="Count",
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5
        ),
        margin=dict(t=50, b=70)
    )

    # Love plot
    covs = list(p["smd_before"].keys())
    cov_labels = [COVARIATE_LABELS.get(c, c) for c in covs]
    smd_before = [p["smd_before"][c] for c in covs]
    smd_after = [p["smd_after"][c] for c in covs]

    love_fig = go.Figure()
    love_fig.add_trace(
        go.Scatter(
            x=smd_before,
            y=cov_labels,
            mode="markers",
            name="Before matching",
            marker=dict(color=DANGER, size=10, symbol="circle"),
            hovertemplate="%{y}<br>SMD: %{x:.3f}<extra>Before matching</extra>",
        )
    )
    love_fig.add_trace(
        go.Scatter(
            x=smd_after,
            y=cov_labels,
            mode="markers",
            name="After matching",
            marker=dict(color=SUCCESS, size=10, symbol="diamond"),
            hovertemplate="%{y}<br>SMD: %{x:.3f}<extra>After matching</extra>",
        )
    )
    love_fig.add_vline(
        x=0.1, line_dash="dash", line_color=WARNING, annotation_text="0.1 threshold"
    )
    love_fig.add_vline(x=-0.1, line_dash="dash", line_color=WARNING)
    love_fig.add_vline(x=0, line_color=BORDER)
    love_fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Love Plot: Standardised Mean Differences",
        xaxis_title="Standardised Mean Difference",
        yaxis=dict(automargin=True),
        margin=dict(t=50, b=70, l=195, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )

    # ATT bar chart
    stats_fig = go.Figure(
        go.Bar(
            x=["ATT Estimate", "CI Lower", "CI Upper"],
            y=[p["att_point"], p["att_ci_lo"], p["att_ci_hi"]],
            marker_color=[arm_color, BORDER, BORDER],
            text=[
                f"${v:.2f}" for v in [p["att_point"], p["att_ci_lo"], p["att_ci_hi"]]
            ],
            textposition="outside",
            textfont=dict(color=TEXT),
            hovertemplate="%{x}: $%{y:.2f}<extra></extra>",
        )
    )
    stats_fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="ATT with 95% Bootstrap CI",
        yaxis_title="Effect on Spend ($)",
        margin=dict(t=50, b=30)
    )

    return kpis, ps_fig, love_fig, stats_fig


@app.callback(
    Output("method-collapse-tab2", "is_open"),
    Input("method-btn-tab2", "n_clicks"),
    State("method-collapse-tab2", "is_open"),
    prevent_initial_call=True
)
def toggle_method_tab2(n, is_open):
    return not is_open


# ===========================================================================
# CALLBACKS - Tab 3: Bayesian A/B
# ===========================================================================


@app.callback(
    Output("bayes-kpi-cards", "children"),
    Output("bayes-posterior-plot", "figure"),
    Output("rope-result-card", "children"),
    Input("bayes-pair-selector", "value"),
    Input("rope-slider", "value")
)
def update_bayesian(pair_key, rope_val):
    rope_val = rope_val if rope_val is not None else 0
    b = BAYESIAN[pair_key]
    delta = b["delta_samples"]

    hdi_str = f"95% HDI: ${b['hdi_lo']:.2f} - ${b['hdi_hi']:.2f}"
    p_pos = b["p_positive"]
    arm_color = MENS_COLOUR if pair_key.startswith("mens") else WOMENS_COLOUR

    kpis = html.Div(
        [
            kpi_card(
                hdi_str,
                f"Treatment effect: {b['arm_a_label']} vs {b['arm_b_label']}",
                accent=arm_color
            ),
            kpi_card(
                f"{p_pos:.1%}",
                "P(effect > 0)",
                color=SUCCESS if p_pos > 0.9 else WARNING,
                accent=SUCCESS if p_pos > 0.9 else WARNING
            ),
            kpi_card(
                f"${b['mean_a']:.2f}",
                f"Mean spend - {b['arm_a_label']}",
                accent=arm_color
            ),
            kpi_card(
                f"${b['mean_b']:.2f}",
                f"Mean spend - {b['arm_b_label']}",
                accent=CTRL_COLOUR
            ),
        ]
    )

    counts, bin_edges = np.histogram(delta, bins=120, density=True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    posterior_fig = go.Figure()

    rope_mask = (bin_centres >= -rope_val) & (bin_centres <= rope_val)
    if rope_mask.any():
        posterior_fig.add_trace(
            go.Scatter(
                x=np.concatenate(
                    [
                        [bin_centres[rope_mask][0]],
                        bin_centres[rope_mask],
                        [bin_centres[rope_mask][-1]],
                    ]
                ),
                y=np.concatenate([[0], counts[rope_mask], [0]]),
                fill="toself",
                fillcolor="rgba(217,119,6,0.2)",
                line=dict(color="rgba(0,0,0,0)"),
                name=f"ROPE ±${rope_val}",
                showlegend=True,
                hoverinfo="skip",
            )
        )

    fill_rgba = (
        "rgba(34,211,238,0.15)"
        if pair_key.startswith("mens")
        else "rgba(244,114,182,0.15)"
    )
    line_color = MENS_COLOUR if pair_key.startswith("mens") else WOMENS_COLOUR

    posterior_fig.add_trace(
        go.Scatter(
            x=bin_centres,
            y=counts,
            mode="lines",
            fill="tozeroy",
            fillcolor=fill_rgba,
            line=dict(color=line_color, width=2),
            name="Posterior δ",
            hovertemplate="Effect: $%{x:.2f}<extra>Posterior δ</extra>",
        )
    )

    posterior_fig.add_vline(
        x=b["hdi_lo"],
        line_dash="dash",
        line_color=MUTED,
        annotation_text="HDI 2.5%",
        annotation_position="top left",
        annotation_font_color=MUTED
    )
    posterior_fig.add_vline(
        x=b["hdi_hi"],
        line_dash="dash",
        line_color=MUTED,
        annotation_text="HDI 97.5%",
        annotation_position="top right",
        annotation_font_color=MUTED
    )
    posterior_fig.add_vline(
        x=0,
        line_color=DANGER,
        line_width=1.5,
        line_dash="dot"
    )

    posterior_fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"Posterior Distribution - Treatment Effect: {b['arm_a_label']} vs {b['arm_b_label']}",
        xaxis_title="Effect on Spend ($)",
        yaxis_title="Density",
        margin=dict(t=50, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )

    p_outside_rope = float(np.mean((delta > rope_val) | (delta < -rope_val)))
    rope_card = kpi_card(
        f"{p_outside_rope:.1%}",
        f"P(effect outside ROPE ±${rope_val})",
        color=SUCCESS if p_outside_rope > 0.9 else WARNING,
        accent=SUCCESS if p_outside_rope > 0.9 else WARNING,
    )

    return kpis, posterior_fig, rope_card


@app.callback(
    Output("trace-collapse", "is_open"),
    Output("bayes-trace-plot", "figure"),
    Input("trace-btn", "n_clicks"),
    State("trace-collapse", "is_open"),
    State("bayes-pair-selector", "value"),
    prevent_initial_call=True,
)
def toggle_trace(n, is_open, pair_key):
    b = BAYESIAN[pair_key]
    delta_chains = b["delta_chains"]

    fig = go.Figure()
    colors = [MENS_COLOUR, WOMENS_COLOUR]
    for i, chain in enumerate(delta_chains):
        fig.add_trace(
            go.Scatter(
                x=list(range(len(chain))),
                y=chain,
                mode="lines",
                name=f"Chain {i + 1}",
                line=dict(color=colors[i % len(colors)], width=0.8),
                opacity=0.8,
                hovertemplate="Draw %{x}<br>δ: $%{y:.2f}<extra>Chain %{fullData.name}</extra>",
            )
        )

    rhat = b.get("rhat_delta", "N/A")
    bulk_ess = b.get("bulk_ess_delta", "N/A")
    tail_ess = b.get("tail_ess_delta", "N/A")

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="MCMC Trace: δ (treatment effect)",
        xaxis_title="Draw",
        yaxis_title="δ value",
        margin=dict(t=50, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        annotations=[
            dict(
                x=1.02,
                y=1.0,
                xref="paper",
                yref="paper",
                text=f"<b>MCMC Diagnostics</b><br>R-hat: {rhat}<br>Bulk ESS: {bulk_ess}<br>Tail ESS: {tail_ess}",
                showarrow=False,
                font=dict(family="Ubuntu Mono, monospace", size=10, color=MUTED),
                align="left",
                bgcolor=SURFACE,
                bordercolor=BORDER,
                borderwidth=1,
                borderpad=6,
            )
        ],
    )
    return not is_open, fig


@app.callback(
    Output("method-collapse-tab3", "is_open"),
    Input("method-btn-tab3", "n_clicks"),
    State("method-collapse-tab3", "is_open"),
    prevent_initial_call=True,
)
def toggle_method_tab3(n, is_open):
    return not is_open


@app.callback(
    Output("ppc-collapse", "is_open"),
    Output("bayes-ppc-plot", "figure"),
    Input("ppc-btn", "n_clicks"),
    State("ppc-collapse", "is_open"),
    State("bayes-pair-selector", "value"),
    prevent_initial_call=True,
)
def toggle_ppc(n, is_open, pair_key):
    b = BAYESIAN[pair_key]
    obs_a = b.get("observed_amount_a")
    obs_b = b.get("observed_amount_b")
    ppc_a = b.get("ppc_amount_a")
    ppc_b = b.get("ppc_amount_b")

    fig = go.Figure()
    if obs_a is not None and ppc_a is not None:
        fig.add_trace(
            go.Histogram(
                x=obs_a,
                name=f"Observed ({b['arm_a_label']})",
                histnorm="probability density",
                marker_color=MENS_COLOUR,
                opacity=0.45,
                nbinsx=60,
            )
        )
        fig.add_trace(
            go.Histogram(
                x=ppc_a,
                name=f"Posterior predictive ({b['arm_a_label']})",
                histnorm="probability density",
                marker_color=ACCENT,
                opacity=0.45,
                nbinsx=60,
            )
        )
    if obs_b is not None and ppc_b is not None:
        fig.add_trace(
            go.Histogram(
                x=obs_b,
                name=f"Observed ({b['arm_b_label']})",
                histnorm="probability density",
                marker_color=WOMENS_COLOUR,
                opacity=0.45,
                nbinsx=60,
                visible="legendonly",
            )
        )
        fig.add_trace(
            go.Histogram(
                x=ppc_b,
                name=f"Posterior predictive ({b['arm_b_label']})",
                histnorm="probability density",
                marker_color=CTRL_COLOUR,
                opacity=0.45,
                nbinsx=60,
                visible="legendonly",
            )
        )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        barmode="overlay",
        title="Posterior predictive check: conditional spend amount (converters only)",
        xaxis_title="Spend ($)",
        yaxis_title="Density",
        margin=dict(t=60, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
    )
    all_obs = [x for x in (obs_a, obs_b) if x is not None and len(x) > 0]
    if all_obs:
        p99 = max(float(np.percentile(o, 99)) for o in all_obs)
        fig.update_xaxes(range=[0, p99])
    return not is_open, fig


@app.callback(
    Output("diag-collapse", "is_open"),
    Output("bayes-diagnostics-table", "children"),
    Input("diag-btn", "n_clicks"),
    State("diag-collapse", "is_open"),
    State("bayes-pair-selector", "value"),
    prevent_initial_call=True,
)
def toggle_diagnostics(n, is_open, pair_key):
    b = BAYESIAN[pair_key]
    diag_table = b.get("diagnostics_table", [])

    if not diag_table:
        return not is_open, "No diagnostics available"

    header = html.Tr(
        [
            html.Th(
                "Parameter",
                style={"fontFamily": "Ubuntu Mono, monospace", "fontSize": "0.75rem"},
            ),
            html.Th(
                "R-hat",
                style={"fontFamily": "Ubuntu Mono, monospace", "fontSize": "0.75rem"},
            ),
            html.Th(
                "Bulk ESS",
                style={"fontFamily": "Ubuntu Mono, monospace", "fontSize": "0.75rem"},
            ),
            html.Th(
                "Tail ESS",
                style={"fontFamily": "Ubuntu Mono, monospace", "fontSize": "0.75rem"},
            ),
        ]
    )

    rows = []
    for row in diag_table:
        rhat_color = (
            SUCCESS
            if row["r_hat"] < 1.05
            else WARNING
            if row["r_hat"] < 1.1
            else DANGER
        )
        rows.append(
            html.Tr(
                [
                    html.Td(
                        row["parameter"],
                        style={
                            "fontFamily": "Ubuntu Mono, monospace",
                            "fontSize": "0.8rem",
                        },
                    ),
                    html.Td(
                        f"{row['r_hat']:.3f}",
                        style={
                            "fontFamily": "Ubuntu Mono, monospace",
                            "fontSize": "0.8rem",
                            "color": rhat_color,
                        },
                    ),
                    html.Td(
                        f"{row['ess_bulk']:.0f}",
                        style={
                            "fontFamily": "Ubuntu Mono, monospace",
                            "fontSize": "0.8rem",
                        },
                    ),
                    html.Td(
                        f"{row['ess_tail']:.0f}",
                        style={
                            "fontFamily": "Ubuntu Mono, monospace",
                            "fontSize": "0.8rem",
                        },
                    ),
                ]
            )
        )

    table = dbc.Table(
        [html.Thead(header), html.Tbody(rows)],
        bordered=False,
        size="sm",
        style={"marginBottom": 0},
    )

    return not is_open, table


# ===========================================================================
# CALLBACKS - Tab 4: Uplift / HTE
# ===========================================================================


@app.callback(
    Output("uplift-kpi-cards", "children"),
    Output("uplift-cate-hist", "figure"),
    Output("uplift-feat-imp", "figure"),
    Output("uplift-decile-chart", "figure"),
    Output("uplift-qini", "figure"),
    Output("uplift-segment-compare", "figure"),
    Input("uplift-arm-selector", "value"),
    Input("uplift-model-selector", "value"),
)
def update_uplift(arm, model):
    u = UPLIFT[arm]
    arm_label = "Men's Email" if arm == "mens" else "Women's Email"
    cate = u["cate_t"] if model == "t" else u["cate_s"]
    model_label = "T-Learner" if model == "t" else "S-Learner"
    color = MENS_COLOUR if arm == "mens" else WOMENS_COLOUR

    kpis = html.Div(
        [
            kpi_card(
                f"${np.mean(cate):.2f}",
                f"Avg CATE ({model_label})",
                f"{arm_label} vs Control",
                np.mean(cate) > 0,
                color=color,
                accent=color,
            ),
            kpi_card(
                f"${np.percentile(cate, 90):.2f}",
                "90th percentile CATE",
                "High-responder threshold",
                accent=ACCENT,
            ),
            kpi_card(f"{np.mean(cate > 0):.1%}", "% customers with positive uplift"),
        ]
    )

    p1, p99 = np.percentile(cate, 1), np.percentile(cate, 99)
    cate_clipped = cate[(cate >= p1) & (cate <= p99)]
    pct_shown = len(cate_clipped) / len(cate) * 100

    hist_fig = go.Figure(
        go.Histogram(
            x=cate_clipped,
            nbinsx=60,
            marker_color=color,
            opacity=0.8,
            name="CATE",
        )
    )
    hist_fig.add_vline(x=0, line_color=DANGER, line_dash="dash")
    hist_fig.add_vline(
        x=np.mean(cate), line_color=WARNING, line_dash="dot", annotation_text="Mean"
    )
    hist_fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"CATE Distribution - {model_label} ({arm_label})",
        xaxis_title=f"Individual Uplift ($):  showing p1-p99 ({pct_shown:.0f}% of customers)",
        yaxis_title="Count",
        margin=dict(t=50, b=30),
    )

    feat_imp = dict(sorted(u["feat_imp"].items(), key=lambda x: x[1]))
    feat_labels = [COVARIATE_LABELS.get(k, k) for k in feat_imp.keys()]
    fi_fig = go.Figure(
        go.Bar(
            x=list(feat_imp.values()),
            y=feat_labels,
            orientation="h",
            marker_color=color,
            hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>",
        )
    )
    fi_fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Heterogeneity importance (|treated − control| model diff)",
        xaxis_title="Relative importance (normalised)",
        margin=dict(t=50, b=30, l=130),
    )

    decile_key = "decile_lift_s" if model == "s" else "decile_lift"
    qini_x_key = "qini_x_s" if model == "s" else "qini_x"
    qini_y_key = "qini_y_s" if model == "s" else "qini_y"
    dec_df = pd.DataFrame(u.get(decile_key, u["decile_lift"]))
    qini_xd = u.get(qini_x_key, u["qini_x"])
    qini_yd = u.get(qini_y_key, u["qini_y"])
    overall_ate = dec_df["lift"].mean()

    decile_fig = go.Figure()
    decile_fig.add_trace(
        go.Bar(
            x=dec_df["decile"],
            y=dec_df["lift"],
            marker_color=[color if v > 0 else DANGER for v in dec_df["lift"]],
            opacity=0.6,
            showlegend=False,
            hovertemplate="Decile %{x}<br>Actual lift: $%{y:.2f}<extra></extra>",
        )
    )
    decile_fig.add_trace(
        go.Scatter(
            x=dec_df["decile"],
            y=dec_df["lift"],
            mode="markers+text",
            marker=dict(
                color=[color if v > 0 else DANGER for v in dec_df["lift"]],
                size=9,
                line=dict(color=BG, width=1),
            ),
            text=[f"${v:.2f}" for v in dec_df["lift"]],
            textposition="top center",
            textfont=dict(size=10),
            showlegend=False,
            hovertemplate="Decile %{x}<br>Actual lift: $%{y:.2f}<extra></extra>",
        )
    )
    decile_fig.add_hline(
        y=overall_ate,
        line_dash="dash",
        line_color=WARNING,
        line_width=1.5,
        annotation_text=f"Avg lift ${overall_ate:.2f}",
        annotation_position="right",
        annotation_font_color=WARNING
    )
    decile_fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"Actual Spend Lift by {model_label} Uplift Decile",
        xaxis=dict(
            title="Decile (1 = highest predicted uplift)",
            tickmode="linear",
            tick0=1,
            dtick=1
        ),
        yaxis_title="Actual Spend Lift ($)",
        margin=dict(t=50, b=30)
    )

    qini_fill = (
        f"rgba({','.join(str(int(c * 255)) for c in px.colors.hex_to_rgb(color))},0.15)"
        if color.startswith("#")
        else "rgba(34,211,238,0.15)"
    )
    qini_fig = go.Figure()
    qini_fig.add_trace(
        go.Scatter(
            x=qini_xd,
            y=qini_yd,
            mode="lines",
            name=f"{model_label} Qini",
            line=dict(color=color, width=2),
            fill="tozeroy",
            fillcolor=qini_fill,
            hovertemplate="Top %{x:.0%} targeted<br>Cumulative incremental spend: $%{y:,.0f}<extra>%{fullData.name}</extra>",
        )
    )
    qini_fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 0],
            mode="lines",
            name="Random",
            line=dict(color=BORDER, dash="dash"),
            hoverinfo="skip",
        )
    )
    qini_auc_key = "qini_auc_s" if model == "s" else "qini_auc_t"
    qini_auc = u.get(qini_auc_key, 0.0)
    qini_fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"Qini Curve — {model_label} (AUC = ${qini_auc:,.0f})",
        xaxis_title="Fraction of population targeted",
        yaxis_title="Cumulative incremental spend ($)",
        margin=dict(t=50, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )

    seg_fig = go.Figure()
    for a, lbl, col in [
        ("mens", "Men's Email", MENS_COLOUR),
        ("womens", "Women's Email", WOMENS_COLOUR),
    ]:
        avg_t = UPLIFT[a]["avg_cate_t"]
        avg_s = UPLIFT[a]["avg_cate_s"]
        seg_fig.add_trace(
            go.Bar(
                name=lbl,
                x=["T-Learner", "S-Learner"],
                y=[avg_t, avg_s],
                marker_color=col,
                opacity=0.85,
                hovertemplate="%{x}<br>Avg CATE: $%{y:.2f}<extra>%{fullData.name}</extra>",
            )
        )
    seg_fig.update_layout(
        barmode="group",
        template=PLOTLY_TEMPLATE,
        title="Average CATE: Men's vs Women's Campaign",
        yaxis_title="Avg CATE ($)",
        margin=dict(t=50, b=70),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5
        ),
    )

    return kpis, hist_fig, fi_fig, decile_fig, qini_fig, seg_fig


@app.callback(
    Output("method-collapse-tab4", "is_open"),
    Input("method-btn-tab4", "n_clicks"),
    State("method-collapse-tab4", "is_open"),
    prevent_initial_call=True,
)
def toggle_method_tab4(n, is_open):
    return not is_open


# ===========================================================================
# CALLBACKS - Tab 5: Multi-Arm OLS
# ===========================================================================


@app.callback(
    Output("ols-coef-plot", "figure"),
    Output("ols-marginal-table", "children"),
    Output("ols-heatmap-mens", "figure"),
    Output("ols-heatmap-womens", "figure"),
    Input("main-tabs", "active_tab")
)
def update_ols(tab):
    if tab != "tab-5":
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    coef_df = OLS["coef_df"].copy()
    subgroup_df = OLS["subgroup_df"].copy()

    keep_terms = [t for t in coef_df["term"] if t not in ["Intercept"]]
    plot_df = coef_df[coef_df["term"].isin(keep_terms)].copy()
    plot_df = plot_df.sort_values("coef")

    colors = [
        SUCCESS if v > 0 and p < 0.05 else DANGER if v < 0 and p < 0.05 else MUTED
        for v, p in zip(plot_df["coef"], plot_df["pvalue"])
    ]

    coef_fig = go.Figure()
    coef_fig.add_trace(
        go.Scatter(
            x=plot_df["coef"],
            y=plot_df["term"],
            mode="markers",
            error_x=dict(
                type="data",
                symmetric=False,
                array=plot_df["ci_hi"] - plot_df["coef"],
                arrayminus=plot_df["coef"] - plot_df["ci_lo"],
                color=MUTED
            ),
            marker=dict(color=colors, size=10),
            name="Coefficient",
            customdata=plot_df[["ci_lo", "ci_hi", "pvalue"]].values,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Coef: $%{x:.2f}<br>"
                "95% CI: $%{customdata[0]:.2f} – $%{customdata[1]:.2f}<br>"
                "p-value: %{customdata[2]:.3f}"
                "<extra></extra>"
            ),
        )
    )
    coef_fig.add_vline(x=0, line_color=DANGER, line_dash="dash")
    coef_fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"OLS Coefficients (n={OLS['n_obs']:,}, R²={OLS['r_squared']:.4f})",
        xaxis_title="Effect on Spend ($)",
        margin=dict(t=50, b=30, l=260)
    )

    # Weight the zip-level marginal effects by actual population shares when
    # collapsing to (newbie, channel). The raw dataset has wildly uneven zip
    # distributions; an unweighted mean would over-represent rare zip cells.
    zip_spell = {"Urban": "Urban", "Surburban": "Suburban", "Rural": "Rural"}
    _cell_counts = (
        DF.assign(
            _newbie=DF["newbie"].map({0: "Existing", 1: "New"}),
            _zip=DF["zip_code"].map(zip_spell),
        )
        .groupby(["_newbie", "channel", "_zip"])
        .size()
        .rename("n")
        .reset_index()
        .rename(columns={"_newbie": "newbie", "_zip": "zip_code"})
    )
    _weighted = subgroup_df.merge(
        _cell_counts, on=["newbie", "channel", "zip_code"], how="left"
    )
    _weighted["n"] = _weighted["n"].fillna(0)

    def _wavg(g):
        w = g["n"].values
        if w.sum() == 0:
            return pd.Series(
                {"me_mens": g["me_mens"].mean(), "me_womens": g["me_womens"].mean()}
            )
        return pd.Series(
            {
                "me_mens": float(np.average(g["me_mens"], weights=w)),
                "me_womens": float(np.average(g["me_womens"], weights=w)),
            }
        )

    weighted_sub = (
        _weighted.groupby(["newbie", "channel"], group_keys=False)
        .apply(_wavg)
        .reset_index()
    )

    disp_df = weighted_sub.rename(
        columns={
            "newbie": "Customer type",
            "channel": "Channel",
            "me_mens": "Men's Email ($)",
            "me_womens": "Women's Email ($)"
        }
    )
    disp_df["Men's Email ($)"] = disp_df["Men's Email ($)"].round(2)
    disp_df["Women's Email ($)"] = disp_df["Women's Email ($)"].round(2)

    table = dash_table.DataTable(
        data=disp_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in disp_df.columns],
        style_table={"overflowX": "auto"},
        style_cell=TABLE_CELL,
        style_header=TABLE_HEADER,
        style_data_conditional=[
            {
                "if": {
                    "filter_query": "{Men's Email ($)} > 0",
                    "column_id": "Men's Email ($)",
                },
                "color": SUCCESS
            },
            {
                "if": {
                    "filter_query": "{Men's Email ($)} < 0",
                    "column_id": "Men's Email ($)",
                },
                "color": DANGER
            },
            {
                "if": {
                    "filter_query": "{Women's Email ($)} > 0",
                    "column_id": "Women's Email ($)",
                },
                "color": SUCCESS
            },
            {
                "if": {
                    "filter_query": "{Women's Email ($)} < 0",
                    "column_id": "Women's Email ($)",
                },
                "color": DANGER
            },
        ],
        page_size=12
    )

    all_vals = pd.concat([weighted_sub["me_mens"], weighted_sub["me_womens"]])
    zmax = max(abs(all_vals.min()), abs(all_vals.max()))
    zmin = -zmax

    def make_heatmap(arm_col):
        heat_pivot = weighted_sub.pivot(
            index="newbie", columns="channel", values=arm_col
        )
        return go.Figure(
            go.Heatmap(
                z=heat_pivot.values,
                x=heat_pivot.columns.tolist(),
                y=heat_pivot.index.tolist(),
                colorscale="RdYlGn",
                zmin=zmin,
                zmax=zmax,
                zmid=0,
                text=[[f"${v:.2f}" for v in row] for row in heat_pivot.values],
                texttemplate="%{text}",
                hovertemplate="%{y} / %{x}<br>Marginal effect: $%{z:.2f}<extra></extra>",
                colorbar=dict(
                    title=dict(text="$ lift", font=dict(color=MUTED)),
                    tickfont=dict(color=MUTED)
                ),
            )
        )

    mens_heat = make_heatmap("me_mens")
    mens_heat.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Men's Email: Marginal Effect ($)",
        xaxis_title="Channel",
        yaxis_title="Customer type",
        margin=dict(t=50, b=30)
    )

    womens_heat = make_heatmap("me_womens")
    womens_heat.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Women's Email: Marginal Effect ($)",
        xaxis_title="Channel",
        yaxis_title="Customer type",
        margin=dict(t=50, b=30)
    )

    return coef_fig, table, mens_heat, womens_heat


@app.callback(
    Output("method-collapse-tab5", "is_open"),
    Input("method-btn-tab5", "n_clicks"),
    State("method-collapse-tab5", "is_open"),
    prevent_initial_call=True
)
def toggle_method_tab5(n, is_open):
    return not is_open


# ===========================================================================
# CALLBACKS - Tab 6: Method Comparison
# ===========================================================================


def _build_comparison_df():
    rows = []
    for arm in ["mens", "womens"]:
        arm_label = "Men's Email" if arm == "mens" else "Women's Email"
        p = PSM[arm]
        rows.append(
            {
                "Method": "PSM (ATT)",
                "Arm": arm_label,
                "Estimate ($)": round(p["att_point"], 2),
                "CI Lower ($)": round(p["att_ci_lo"], 2),
                "CI Upper ($)": round(p["att_ci_hi"], 2)
            }
        )

    pair_map = {"mens": "mens_vs_control", "womens": "womens_vs_control"}
    for arm in ["mens", "womens"]:
        arm_label = "Men's Email" if arm == "mens" else "Women's Email"
        b = BAYESIAN[pair_map[arm]]
        rows.append(
            {
                "Method": "Bayesian A/B (posterior mean)",
                "Arm": arm_label,
                "Estimate ($)": round(float(np.mean(b["delta_samples"])), 2),
                "CI Lower ($)": round(b["hdi_lo"], 2),
                "CI Upper ($)": round(b["hdi_hi"], 2)
            }
        )

    for arm in ["mens", "womens"]:
        arm_label = "Men's Email" if arm == "mens" else "Women's Email"
        u = UPLIFT[arm]
        rows.append(
            {
                "Method": "T-Learner (avg CATE)",
                "Arm": arm_label,
                "Estimate ($)": round(u["avg_cate_t"], 2),
                "CI Lower ($)": None,
                "CI Upper ($)": None
            }
        )
        rows.append(
            {
                "Method": "S-Learner (avg CATE)",
                "Arm": arm_label,
                "Estimate ($)": round(u["avg_cate_s"], 2),
                "CI Lower ($)": None,
                "CI Upper ($)": None
            }
        )

    # OLS: report the *population-weighted ATE* (average marginal effect over
    # the sample's actual covariate distribution) with its HC3 delta-method CI.
    # The raw `mens_email` / `womens_email` coefficients are only the effect
    # for the reference subgroup (Existing + Phone + Urban) and are not
    # directly comparable to PSM's ATT or the Bayesian delta.
    for arm, ate_key, lo_key, hi_key, arm_label in [
        ("mens", "ate_mens", "ate_mens_lo", "ate_mens_hi", "Men's Email"),
        ("womens", "ate_womens", "ate_womens_lo", "ate_womens_hi", "Women's Email"),
    ]:
        rows.append(
            {
                "Method": "OLS (avg marginal effect, HC3)",
                "Arm": arm_label,
                "Estimate ($)": round(OLS.get(ate_key, 0.0), 2),
                "CI Lower ($)": round(OLS.get(lo_key, 0.0), 2),
                "CI Upper ($)": round(OLS.get(hi_key, 0.0), 2),
            }
        )

    return pd.DataFrame(rows)


@app.callback(
    Output("comparison-table", "children"),
    Output("forest-plot-mens", "figure"),
    Output("forest-plot-womens", "figure"),
    Output("key-takeaway-card", "children"),
    Input("main-tabs", "active_tab")
)
def update_comparison(tab):
    if tab != "tab-6":
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    comp_df = _build_comparison_df()

    table = dash_table.DataTable(
        data=comp_df.fillna("-").to_dict("records"),
        columns=[{"name": c, "id": c} for c in comp_df.columns],
        export_format="csv",
        export_headers="display",
        style_table={"overflowX": "auto"},
        style_cell=TABLE_CELL,
        style_header=TABLE_HEADER,
        style_data_conditional=[
            {
                "if": {
                    "filter_query": "{Estimate ($)} > 0",
                    "column_id": "Estimate ($)"
                },
                "color": SUCCESS
            },
            {
                "if": {
                    "filter_query": "{Estimate ($)} < 0",
                    "column_id": "Estimate ($)"
                },
                "color": DANGER
            },
        ],
    )

    def forest_plot(arm_label, color):
        sub = comp_df[comp_df["Arm"] == arm_label].copy()
        fig = go.Figure()
        for i, row in sub.iterrows():
            has_ci = pd.notna(row["CI Lower ($)"]) and pd.notna(row["CI Upper ($)"])
            fig.add_trace(
                go.Scatter(
                    x=[row["Estimate ($)"]],
                    y=[row["Method"]],
                    mode="markers",
                    marker=dict(
                        color=color,
                        size=12,
                        symbol="diamond",
                        line=dict(color=BG, width=1)
                    ),
                    name=row["Method"],
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{row['Method']}</b><br>"
                        f"Estimate: ${row['Estimate ($)']:.2f}<br>"
                        + (
                            f"95% CI: ${row['CI Lower ($)']:.2f} – ${row['CI Upper ($)']:.2f}"
                            if has_ci
                            else "No CI available"
                        )
                        + "<extra></extra>"
                    ),
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=[row["CI Upper ($)"] - row["Estimate ($)"]]
                        if has_ci
                        else [0],
                        arrayminus=[row["Estimate ($)"] - row["CI Lower ($)"]]
                        if has_ci
                        else [0],
                        color=MUTED
                    )
                    if has_ci
                    else None,
                )
            )
        fig.add_vline(x=0, line_color=DANGER, line_dash="dash")
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            title=f"Forest Plot - {arm_label}",
            xaxis_title="Effect on Spend ($)",
            margin=dict(t=50, b=30, l=210),
            height=350,
        )
        return fig

    mens_fig = forest_plot("Men's Email", MENS_COLOUR)
    womens_fig = forest_plot("Women's Email", WOMENS_COLOUR)

    mens_estimates = comp_df[comp_df["Arm"] == "Men's Email"]["Estimate ($)"].values
    womens_estimates = comp_df[comp_df["Arm"] == "Women's Email"]["Estimate ($)"].values
    mens_valid = [v for v in mens_estimates if pd.notna(v)]
    womens_valid = [v for v in womens_estimates if pd.notna(v)]
    mens_min, mens_max = min(mens_valid), max(mens_valid)
    womens_min, womens_max = min(womens_valid), max(womens_valid)

    # Robust verdict: don't flip on a single near-zero estimate. Treat
    # |effect| < $0.10 as "noise zone" — smaller than any plausible action
    # threshold in this dataset. "Agree" requires (a) no method in the noise
    # zone is on the opposite side, AND (b) all material estimates share sign.
    NOISE_EPS = 0.10

    def _verdict(estimates):
        material = [v for v in estimates if abs(v) >= NOISE_EPS]
        near_zero = [v for v in estimates if abs(v) < NOISE_EPS]
        if not material:
            return "All methods indistinguishable from zero."
        pos = sum(1 for v in material if v > 0)
        neg = sum(1 for v in material if v < 0)
        if pos > 0 and neg == 0:
            tail = (
                f" ({len(near_zero)} method[s] near zero.)" if near_zero else ""
            )
            return "All methods point to a positive effect." + tail
        if neg > 0 and pos == 0:
            tail = (
                f" ({len(near_zero)} method[s] near zero.)" if near_zero else ""
            )
            return "All methods point to a negative effect." + tail
        return (
            f"Methods disagree on direction ({pos} positive, {neg} negative): "
            "inspect assumptions carefully."
        )

    mens_verdict = _verdict(mens_valid)
    womens_verdict = _verdict(womens_valid)

    takeaway = dbc.Card(
        [
            dbc.CardHeader("Key Takeaway"),
            dbc.CardBody(
                [
                    html.P(
                        [
                            html.Strong(
                                "Men's campaign: ", style={"color": MENS_COLOUR}
                            ),
                            f"Estimated spend uplift ranges from ${mens_min:.2f} to ${mens_max:.2f} across "
                            f"{len(mens_valid)} methods. {mens_verdict}"
                        ]
                    ),
                    html.P(
                        [
                            html.Strong(
                                "Women's campaign: ", style={"color": WOMENS_COLOUR}
                            ),
                            f"Estimated spend uplift ranges from ${womens_min:.2f} to ${womens_max:.2f} across "
                            f"{len(womens_valid)} methods. {womens_verdict}"
                        ]
                    ),
                    html.P(
                        "Method agreement strengthens causal credibility. When estimates diverge, "
                        "the gap reflects differing assumptions: PSM relies on covariate overlap, "
                        "Bayesian A/B on distribution, and uplift on out-of-sample generalisation.",
                        className="text-muted small mb-0"
                    ),
                ]
            ),
        ],
        style={
            **CARD_STYLE,
            "borderLeft": f"3px solid {SUCCESS if ('point to' in mens_verdict and 'point to' in womens_verdict) else WARNING}"
        },
    )

    return table, mens_fig, womens_fig, takeaway


# ===========================================================================
# CALLBACKS - Tab 1 methodology collapse
# ===========================================================================


@app.callback(
    Output("method-collapse-tab1", "is_open"),
    Input("method-btn-tab1", "n_clicks"),
    State("method-collapse-tab1", "is_open"),
    prevent_initial_call=True
)
def toggle_method_tab1(n, is_open):
    return not is_open


# ===========================================================================
# Run
# ===========================================================================

if __name__ == "__main__":
    app.run(debug=True, port=8050)
