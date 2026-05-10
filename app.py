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
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, dash_table, Input, Output, State, ctx
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
# Shared active/selected cell state styles — subdued teal instead of the
# default bright blue that clashes with the dark theme.
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

            /* Segmented control (modern radio group) */
            .segmented-control {
                display: inline-flex;
                background: #0D3535;
                border: 1px solid #1A4040;
                border-radius: 6px;
                padding: 3px;
                gap: 2px;
            }
            .segmented-control .form-check {
                margin: 0 !important;
                padding-left: 0 !important;
                display: flex;
            }
            .segmented-control .form-check-input {
                position: absolute;
                opacity: 0;
                pointer-events: none;
                margin: 0;
            }
            .segmented-control .form-check-label {
                cursor: pointer;
                padding: 0.45rem 1.1rem;
                border-radius: 4px;
                color: #6B9090;
                font-family: "Ubuntu Mono", monospace;
                font-size: 0.72rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                transition: background-color 0.15s ease, color 0.15s ease, border-color 0.15s ease;
                user-select: none;
                border: 1px solid transparent;
                line-height: 1.2;
                white-space: nowrap;
            }
            .segmented-control .form-check-label:hover {
                color: #E8F1F1;
                background: rgba(255,255,255,0.04);
            }
            .segmented-control .form-check-input:checked + .form-check-label {
                color: #FF5F03;
                background: rgba(255,95,3,0.1);
                border-color: rgba(255,95,3,0.35);
            }
            .segmented-control .form-check-input:focus-visible + .form-check-label {
                outline: 2px solid rgba(255,95,3,0.5);
                outline-offset: 1px;
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

            /* DataTable cell selection — override Dash's default bright blue */
            .dash-table-container td.focused,
            .dash-table-container td.cell--selected {
                background-color: #0D3535 !important;
                box-shadow: none !important;
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

            /* Overview: page subtitle strip */
            .overview-context {
                display: flex;
                align-items: center;
                flex-wrap: wrap;
                gap: 0.6rem 1.4rem;
                padding: 0.55rem 0 1rem 0;
                margin-bottom: 0.75rem;
                border-bottom: 1px solid #1A4040;
                font-family: "Ubuntu Mono", monospace;
                font-size: 0.74rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #6B9090;
            }
            .overview-context strong {
                color: #E2F0EF;
                font-weight: 700;
                letter-spacing: 0.04em;
            }
            .overview-context .sep {
                width: 4px;
                height: 4px;
                border-radius: 50%;
                background: #1A4040;
                display: inline-block;
            }

            /* Overview: segment summary cards */
            .segment-card {
                background-color: #072C2C;
                border: 1px solid #1A4040;
                border-radius: 4px;
                overflow: hidden;
                transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
                height: 100%;
            }
            .segment-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 26px rgba(0, 0, 0, 0.38);
                border-color: #254848;
            }
            .segment-card-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 0.75rem 1.05rem;
                border-bottom: 1px solid #1A4040;
                background-color: #051F1F;
            }
            .segment-card-title {
                display: flex;
                align-items: center;
                font-family: "Oswald", sans-serif;
                font-weight: 500;
                font-size: 0.82rem;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: #E2F0EF;
            }
            .segment-dot {
                width: 9px;
                height: 9px;
                border-radius: 50%;
                display: inline-block;
                margin-right: 0.6rem;
                box-shadow: 0 0 0 3px rgba(255,255,255,0.04);
            }
            .segment-card-count {
                font-family: "Ubuntu Mono", monospace;
                font-size: 0.72rem;
                color: #6B9090;
                letter-spacing: 0.06em;
            }
            .segment-card-count strong {
                color: #C8E0DF;
                font-weight: 700;
                letter-spacing: 0.02em;
            }
            .segment-metric-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 1px;
                background-color: #1A4040;
            }
            .segment-metric {
                background-color: #072C2C;
                padding: 1.05rem 1.1rem 1.1rem 1.1rem;
                min-height: 118px;
                display: flex;
                flex-direction: column;
                justify-content: flex-start;
            }
            .segment-metric-label {
                font-family: "Ubuntu Mono", monospace;
                font-size: 0.68rem;
                color: #6B9090;
                letter-spacing: 0.1em;
                text-transform: uppercase;
                margin-bottom: 0.35rem;
            }
            .segment-metric-value {
                font-family: "Oswald", sans-serif;
                font-size: 1.95rem;
                font-weight: 500;
                line-height: 1.05;
                letter-spacing: 0.01em;
            }
            .segment-metric-delta {
                font-family: "Ubuntu", sans-serif;
                font-size: 0.76rem;
                margin-top: 0.55rem;
                color: #6B9090;
                display: flex;
                align-items: center;
                flex-wrap: wrap;
                gap: 0.35rem;
            }
            .segment-metric-delta .delta-num {
                font-family: "Ubuntu Mono", monospace;
                font-weight: 700;
                letter-spacing: 0.02em;
            }
            .segment-sig-badge {
                font-family: "Ubuntu Mono", monospace;
                font-size: 0.58rem;
                letter-spacing: 0.1em;
                text-transform: uppercase;
                padding: 1px 6px;
                border-radius: 2px;
                border: 1px solid currentColor;
                line-height: 1.3;
            }
            .segment-metric-baseline {
                font-family: "Ubuntu Mono", monospace;
                font-size: 0.68rem;
                color: #6B9090;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-top: 0.55rem;
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
    """Metric card with optional delta row, ⓘ tooltip, and right-side pct-change badge."""
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


def segment_overview_card(
    name,
    color,
    n,
    revenue_per,
    conversion_rate,
    rev_lift=None,
    rev_pct=None,
    rev_sig=None,
    conv_lift_pp=None,
    conv_pct=None,
    is_control=False,
):
    """Consolidated per-segment overview card: N in header, revenue & conversion as metrics."""

    def _delta_row(lift_text, pct_change, positive, sig=None):
        if pct_change is None:
            return None
        arrow = "▲" if positive else "▼"
        color_ok = SUCCESS if positive else DANGER
        children = [
            html.Span(arrow, style={"color": color_ok, "fontSize": "0.7rem"}),
            html.Span(lift_text, className="delta-num", style={"color": TEXT}),
            html.Span(f"({pct_change:+.1f}%)", style={"color": MUTED}),
        ]
        if sig is not None:
            badge_color = SUCCESS if sig else MUTED
            badge_text = "SIG" if sig else "N.S."
            children.append(
                html.Span(
                    badge_text,
                    className="segment-sig-badge",
                    style={"color": badge_color},
                )
            )
        return html.Div(children, className="segment-metric-delta")

    if is_control:
        rev_delta = html.Div("Baseline", className="segment-metric-baseline")
        conv_delta = html.Div("Baseline", className="segment-metric-baseline")
    else:
        rev_delta = _delta_row(
            f"+${rev_lift:.2f}" if rev_lift >= 0 else f"-${abs(rev_lift):.2f}",
            rev_pct,
            rev_lift >= 0,
        ) or html.Div()
        conv_delta = _delta_row(
            f"{conv_lift_pp:+.2f}pp",
            conv_pct,
            conv_lift_pp >= 0,
        ) or html.Div()

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                className="segment-dot",
                                style={"backgroundColor": color},
                            ),
                            html.Span(name),
                        ],
                        className="segment-card-title",
                    ),
                    html.Div(
                        [html.Strong(f"{n:,}"), " users"],
                        className="segment-card-count",
                    ),
                ],
                className="segment-card-header",
                style={"borderTop": f"2px solid {color}"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Revenue / recipient", className="segment-metric-label"),
                            html.Div(
                                f"${revenue_per:.2f}",
                                className="segment-metric-value",
                                style={"color": color},
                            ),
                            rev_delta,
                        ],
                        className="segment-metric",
                    ),
                    html.Div(
                        [
                            html.Div("Conversion rate", className="segment-metric-label"),
                            html.Div(
                                f"{conversion_rate:.2f}%",
                                className="segment-metric-value",
                                style={"color": color},
                            ),
                            conv_delta,
                        ],
                        className="segment-metric",
                    ),
                ],
                className="segment-metric-grid",
            ),
        ],
        className="segment-card",
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
        # Simple difference-of-means bootstrap — not the causal bootstrap used in PSM.
        # Valid here because the randomised design makes the raw difference unbiased.
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
        headline = "Women's campaign drove significant lift. Men's result is inconclusive at 95% confidence."
        headline_color = SUCCESS
    elif mens_sig and wom_sig:
        headline = "Both campaigns drove statistically significant incremental revenue."
        headline_color = SUCCESS
    else:
        headline = "Results warrant further review across methods — see Tab 6 for cross-method comparison."
        headline_color = WARNING

    rev_pct_mens = (lift_mens / avg_control * 100) if avg_control else None
    rev_pct_womens = (lift_womens / avg_control * 100) if avg_control else None
    conv_pct_mens = ((conv_mens - conv_control) / conv_control * 100) if conv_control else None
    conv_pct_womens = ((conv_womens - conv_control) / conv_control * 100) if conv_control else None

    def _hl_col(label, color, sig, lift, lo, hi, proj, proj_lo, proj_hi):
        badge_text, badge_color = ("SIG", SUCCESS) if sig else ("n.s.", DANGER)
        badge_id = f"hl-sig-{label.lower().replace(' ', '-').replace(chr(39), '')}"
        tooltip_text = (
            "95% CI excludes zero: the lift is statistically distinguishable from no effect "
            "(2,000-resample percentile bootstrap, α = 0.05)."
            if sig else
            "95% CI includes zero: the lift cannot be distinguished from chance at the 5% level."
        )
        return dbc.Col(
            [
            html.Div(
                [
                    html.Div(label, style={"fontSize": "0.7rem", "fontFamily": "Ubuntu Mono, monospace",
                                           "textTransform": "uppercase", "letterSpacing": "0.1em",
                                           "color": MUTED, "marginBottom": "0.5rem"}),
                    html.Div(
                        [
                            html.Span(f"${lift:.2f}",
                                      style={"fontSize": "1.9rem", "fontFamily": "Oswald, sans-serif",
                                             "fontWeight": "500",
                                             "color": color if sig else MUTED, "lineHeight": "1"}),
                            html.Span(" / recipient",
                                      style={"fontSize": "0.73rem", "color": MUTED,
                                             "marginLeft": "5px", "verticalAlign": "middle"}),
                        ],
                        style={"marginBottom": "0.3rem"},
                    ),
                    html.Div(
                        [
                            html.Span(f"95% CI  ${lo:.2f}–${hi:.2f}",
                                      style={"fontSize": "0.7rem", "fontFamily": "Ubuntu Mono, monospace",
                                             "color": MUTED}),
                            html.Span(badge_text,
                                      id=badge_id,
                                      style={"fontSize": "0.6rem", "fontFamily": "Ubuntu Mono, monospace",
                                             "textTransform": "uppercase", "letterSpacing": "0.12em",
                                             "color": badge_color, "border": f"1px solid {badge_color}",
                                             "borderRadius": "2px", "padding": "1px 5px",
                                             "cursor": "help"}),
                        ],
                        style={"display": "flex", "alignItems": "center", "gap": "8px",
                               "marginBottom": "0.8rem"},
                    ),
                    html.Hr(style={"borderColor": BORDER, "margin": "0.7rem 0"}),
                    html.Div("Projected total lift",
                             style={"fontSize": "0.68rem", "fontFamily": "Ubuntu Mono, monospace",
                                    "textTransform": "uppercase", "letterSpacing": "0.08em",
                                    "color": MUTED, "marginBottom": "0.3rem"}),
                    html.Div(f"${proj:,.0f}",
                             style={"fontSize": "1.35rem", "fontFamily": "Oswald, sans-serif",
                                    "fontWeight": "500", "color": color if sig else MUTED,
                                    "lineHeight": "1", "marginBottom": "0.3rem"}),
                    html.Div(f"${proj_lo:,.0f}–${proj_hi:,.0f}",
                             style={"fontSize": "0.68rem", "fontFamily": "Ubuntu Mono, monospace",
                                    "color": MUTED}),
                ],
                style={"borderLeft": f"3px solid {color if sig else BORDER}",
                       "paddingLeft": "0.85rem"},
            ),
            dbc.Tooltip(tooltip_text, target=badge_id, placement="bottom"),
            ],
            md=4,
        )

    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        segment_overview_card(
                            name="Men's Email",
                            color=MENS_COLOUR,
                            n=n_mens,
                            revenue_per=avg_mens,
                            conversion_rate=conv_mens,
                            rev_lift=lift_mens,
                            rev_pct=rev_pct_mens,
                            rev_sig=mens_sig,
                            conv_lift_pp=conv_mens - conv_control,
                            conv_pct=conv_pct_mens,
                        ),
                        md=4,
                        className="mb-3",
                    ),
                    dbc.Col(
                        segment_overview_card(
                            name="Women's Email",
                            color=WOMENS_COLOUR,
                            n=n_womens,
                            revenue_per=avg_womens,
                            conversion_rate=conv_womens,
                            rev_lift=lift_womens,
                            rev_pct=rev_pct_womens,
                            rev_sig=wom_sig,
                            conv_lift_pp=conv_womens - conv_control,
                            conv_pct=conv_pct_womens,
                        ),
                        md=4,
                        className="mb-3",
                    ),
                    dbc.Col(
                        segment_overview_card(
                            name="Control",
                            color=CTRL_COLOUR,
                            n=n_control,
                            revenue_per=avg_control,
                            conversion_rate=conv_control,
                            is_control=True,
                        ),
                        md=4,
                        className="mb-3",
                    ),
                ],
                className="mb-4 g-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Headline Finding"),
                                dbc.CardBody(
                                    [
                                        html.P(
                                            headline,
                                            style={"fontSize": "0.82rem", "color": MUTED, "marginBottom": "1.25rem"},
                                        ),
                                        dbc.Row(
                                            [
                                                _hl_col(
                                                    "Men's Email", MENS_COLOUR, mens_sig,
                                                    lift_mens, mens_lo, mens_hi,
                                                    proj_mens, proj_mens_lo, proj_mens_hi,
                                                ),
                                                _hl_col(
                                                    "Women's Email", WOMENS_COLOUR, wom_sig,
                                                    lift_womens, wom_lo, wom_hi,
                                                    proj_womens, proj_wom_lo, proj_wom_hi,
                                                ),
                                            ],
                                            className="g-3 mb-3",
                                        ),
                                        html.P(
                                            "Difference-in-means from the randomised experiment; confidence intervals "
                                            "use a percentile bootstrap (2k resamples) to account for the "
                                            "zero-inflated spend distribution. Subsequent tabs test this with "
                                            "matching, Bayesian and uplift methods.",
                                            className="text-muted small mb-0",
                                        ),
                                    ]
                                ),
                            ],
                            style={**CARD_STYLE, "borderLeft": f"3px solid {headline_color}"},
                            className="h-100"
                        ),
                        md=6,
                        className="mb-3",
                    ),
                    dbc.Col(
                        [
                            section_header("About this Dataset"),
                            html.P(
                                "The Hillstrom MineThatData dataset captures a randomised email marketing "
                                "experiment across 64k US retail customers. Two treatment groups received "
                                "targeted email campaigns (Men's or Women's catalogue), while the control "
                                "group received nothing. The groups were split around 33% each, and the "
                                "experiment ran for 30 days. The dataset includes customer attributes "
                                "(recency, frequency, monetary value, channel) and outcomes (conversion, spend).",
                                className="small text-muted"
                            ),
                            html.P(
                                "The causal question: does receiving an email cause customers to spend more? "
                                "Even in a randomised experiment, causal analysis adds value by quantifying "
                                "uncertainty, identifying which customers respond most (HTE), and stress-testing "
                                "results across different methodologies.",
                                className="small text-muted"
                            ),
                        ],
                        md=6,
                        className="mb-3",
                    ),
                ],
                className="mb-4 g-3 align-items-start"
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            section_header("Spend Distribution by Segment (among spenders)"),
                            dcc.Graph(id="tab1-box", figure=_fig_spend_box(), config=GRAPH_CONFIG)
                        ],
                        md=6,
                    ),
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
                        ],
                        md=6,
                    ),
                ],
                className="mb-4 g-3 align-items-start"
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
                                "For each email recipient we search the control group for the closest lookalike ",
                                "on things you could see before the send: spend history, recency, channel, ",
                                "broad postcode type, which catalogues someone leans toward, and similar ",
                                "flags. If even the nearest control still looks too different, ",
                                "we omit that shopper from this sensitivity view instead of stretching the comparison. ",
                                "Assignments were ",
                                html.Strong("randomised"),
                                ", so almost everyone pairs cleanly ",
                                "and the headline uplift barely budges versus always grabbing the ",
                                "nearest neighbour. For the algebraic details (propensity model, ",
                                "logit caliper, pooled bootstrap), expand ",
                                html.Strong("Methodology"),
                                " below.",
                            ],
                            className="mb-2 small"
                        ),
                        html.P(
                            [
                                html.Strong("Why this matters: "),
                                "This view is styled like the ",
                                html.Em("observational"),
                                " dashboards marketers run when randomisation is unavailable; ",
                                "it does not replace the Overview or Tab 5 estimates that lean on the ",
                                html.Strong("randomised"),
                                " assignment. ",
                                "The Love plot shows whether shopper traits line ",
                                "up sensibly ",
                                html.Em("before"),
                                " and ",
                                html.Em("after"),
                                " pairing, and the bar chart repeats the lift with ",
                                "a band computed by redoing matching 200 times on resampled data ",
                                "(a practical stress test rather than textbook matching confidence intervals)."
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
                        [
                            html.Strong("Statistical machinery (matching): "),
                            "A logistic regression uses nine pre-treatment covariates—recency, history, men's and "
                            "women's catalogue interest flags, two zip-type indicators (suburban/rural vs urban), "
                            "two channel indicators (web / multichannel vs phone), and the newbie flag—to estimate "
                            "each customer's ",
                            html.Em("propensity score"),
                            ": the modelled probability they would appear in the email arm given those traits.",
                            " Matching is ",
                            html.Strong("1:1 nearest neighbour on the propensity score"),
                            " with replacement.",
                            " A ",
                            html.Strong("caliper"),
                            " trims pairs where the ",
                            html.Em("absolute gap in logit(propensity)"),
                            " exceeds ",
                            html.Strong("0.2 times the pooled standard deviation"),
                            " of logit(PS) across all rows in that arm-vs-control subset.",
                            " Unmatched treated customers are omitted from ATT and balance diagnostics on the ",
                            "\"after\" cohort."
                        ]
                    ),
                    html.P(
                        "Because Hillstrom is randomised, treatment assignment is exogenous by design and "
                        "the primary causal estimand is an ITT/ATE contrast (see Overview bootstrap and Tab 5 "
                        "population-weighted OLS). "
                        "PSM illustrates observational workflows: overlap tightening, pruning, and covariance "
                        "balance on the matched cohort—not the project's headline estimator."
                    ),
                    html.P(
                        "Uncertainty is a pooled re-fit/re-match bootstrap (200 replicates): each replicate "
                        "resamples treated+control, re-estimates propensity, reapplies NN + caliper, and "
                        "recomputes ATT on surviving pairs. Interpret as a heuristic sensitivity envelope, "
                        "not textbook nearest-neighbour CIs."
                    ),
                    html.P(
                        "The Love plot contrasts all treated covariates to pooled controls ",
                        "(before pairing) versus caliper-kept treated units to their ",
                        "matched controls."
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
            html.Div(
                [
                    html.Span([html.Strong("Hurdle model"), ": Bernoulli x LogNormal"]),
                    html.Span(className="sep"),
                    html.Span("2,000 draws x 2 chains"),
                    html.Span(className="sep"),
                    html.Span("Full arm data (no subsampling)"),
                ],
                className="overview-context",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            section_header("Comparison"),
                            dbc.RadioItems(
                                id="bayes-pair-selector",
                                options=[
                                    {"label": "Men's vs Control", "value": "mens_vs_control"},
                                    {"label": "Women's vs Control", "value": "womens_vs_control"},
                                    {"label": "Men's vs Women's", "value": "mens_vs_womens"},
                                ],
                                value="mens_vs_control",
                                inline=True,
                                className="segmented-control mt-2",
                            ),
                        ]
                    ),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            section_header("Posterior Summary"),
                            html.Div(id="bayes-kpi-cards", className="mt-2"),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            section_header("Posterior Distribution — Treatment Effect δ"),
                            dcc.Graph(id="bayes-posterior-plot", config=GRAPH_CONFIG, className="mt-2"),
                        ],
                        md=8,
                    ),
                ],
                className="mb-4",
            ),
            dbc.Card(
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Div(
                                            [
                                                html.Span("Practical Significance (ROPE)",
                                                          style={**SECTION_HEADER_STYLE,
                                                                 "borderBottom": "none",
                                                                 "paddingBottom": 0,
                                                                 "marginBottom": "0.3rem",
                                                                 "display": "block"}),
                                                html.Span(
                                                    "Posterior mass outside the Region of Practical Equivalence. "
                                                    "Set ±$X to the smallest per-customer lift worth acting on.",
                                                    className="small text-muted",
                                                    style={"display": "block", "marginBottom": "0.6rem"},
                                                ),
                                            ],
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
                                                    debounce=True,
                                                ),
                                                dbc.InputGroupText("per customer"),
                                            ],
                                            size="sm",
                                            style={"maxWidth": "240px"},
                                        ),
                                    ],
                                    md=5,
                                    className="d-flex flex-column justify-content-center",
                                ),
                                dbc.Col(
                                    html.Div(id="rope-result-card"),
                                    md=7,
                                    className="d-flex align-items-center",
                                ),
                            ],
                            className="g-3",
                        ),
                    ],
                    style={"paddingTop": "1rem", "paddingBottom": "1rem"},
                ),
                style={**CARD_STYLE, "borderLeft": f"3px solid {ACCENT}"},
                className="mb-4",
            ),
            section_header("Model Diagnostics & Checks"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Button(
                                "▸ Posterior Predictive Check",
                                id="ppc-btn",
                                className="btn-methodology mb-2 w-100",
                                n_clicks=0,
                            ),
                            dbc.Collapse(
                                [
                                    html.P(
                                        [
                                            html.Strong("What this shows: "),
                                            "three hurdle-consistent posterior mimics stacked per arm. Row 1 simulates ",
                                            html.Code("Bernoulli(p) × LogNormal(μ, σ)"),
                                            " so the spike at ",
                                            html.Code("$0"),
                                            " enters the replicated spend distribution alongside the skewed positives. ",
                                            "Row 2 conditions on converters only versus observed positive amounts; ",
                                            "row 3 compares the replicated batch conversion fractions to empirical rates. ",
                                            html.Strong("What to watch for: "),
                                            "overall alignment in mass at zero (row 1), bulk positive-tail shape ",
                                            "(row 2), and calibrated conversion dispersion (row 3). Discrete catalogue ",
                                            "price ladders still spike the observed converters; judge LogNormal ",
                                            "fit on the smoothed analogue, not every SKU notch.",
                                        ],
                                        className="text-muted small mb-2",
                                    ),
                                    dcc.Graph(id="bayes-ppc-plot", config=GRAPH_CONFIG),
                                ],
                                id="ppc-collapse",
                                is_open=False,
                            ),
                        ],
                        md=12,
                        className="mb-2",
                    ),
                    dbc.Col(
                        [
                            html.Button(
                                "▸ MCMC Trace Plots",
                                id="trace-btn",
                                className="btn-methodology mb-2 w-100",
                                n_clicks=0,
                            ),
                            dbc.Collapse(
                                dcc.Graph(id="bayes-trace-plot", config=GRAPH_CONFIG),
                                id="trace-collapse",
                                is_open=False,
                            ),
                        ],
                        md=12,
                        className="mb-2",
                    ),
                    dbc.Col(
                        [
                            html.Button(
                                "▸ Convergence Diagnostics (R̂, ESS)",
                                id="diag-btn",
                                className="btn-methodology mb-2 w-100",
                                n_clicks=0,
                            ),
                            dbc.Collapse(
                                html.Div(id="bayes-diagnostics-table"),
                                id="diag-collapse",
                                is_open=False,
                            ),
                        ],
                        md=12,
                        className="mb-2",
                    ),
                ],
                className="mb-3 mt-2",
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
                        "95% of the posterior probability, i.e. a 95% probability the true expected "
                        "spend difference lies in this range (given the model and data)."
                    ),
                    html.P(
                        "The ROPE (Region of Practical Equivalence) lets you define a minimum effect "
                        "size that matters for business decisions. The dashboard shows the probability "
                        "mass outside the ROPE."
                    ),
                    html.P(
                        "Posterior predictive checks draw Monte Carlo batches where each replicated "
                        "customer spends zero or samples a fresh LogNormal amount conditional on "
                        "conversion — mirroring the generative hurdle story rather than extrapolating "
                        "only the conditional tail likelihood."
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
                        "treated/control ratio. The chart overlays a random-targeting baseline "
                        "line from zero to the full-population gain; ranking quality is the "
                        "excess area above that baseline. The decile chart shows actual spend lift for "
                        "customers ranked by predicted uplift: good models show declining lift."
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
                        "S-Learner with a RandomForest and 'treatment x covariate' interactions is known "
                        "to shrink CATE toward zero when outcome variance is large relative to the "
                        "treatment signal: a pattern we see here, where the T-Learner's average "
                        "CATE tends to be larger in magnitude than the S-Learner's."
                    ),
                    html.P(
                        "Because assignment is randomised, these models focus on treatment-effect heterogeneity "
                        "and policy ranking rather than confounding control. HTE estimates typically carry "
                        "more sampling uncertainty than ATE estimates and should be treated as directional "
                        "unless accompanied by explicit uncertainty intervals."
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
                        "observed baseline covariates. In this randomised experiment, covariate adjustment "
                        "is primarily for precision. All three arms (Men, Women, Control) are estimated in "
                        "one model via treatment indicators."
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
                        "by OLS (not IV or 2SLS). P-values for many interaction terms are exploratory unless "
                        "you apply a multiplicity adjustment (e.g., Holm/FDR) for confirmatory claims."
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
                            html.P(
                                [
                                    html.Strong("RCT-first read: "),
                                    "Difference-in-means on the Overview (bootstrap) and ",
                                    html.Strong("OLS population-weighted ATE"),
                                    " with HC3 robust errors align with ",
                                    "randomised-design ITT contrasts. ",
                                    "PSM here is an ATT on a caliper-pruned ",
                                    "matched subset—as if treatment were selective—purely ",
                                    "for diagnostics and observational-method comparison.",
                                ],
                                className="text-muted small mb-2",
                            ),
                            html.Div(id="comparison-table"),
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
                                                "Matched-sample ATT with a 0.2 SD (logit propensity) caliper: unmatched treated rows are discarded. Bootstrapped CIs reflect re-fit/rematch variability, not textbook NN match CIs."
                                            ),
                                        ],
                                        title="PSM sensitivity (observational-style)"
                                    ),
                                    dbc.AccordionItem(
                                        [
                                            html.P(
                                                "Provides a full posterior distribution over treatment effects. Best for communicating uncertainty and making probabilistic decisions. Uses a hurdle model (Bernoulli conversion x LogNormal amount)."
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
    dbc.Tab(tab2_layout(), label="2 PSM sensitivity", tab_id="tab-2"),
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
                                src="/assets/jc_logo_dark.png",
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

    att_pt = p.get("att_point", float("nan"))
    ci_lo = p.get("att_ci_lo", float("nan"))
    ci_hi = p.get("att_ci_hi", float("nan"))
    att_ok = bool(np.isfinite(att_pt))
    ci_ok = bool(np.isfinite(ci_lo) and np.isfinite(ci_hi))
    ci_str = (
        f"95% CI: ${ci_lo:.2f} – ${ci_hi:.2f}"
        if ci_ok
        else "95% CI unavailable (insufficient bootstrap samples)"
    )
    ps_dist = p.get("avg_ps_distance")
    cal_w = p.get("caliper_width")
    pct_mt = float(p.get("pct_matched", 100.0))
    n_drop = int(p.get("n_dropped_no_caliper", 0))
    n_tot = int(p.get("n_treated_total", p.get("n_matched", 0)))

    if ps_dist is not None and np.isfinite(ps_dist) and cal_w is not None:
        dist_str = (
            f"{pct_mt:.1f}% paired · mean |Δlogit(P)| = {ps_dist:.4f} (caliper ≤ {cal_w:.4f}) · "
            f"{n_drop:,} treated dropped"
        )
    else:
        dist_str = "Matching statistics unavailable"
    cs_str = f"{p['cs_lower']:.3f} - {p['cs_upper']:.3f}"
    pct_label = f"{pct_mt:.1f}% matched"

    kpis = html.Div(
        [
            kpi_card(
                f"${att_pt:.2f}" if att_ok else "—",
                f"ATT — {arm_label}",
                ci_str,
                (att_pt > 0) if att_ok else None,
                color=arm_color,
                accent=arm_color,
                info=(
                    "Average Treatment Effect on the Treated (ATT): mean spend difference between "
                    "each caliper-retained treated row and its nearest matched control."
                ),
                info_id="psm-info-att"
            ),
            kpi_card(
                pct_label,
                "% treated retained after caliper",
                f"{p.get('n_matched', 0):,} paired of {n_tot:,} treated",
                None,
                accent=ACCENT,
                info=(
                    "1:1 nearest-neighbour pairing on propensity scores with replacement. "
                    "Pairs are discarded if the absolute gap in logit(propensity score) exceeds "
                    "0.2× pooled SD of logit(PS) among all pooled rows."
                ),
                info_id="psm-info-pct"
            ),
            kpi_card(
                f"{p.get('n_matched', 0):,}",
                "Matched pairs",
                dist_str,
                None,
                accent=MUTED,
                info=(
                    "Matched rows only. Mean |Δlogit(PS)| along accepted pairs summarizes "
                    "how tight NN matches were before caliper pruning."
                ),
                info_id="psm-info-pairs"
            ),
            kpi_card(
                cs_str,
                "Common support range",
                info=(
                    "Overlap window on raw propensity scores across both groups. Wide overlap is "
                    "expected here; the logit caliper is what enforces pairwise closeness."
                ),
                info_id="psm-info-cs"
            ),
        ]
    )

    # Propensity score distribution (full cohorts)
    ps_fig = go.Figure()
    ps_fig.add_trace(
        go.Histogram(
            x=p["propensity_treated"],
            name=f"{arm_label} (all treated)",
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
        title=(
            "Propensity score distribution<br>"
            f"<sup>Overlap diagnostic - {pct_mt:.1f}% retained after 0.2 SD logit caliper</sup>"
        ),
        xaxis_title="Propensity score",
        yaxis_title="Count",
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5
        ),
        margin=dict(t=50, b=70)
    )

    # Love plot — “after” = caliper-kept treated vs their matched controls
    covs = list(p["smd_before"].keys())
    cov_labels = [COVARIATE_LABELS.get(c, c) for c in covs]
    smd_before = [p["smd_before"][c] for c in covs]
    sad = p.get("smd_after", {})
    if isinstance(sad, dict) and sad:
        smd_after_plot = []
        for c in covs:
            v = sad.get(c, float("nan"))
            smd_after_plot.append(float(v) if pd.notna(v) else float("nan"))
    else:
        smd_after_plot = [float("nan")] * len(covs)

    love_fig = go.Figure()
    love_fig.add_trace(
        go.Scatter(
            x=smd_before,
            y=cov_labels,
            mode="markers",
            name="Before (all treated vs controls)",
            marker=dict(color=DANGER, size=10, symbol="circle"),
            hovertemplate="%{y}<br>SMD: %{x:.3f}<extra>Before matching</extra>",
        )
    )
    love_fig.add_trace(
        go.Scatter(
            x=smd_after_plot,
            y=cov_labels,
            mode="markers",
            name="After (retained cohort vs matched ctrl)",
            marker=dict(color=SUCCESS, size=10, symbol="diamond"),
            hovertemplate="%{y}<br>SMD: %{x:.3f}<extra>After caliper NN</extra>",
        )
    )
    love_fig.add_vline(
        x=0.1, line_dash="dash", line_color=WARNING, annotation_text="0.1 threshold"
    )
    love_fig.add_vline(x=-0.1, line_dash="dash", line_color=WARNING)
    love_fig.add_vline(x=0, line_color=BORDER)
    love_fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=(
            "Love plot: covariate balance<br>"
            f"<sup>{p.get('n_matched', 0):,} caliper-kept treated units</sup>"
        ),
        xaxis_title="Standardised mean difference",
        yaxis=dict(automargin=True),
        margin=dict(t=50, b=70, l=195, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )

    if att_ok and ci_ok:
        bar_y = [att_pt, ci_lo, ci_hi]
        bar_text = [f"${v:.2f}" for v in bar_y]
    else:
        bar_y = [0.0, 0.0, 0.0]
        bar_text = ["—", "—", "—"]

    stats_fig = go.Figure(
        go.Bar(
            x=["ATT estimate", "CI lower", "CI upper"],
            y=bar_y,
            marker_color=[arm_color, BORDER, BORDER],
            text=bar_text,
            textposition="outside",
            textfont=dict(color=TEXT),
            hovertemplate="%{x}: $%{y:.2f}<extra></extra>",
        )
    )
    stats_fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="ATT with pooled re-fit/rematch bootstrap (200 reps)",
        yaxis_title="Effect on spend ($)",
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
    rope_val = rope_val if rope_val is not None else 0  # Dash returns None for empty number inputs
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
                accent=arm_color,
                info=(
                    "95% Highest Density Interval: the narrowest range containing 95% of the "
                    "posterior probability. These are the most credible values of the per-customer "
                    "treatment effect."
                ),
                info_id="bayes-kpi-hdi-info",
            ),
            kpi_card(
                f"{p_pos:.1%}",
                "P(effect > 0)",
                color=SUCCESS if p_pos > 0.9 else WARNING,
                accent=SUCCESS if p_pos > 0.9 else WARNING,
                info=(
                    "Posterior probability that the treatment truly lifts per-customer spend above zero. "
                    "Values above 95% are strong evidence of a positive effect. Values near 50% mean "
                    "the data is indifferent about the direction."
                ),
                info_id="bayes-kpi-ppos-info",
            ),
            kpi_card(
                f"${b['mean_a']:.2f}",
                f"Mean spend - {b['arm_a_label']}",
                accent=arm_color,
                info=(
                    "Posterior mean of expected per-customer spend for this arm, combining the "
                    "conversion probability and the log-normal amount component of the hurdle model."
                ),
                info_id="bayes-kpi-meana-info",
            ),
            kpi_card(
                f"${b['mean_b']:.2f}",
                f"Mean spend - {b['arm_b_label']}",
                accent=CTRL_COLOUR,
                info=(
                    "Posterior mean of expected per-customer spend for this arm, combining the "
                    "conversion probability and the log-normal amount component of the hurdle model."
                ),
                info_id="bayes-kpi-meanb-info",
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
        title=f"{b['arm_a_label']} vs {b['arm_b_label']}",
        xaxis_title="Effect on Spend ($)",
        yaxis_title="Density",
        margin=dict(t=40, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )

    p_outside_rope = float(np.mean((delta > rope_val) | (delta < -rope_val)))
    rope_color = SUCCESS if p_outside_rope > 0.9 else WARNING
    verdict = (
        "Strong evidence the effect is practically non-zero."
        if p_outside_rope > 0.9
        else "Not yet decisive, treat as practically equivalent to zero for now."
    )
    rope_card = html.Div(
        [
            html.Div(
                [
                    html.Span(f"{p_outside_rope:.1%}",
                              style={"fontSize": "2rem", "fontFamily": "Oswald, sans-serif",
                                     "fontWeight": "500", "color": rope_color, "lineHeight": "1"}),
                    html.Span(
                        [
                            f"  P(|δ| > ${rope_val})",
                            html.Span(" ⓘ", id="bayes-rope-info",
                                      style={"cursor": "help", "color": MUTED,
                                             "fontSize": "0.85em", "marginLeft": "4px"}),
                        ],
                        style={"fontSize": "0.78rem", "color": MUTED,
                               "fontFamily": "Ubuntu Mono, monospace",
                               "textTransform": "uppercase", "letterSpacing": "0.08em",
                               "marginLeft": "10px", "verticalAlign": "middle"},
                    ),
                ],
                style={"display": "flex", "alignItems": "baseline", "marginBottom": "0.3rem"},
            ),
            html.Div(verdict, className="small text-muted",
                     style={"fontStyle": "italic", "marginBottom": 0}),
            dbc.Tooltip(
                f"Posterior probability that the true per-customer treatment effect exceeds ±${rope_val} "
                f"in magnitude. Values inside the ROPE are treated as practically equivalent to zero. "
                f"Probability outside is the evidence the effect is large enough to matter.",
                target="bayes-rope-info",
                placement="top",
                style={"fontFamily": "Ubuntu, sans-serif", "fontSize": "0.8rem"},
            ),
        ],
        style={"borderLeft": f"3px solid {rope_color}", "paddingLeft": "0.9rem"},
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
                text=f"<b>MCMC Diagnostics</b><br>R̂: {rhat}<br>Bulk ESS: {bulk_ess}<br>Tail ESS: {tail_ess}",
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
    Input("bayes-pair-selector", "value"),
    State("ppc-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_ppc(n, pair_key, is_open):
    # Toggle is_open only on button click; pair changes only refresh the figure.
    new_is_open = not is_open if ctx.triggered_id == "ppc-btn" else is_open
    b = BAYESIAN[pair_key]
    lab_a = b["arm_a_label"]
    lab_b = b["arm_b_label"]
    pack = b.get("ppc_pack")

    if pack is None:
        fig = go.Figure()
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            title="Posterior predictive: rebuild `.cache/results.pkl` after upgrading causal_utils",
            height=420,
            margin=dict(t=60, b=50),
        )
        return new_is_open, fig

    nd = pack.get("ppc_n_draws", "—")
    nf = pack.get("ppc_n_fake_per_draw", "—")

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            f"Full spend — {lab_a}",
            f"Full spend — {lab_b}",
            f"Amount | spend > 0 — {lab_a}",
            f"Amount | spend > 0 — {lab_b}",
            f"Conversion rate mimic — {lab_a}",
            f"Conversion rate mimic — {lab_b}",
        ),
        vertical_spacing=0.09,
        horizontal_spacing=0.07,
    )

    obs_sa = b.get("observed_spend_a")
    obs_sb = b.get("observed_spend_b")
    ppc_sa = pack.get("ppc_spend_display_a")
    ppc_sb = pack.get("ppc_spend_display_b")
    obs_pa = b.get("observed_amount_a")
    obs_pb = b.get("observed_amount_b")
    ppc_pa = pack.get("ppc_amount_pos_a")
    ppc_pb = pack.get("ppc_amount_pos_b")
    ocra = b.get("obs_conv_rate_a")
    ocrb = b.get("obs_conv_rate_b")
    pcrma = pack.get("ppc_conv_rep_mean_a")
    pcrmb = pack.get("ppc_conv_rep_mean_b")

    def _add_hist_pair(row, col, obs_x, ppc_x, obs_name, ppc_name, color_o, color_p):
        if obs_x is None or ppc_x is None or len(obs_x) == 0 or len(ppc_x) == 0:
            return
        fig.add_trace(
            go.Histogram(
                x=obs_x,
                name=obs_name,
                histnorm="probability density",
                marker_color=color_o,
                opacity=0.42,
                nbinsx=70,
                showlegend=(row == 1 and col == 1),
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Histogram(
                x=ppc_x,
                name=ppc_name,
                histnorm="probability density",
                marker_color=color_p,
                opacity=0.42,
                nbinsx=70,
                showlegend=(row == 1 and col == 1),
            ),
            row=row,
            col=col,
        )

    _add_hist_pair(
        1, 1, obs_sa, ppc_sa, f"Observed — {lab_a}", f"PPC mimic — {lab_a}",
        MENS_COLOUR, ACCENT,
    )
    _add_hist_pair(
        1, 2, obs_sb, ppc_sb, f"Observed — {lab_b}", f"PPC mimic — {lab_b}",
        WOMENS_COLOUR, CTRL_COLOUR,
    )
    _add_hist_pair(
        2, 1, obs_pa, ppc_pa, f"Obs converters — {lab_a}", f"PPC | spend>0 — {lab_a}",
        MENS_COLOUR, ACCENT,
    )
    _add_hist_pair(
        2, 2, obs_pb, ppc_pb, f"Obs converters — {lab_b}", f"PPC | spend>0 — {lab_b}",
        WOMENS_COLOUR, CTRL_COLOUR,
    )

    if pcrma is not None and len(pcrma):
        fig.add_trace(
            go.Histogram(
                x=pcrma,
                nbinsx=45,
                name="PPC draw mean conversion",
                marker_color=ACCENT,
                opacity=0.55,
                showlegend=False,
            ),
            row=3,
            col=1,
        )
    if ocra is not None and np.isfinite(ocra):
        fig.add_vline(
            x=ocra,
            line_dash="dash",
            line_color=TEXT,
            annotation_text=f"Observed {ocra * 100:.2f}%",
            annotation_position="top",
            row=3,
            col=1,
        )

    if pcrmb is not None and len(pcrmb):
        fig.add_trace(
            go.Histogram(
                x=pcrmb,
                nbinsx=45,
                name="PPC draw mean conversion",
                marker_color=CTRL_COLOUR,
                opacity=0.55,
                showlegend=False,
            ),
            row=3,
            col=2,
        )
    if ocrb is not None and np.isfinite(ocrb):
        fig.add_vline(
            x=ocrb,
            line_dash="dash",
            line_color=TEXT,
            annotation_text=f"Observed {ocrb * 100:.2f}%",
            annotation_position="top",
            row=3,
            col=2,
        )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        barmode="overlay",
        height=980,
        title=(
            "Hurdle posterior predictive check "
            f"(Bernoulli × LogNormal draws; {nd} posterior slices × {nf} "
            "synthetic customers per slice, downsampled for spend histograms)"
        ),
        margin=dict(t=80, b=46, l=54, r=36),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
    )

    for r in (1, 2):
        for c in (1, 2):
            fig.update_xaxes(title_text="Spend ($)", row=r, col=c)
            fig.update_yaxes(title_text="Density", row=r, col=c)

    fig.update_xaxes(title_text="Conversion fraction", tickformat=".0%", row=3, col=1)
    fig.update_xaxes(title_text="Conversion fraction", tickformat=".0%", row=3, col=2)
    fig.update_yaxes(title_text="Count", row=3, col=1)
    fig.update_yaxes(title_text="Count", row=3, col=2)

    return new_is_open, fig


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
                "R̂",
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
                info=(
                    "Average Conditional Average Treatment Effect: the mean individual-level uplift "
                    "predicted by the model across all customers. This should be close to the overall "
                    "ATE seen on other tabs. Divergence suggests heavy tails or modelling artefacts."
                ),
                info_id="uplift-kpi-avg-info",
            ),
            kpi_card(
                f"${np.percentile(cate, 90):.2f}",
                "90th percentile CATE",
                "High-responder threshold",
                accent=ACCENT,
                info=(
                    "The uplift threshold above which the top 10% of customers sit. Useful for sizing "
                    "a high-value targeting segment: mailing only customers with predicted uplift above "
                    "this value captures the strongest responders."
                ),
                info_id="uplift-kpi-p90-info",
            ),
            kpi_card(
                f"{np.mean(cate > 0):.1%}",
                "% customers with positive uplift",
                info=(
                    "Share of customers for whom the model predicts a positive treatment effect. A "
                    "ceiling on the profitable targetable audience: customers with negative predicted "
                    "uplift should not be mailed regardless of cost."
                ),
                info_id="uplift-kpi-ppos-info",
            ),
        ]
    )

    p1, p99 = np.percentile(cate, 1), np.percentile(cate, 99)
    cate_clipped = cate[(cate >= p1) & (cate <= p99)]  # display only, full CATE used for all analysis
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
        title="Heterogeneity importance (|treated - control| model diff)",
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

    # Plotly fill colours don't accept hex+alpha directly so convert to rgba string
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
            y=[0, qini_yd[-1] if len(qini_yd) else 0],
            mode="lines",
            name="Random baseline",
            line=dict(color=BORDER, dash="dash"),
            hoverinfo="skip",
        )
    )
    qini_auc_key = "qini_auc_s" if model == "s" else "qini_auc_t"
    qini_excess_key = "qini_excess_auc_s" if model == "s" else "qini_excess_auc_t"
    qini_auc = u.get(qini_auc_key, 0.0)
    qini_excess = u.get(qini_excess_key, 0.0)
    qini_fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"Qini Curve — {model_label} (AUUC = ${qini_auc:,.0f}, Excess vs random = ${qini_excess:,.0f})",
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
    zip_spell = {"Urban": "Urban", "Surburban": "Suburban", "Rural": "Rural"}  # raw data misspells "Suburban"
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
        """Population-weighted average of marginal effects within a (newbie, channel) group."""
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
            *TABLE_SELECTED,
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
    """Assemble a tidy DataFrame of point estimates and CIs across all 5 methods x 2 arms."""
    def _rnd_money(v):
        try:
            x = float(v)
            return round(x, 2) if np.isfinite(x) else None
        except (TypeError, ValueError):
            return None

    rows = []
    for arm in ["mens", "womens"]:
        arm_label = "Men's Email" if arm == "mens" else "Women's Email"
        p = PSM[arm]
        rows.append(
            {
                "Method": "PSM (ATT, caliper NN)",
                "Arm": arm_label,
                "Estimate ($)": _rnd_money(p.get("att_point")),
                "CI Lower ($)": _rnd_money(p.get("att_ci_lo")),
                "CI Upper ($)": _rnd_money(p.get("att_ci_hi")),
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
            *TABLE_SELECTED,
        ],
    )

    def forest_plot(arm_label, color):
        sub = comp_df[comp_df["Arm"] == arm_label].copy()
        fig = go.Figure()
        for i, row in sub.iterrows():
            est_cell = row["Estimate ($)"]
            if pd.isna(est_cell) or est_cell is None:
                continue
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
    mens_valid = [float(v) for v in mens_estimates if pd.notna(v)]
    womens_valid = [float(v) for v in womens_estimates if pd.notna(v)]
    mens_min = min(mens_valid) if mens_valid else 0.0
    mens_max = max(mens_valid) if mens_valid else 0.0
    womens_min = min(womens_valid) if womens_valid else 0.0
    womens_max = max(womens_valid) if womens_valid else 0.0

    # Robust verdict: don't flip on a single near-zero estimate. Treat
    # |effect| < $0.10 as "noise zone": smaller than any plausible action
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
                        "Agreement strengthens credibility; divergence surfaces differing assumptions. "
                        "Randomisation-grounded contrasts are on the Overview and Tab 5 (OLS); "
                        "Bayesian focuses on posterior uncertainty for distribution shifts; uplift "
                        "prioritises out-of-sample targeting discrimination.",
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
    app.run(debug=False, port=8050)
