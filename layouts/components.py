
from dash import html
import dash_bootstrap_components as dbc
from dashboard.theme import (
    SUCCESS, DANGER, MUTED, BORDER, TEXT,
    KPI_LABEL_STYLE, KPI_VALUE_STYLE, KPI_DELTA_STYLE,
    SECTION_HEADER_STYLE, CARD_STYLE,
)

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
        className="dashboard-card mb-2"
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
                    className="dashboard-card methodology-content",
                    style={**CARD_STYLE, "marginTop": "4px"}
                ),
                id=f"method-collapse-{tab_id}",
                is_open=False
            ),
        ],
        className="mt-3"
    )

