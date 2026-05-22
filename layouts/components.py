
from dash import html
import dash_bootstrap_components as dbc
from dashboard.theme import (
    SUCCESS, DANGER, MUTED, TEXT,
    KPI_LABEL_STYLE, KPI_VALUE_STYLE, KPI_DELTA_STYLE,
    SECTION_HEADER_STYLE,
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
    """Metric stat block with optional delta row, info tooltip, and pct-change figure.

    The info affordance is a dotted underline on the label (no glyph); the
    pct-change is restrained mono text rather than a coloured arrow.
    """
    delta_color = (
        SUCCESS if delta_positive else DANGER if delta_positive is False else MUTED
    )

    if info and info_id:
        label_node = html.Span(label, id=info_id, className="info-term")
        extra = [dbc.Tooltip(info, target=info_id, placement="right")]
    else:
        label_node = label
        extra = []

    left_block = html.Div(
        [
            html.P(label_node, style=KPI_LABEL_STYLE),
            html.P(value, style={**KPI_VALUE_STYLE, "color": color}),
            html.P(delta or " ", style={**KPI_DELTA_STYLE, "color": delta_color}),
        ],
        style={"flex": "1"},
    )

    if pct_change is not None:
        pct_color = SUCCESS if pct_change >= 0 else DANGER
        right_block = html.Div(
            f"{pct_change:+.1f}%",
            style={
                "color": pct_color,
                "textAlign": "right",
                "alignSelf": "flex-end",
                "paddingLeft": "14px",
                "fontFamily": "IBM Plex Mono, monospace",
                "fontSize": "0.95rem",
                "fontWeight": "600",
                "minWidth": "58px",
            },
        )
        body_children = [html.Div([left_block, right_block], style={"display": "flex"}), *extra]
    else:
        body_children = [left_block, *extra]

    return html.Div(body_children, className="kpi-stat")

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
        delta_color = SUCCESS if positive else DANGER
        children = [
            html.Span(lift_text, className="delta-num", style={"color": delta_color}),
            html.Span(f"({pct_change:+.1f}%)", style={"color": MUTED}),
        ]
        if sig is not None:
            children.append(
                html.Span(
                    "significant" if sig else "not significant",
                    className="segment-sig-badge",
                    style={"color": SUCCESS if sig else MUTED},
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
                            html.Div(
                                html.Div(name, className="segment-card-name"),
                                className="segment-card-title",
                            ),
                        ],
                    ),
                    html.Div(
                        [html.Strong(f"{n:,}"), " users"],
                        className="segment-card-count",
                    ),
                ],
                className="segment-card-header",
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
        style={"borderTopColor": color},
    )

def section_header(text):
    return html.H5(text, style=SECTION_HEADER_STYLE)

def labeled_radio(label, id, options, value, className="dashboard-radio-group mb-2", inline=True):
    return html.Div(
        [
            html.Label(label, className="small text-muted"),
            dbc.RadioItems(id=id, options=options, value=value, inline=inline, className=className),
        ]
    )

def collapsible_panel(btn_id, collapse_id, label, content,
                      btn_className="btn-methodology mb-2 w-100"):
    return html.Div(
        [
            html.Button(label, id=btn_id, className=btn_className, n_clicks=0),
            dbc.Collapse(content, id=collapse_id, is_open=False),
        ]
    )

def labeled_input_group(label, children, size="sm", group_style=None):
    return html.Div(
        [
            html.Label(label, className="small text-muted"),
            dbc.InputGroup(children, size=size, style=group_style),
        ]
    )

def methodology_collapse(tab_id, content):
    return html.Div(
        [
            html.Button(
                "Methodology & assumptions",
                id=f"method-btn-{tab_id}",
                className="btn-methodology mb-2",
                n_clicks=0
            ),
            dbc.Collapse(
                dbc.Card(
                    dbc.CardBody(
                        content,
                        style={
                            "fontSize": "0.92rem",
                            "color": MUTED,
                            "lineHeight": "1.7",
                        },
                    ),
                    className="dashboard-card methodology-content",
                    style={"marginTop": "4px"},
                ),
                id=f"method-collapse-{tab_id}",
                is_open=False
            ),
        ],
        className="methodology-section",
    )

