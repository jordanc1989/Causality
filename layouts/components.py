
from dash import dcc, html
import dash_bootstrap_components as dbc
from dashboard.theme import (
    SUCCESS, DANGER, MUTED, TEXT, ACCENT,
    KPI_LABEL_STYLE, KPI_VALUE_STYLE, KPI_DELTA_STYLE,
    SECTION_HEADER_STYLE, GRAPH_CONFIG, LOCKED_GRAPH_CONFIG,
)


def loading(children):
    """House spinner for callback-fed content. `delay_show` keeps fast
    in-page updates from flickering the overlay, the spinner only appears
    when a real wait (page navigation, slow client) is happening."""
    return dcc.Loading(children, type="circle", color=ACCENT, delay_show=300)


def spec_strip(*items):
    """Methodology kicker shown at the top of each tab.

    Each item is either a string ("Hurdle model"), a `(label, value)` tuple
    rendered as `label <strong>value</strong>`, or any already-built component.
    Items are separated by a small bullet so the strip reads like a chart
    source line.
    """
    children = []
    for i, item in enumerate(items):
        if i:
            children.append(html.Span(className="sep"))
        if isinstance(item, tuple) and len(item) == 2:
            label, value = item
            children.append(html.Span([f"{label} ", html.Strong(str(value))]))
        elif isinstance(item, str):
            children.append(html.Span(item))
        else:
            children.append(item)
    return html.Div(children, className="spec-strip")


def masthead_dateline(source, updated=None, source_href=None, code_href=None):
    """Source attribution slug for the masthead. FT/NYT-style trailing line.

    `source_href` links the source name to the dataset's origin, `code_href`
    appends a "Code on GitHub" slug. Both open in a new tab.
    """
    source_node = html.Strong(source)
    if source_href:
        source_node = html.A(
            source_node,
            href=source_href,
            target="_blank",
            rel="noopener",
            className="dateline-link",
        )
    parts = [html.Span(["Source: ", source_node])]
    if updated:
        parts.extend([html.Span(className="sep"), html.Span(f"Updated {updated}")])
    if code_href:
        parts.extend(
            [
                html.Span(className="sep"),
                html.A(
                    "Code on GitHub",
                    href=code_href,
                    target="_blank",
                    rel="noopener",
                    className="dateline-link",
                ),
            ]
        )
    return html.Div(parts, className="masthead-dateline")


def page_lede(headline, kicker=None, caveat=None):
    """The page's lede block: optional mono kicker, serif headline, optional caveat.

    Replaces the inline `borderLeft` accent motif. The headline carries
    its own weight, no extra rule needed beside it.
    """
    children = []
    if kicker:
        children.append(html.Div(kicker, className="page-lede__kicker"))
    children.append(html.H2(headline, className="page-lede__headline"))
    if caveat:
        children.append(html.P(caveat, className="page-lede__caveat"))
    return html.Div(children, className="page-lede")


def headline_tile(kicker, value, label, meta, accent=None, significant=False):
    """A single "projected total lift" tile.

    Layout reads: small kicker (arm name) > big serif number > small label
    > meta line (CI). The arm colour only appears as a top rule when the
    arm's effect is significant, otherwise the rule stays muted.
    """
    style = {}
    if accent:
        style["--tile-accent"] = accent
    tile_class = "headline-tile"
    value_class = "headline-tile__value"
    if significant:
        tile_class += " headline-tile--sig"
        value_class += " headline-tile__value--sig"
    return html.Div(
        [
            html.Div(kicker, className="headline-tile__kicker"),
            html.Div(value, className=value_class),
            html.Div(label, className="headline-tile__label"),
            html.Div(meta, className="headline-tile__meta"),
        ],
        className=tile_class,
        style=style,
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

    The info affordance is a dotted underline on the label (no glyph). The
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

def section_col(title, *children, **kwargs):
    return dbc.Col([section_header(title), *children], **kwargs)

def graph(graph_id, figure=None, locked=False, **kwargs):
    config = LOCKED_GRAPH_CONFIG if locked else GRAPH_CONFIG
    props = {"id": graph_id, "config": config, **kwargs}
    if figure is not None:
        # Static figure embedded at layout build time - never enters a
        # loading state, so no spinner wrapper.
        props["figure"] = figure
        return dcc.Graph(**props)
    # Callback-fed: wrap in a spinner so navigating to the page shows a
    # loading state instead of a flash of empty axes.
    return loading(dcc.Graph(**props))

def graph_col(graph_id, md=6, figure=None, locked=False, **kwargs):
    return dbc.Col(graph(graph_id, figure=figure, locked=locked, **kwargs), md=md)

def graph_row(*columns, className="mb-3"):
    return dbc.Row(list(columns), className=className)

def graph_row_ids(*items, className="mb-3", locked=False):
    return graph_row(
        *(
            graph_col(item[0], md=item[1], locked=item[2] if len(item) > 2 else locked)
            if isinstance(item, tuple)
            else graph_col(item, locked=locked)
            for item in items
        ),
        className=className,
    )

def kpi_graph_row(kpi_id, graph_id, graph_md=8):
    return graph_row(
        dbc.Col(
            loading(html.Div(id=kpi_id, className="kpi-stack")),
            md=12 - graph_md,
        ),
        graph_col(graph_id, md=graph_md),
    )

def labeled_radio(label, id, options, value, className="dashboard-radio-group mb-2", inline=True):
    return html.Div(
        [
            html.Label(label, className="small text-muted"),
            dbc.RadioItems(
                id=id, options=options, value=value, inline=inline, className=className,
                persistence=True, persistence_type="session",
            ),
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
