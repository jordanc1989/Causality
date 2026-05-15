import dash_bootstrap_components as dbc
from dash import html, dcc
from dashboard.theme import ACCENT, TEXT, MUTED, BG
from layouts.overview import tab1_layout
from layouts.psm import tab2_layout
from layouts.bayesian import tab3_layout
from layouts.uplift import tab4_layout
from layouts.ols import tab5_layout
from layouts.comparison import tab6_layout

# Inline SVG mark — a "cause → effect" node-link icon, themed for the dashboard.
# Inlining as a data URI keeps the navbar logo self-contained (no LFS asset,
# no missing-image broken icon when the binary file is unresolved).
_LOGO_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32' "
    "width='32' height='32' role='img' aria-label='Causal inference mark'>"
    f"<circle cx='7' cy='16' r='4' fill='{ACCENT}'/>"
    f"<line x1='11' y1='16' x2='20' y2='16' stroke='{TEXT}' "
    "stroke-width='1.8' stroke-linecap='round'/>"
    f"<polygon points='19,12.6 25,16 19,19.4' fill='{TEXT}'/>"
    f"<circle cx='25' cy='16' r='3' fill='none' stroke='{TEXT}' stroke-width='1.8'/>"
    "</svg>"
)


def _logo_data_uri():
    # url-encode the few characters that would otherwise need %-escaping in a
    # data URI
    encoded = _LOGO_SVG.replace("#", "%23").replace("\n", "")
    return f"data:image/svg+xml;charset=utf-8,{encoded}"


def build_tabs():
    # Visual ordering puts the randomisation-grounded estimators (Bayesian, OLS)
    # before the heterogeneity / observational diagnostics (Uplift, PSM).
    # `tab_id`s stay stable to the method they identify so callbacks and element
    # IDs don't need to be renumbered alongside the visual order.
    return [
        dbc.Tab(tab1_layout(), label="1 Overview", tab_id="tab-1"),
        dbc.Tab(tab3_layout(), label="2 Bayesian A/B", tab_id="tab-3"),
        dbc.Tab(tab5_layout(), label="3 Multi-Arm OLS", tab_id="tab-5"),
        dbc.Tab(tab4_layout(), label="4 Uplift / HTE", tab_id="tab-4"),
        dbc.Tab(tab2_layout(), label="5 PSM sensitivity", tab_id="tab-2"),
        dbc.Tab(tab6_layout(), label="6 Method Comparison", tab_id="tab-6"),
    ]


def build_layout():
    tabs = build_tabs()
    return html.Div(
        [
            dbc.Navbar(
                dbc.Container(
                    [
                        html.Div(
                            [
                                html.Img(
                                    src=_logo_data_uri(),
                                    alt="Causal inference mark",
                                    style={
                                        "height": "26px",
                                        "width": "26px",
                                        "marginRight": "0.7rem",
                                    },
                                ),
                                html.Span(
                                    "Causal",
                                    style={
                                        "fontFamily": "Ubuntu, sans-serif",
                                        "fontWeight": "700",
                                        "fontSize": "1.05rem",
                                        "color": ACCENT,
                                        "letterSpacing": "-0.02em",
                                    },
                                ),
                                html.Span(
                                    " inference",
                                    style={
                                        "fontFamily": "Ubuntu, sans-serif",
                                        "fontWeight": "500",
                                        "fontSize": "1.05rem",
                                        "color": TEXT,
                                    },
                                ),
                            ],
                            style={"display": "flex", "alignItems": "center"}
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "Hillstrom randomised email test",
                                    style={
                                        "fontFamily": "Ubuntu, sans-serif",
                                        "fontSize": "0.72rem",
                                        "fontWeight": "500",
                                        "color": MUTED,
                                    },
                                )
                            ],
                            className="ms-auto d-none d-md-flex align-items-center",
                        ),
                    ],
                    fluid=True,
                ),
                className="dashboard-navbar mb-0",
                color="dark",
                dark=True,
                sticky="top",
            ),
            dbc.Container(
                [
                    dcc.Location(id="url", refresh=False),
                    dcc.Loading(
                        id="tabs-loading",
                        type="circle",
                        color=ACCENT,
                        children=dbc.Tabs(
                            tabs,
                            id="main-tabs",
                            active_tab="tab-1",
                            className="dashboard-tabs",
                        ),
                    ),
                ],
                fluid=True,
            ),
        ],
        className="dashboard-app",
        style={"backgroundColor": BG, "minHeight": "100vh"},
    )
