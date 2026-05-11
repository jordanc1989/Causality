import dash_bootstrap_components as dbc
from dash import html
from dashboard.theme import ACCENT, TEXT, MUTED, BG
from layouts.overview import tab1_layout
from layouts.psm import tab2_layout
from layouts.bayesian import tab3_layout
from layouts.uplift import tab4_layout
from layouts.ols import tab5_layout
from layouts.comparison import tab6_layout


def build_tabs():
    return [
        dbc.Tab(tab1_layout(), label="1 Overview", tab_id="tab-1"),
        dbc.Tab(tab2_layout(), label="2 PSM sensitivity", tab_id="tab-2"),
        dbc.Tab(tab3_layout(), label="3 Bayesian A/B", tab_id="tab-3"),
        dbc.Tab(tab4_layout(), label="4 Uplift / HTE", tab_id="tab-4"),
        dbc.Tab(tab5_layout(), label="5 Multi-Arm OLS", tab_id="tab-5"),
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
                [dbc.Tabs(tabs, id="main-tabs", active_tab="tab-1", className="dashboard-tabs")],
                fluid=True,
            ),
        ],
        className="dashboard-app",
        style={"backgroundColor": BG, "minHeight": "100vh"},
    )
