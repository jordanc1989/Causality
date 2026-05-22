
from dash import dcc, html
import dash_bootstrap_components as dbc
from dashboard.theme import *
from layouts.components import section_header, methodology_collapse, collapsible_panel

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
                            section_header("Posterior summary"),
                            html.Div(id="bayes-kpi-cards", className="kpi-stack mt-2"),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            section_header("Posterior distribution · treatment effect δ"),
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
                                                html.Span("Practical significance (ROPE)",
                                                          style={**SECTION_HEADER_STYLE,
                                                                 "borderBottom": "none",
                                                                 "paddingBottom": 0,
                                                                 "marginBottom": "0.3rem",
                                                                 "display": "block"}),
                                                html.Span(
                                                    "Set this to the smallest per-customer lift worth acting on. "
                                                    "The result is how much of the posterior sits beyond that threshold.",
                                                    className="small text-muted",
                                                    style={"display": "block", "marginBottom": "0.6rem"},
                                                ),
                                            ],
                                        ),
                                        dbc.InputGroup(
                                            [
                                                dbc.InputGroupText("±$", className="dashboard-input-group-text"),
                                                dbc.Input(
                                                    id="rope-slider",
                                                    type="number",
                                                    min=0,
                                                    step=0.5,
                                                    value=1,
                                                    debounce=True,
                                                    className="dashboard-input",
                                                ),
                                                dbc.InputGroupText("per customer", className="dashboard-input-group-text"),
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
                style=CARD_STYLE,
                className="dashboard-card mb-4",
            ),
            section_header("Model diagnostics & checks"),
            dbc.Row(
                [
                    dbc.Col(
                        collapsible_panel(
                            "ppc-btn",
                            "ppc-collapse",
                            "Posterior predictive check",
                            [
                                html.P(
                                    [
                                        html.Strong("What this shows. "),
                                        "Three stacked checks per arm that compare the model's simulated data "
                                        "against the real data. The first row covers the full spend distribution "
                                        "including the spike at $0. The second row focuses on the size of "
                                        "purchases among customers who did spend. The third row checks the "
                                        "conversion rate at the actual arm size. ",
                                        html.Strong("What to look for. "),
                                        "The simulated and observed distributions should sit roughly on top of "
                                        "each other. Small spikes in the observed data come from real catalogue "
                                        "price points, so judge the fit on the overall shape rather than every "
                                        "individual bump.",
                                    ],
                                    className="text-muted small mb-2",
                                ),
                                dcc.Graph(id="bayes-ppc-plot", config=GRAPH_CONFIG),
                            ],
                        ),
                        md=12,
                        className="mb-2",
                    ),
                    dbc.Col(
                        collapsible_panel(
                            "trace-btn",
                            "trace-collapse",
                            "MCMC trace plots",
                            dcc.Graph(id="bayes-trace-plot", config=GRAPH_CONFIG),
                        ),
                        md=12,
                        className="mb-2",
                    ),
                    dbc.Col(
                        collapsible_panel(
                            "diag-btn",
                            "diag-collapse",
                            "Convergence diagnostics (R̂, ESS)",
                            html.Div(id="bayes-diagnostics-table"),
                        ),
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
                        "Spend is mostly zero with a long right tail. A plain Normal model would "
                        "fit badly because it puts weight on negative spend and can't represent "
                        "the spike at zero. The model splits the problem in two instead. A "
                        "Bernoulli decides whether the customer spends at all. A LogNormal "
                        "decides how much they spend if they do. The expected per-customer spend "
                        "is the product of those two pieces, and delta is the difference between "
                        "the two arms."
                    ),
                    html.P(
                        [
                            "Priors. Beta(1, 1) (uniform) on conversion probability. The Normal "
                            "on log-mean and HalfNormal on log-sigma are centred and scaled on "
                            "the pooled positive-spend distribution, which makes them weakly "
                            "informative but ",
                            html.Strong("data-derived"),
                            ". With about 21,000 positive observations per arm the data dominates "
                            "the priors, so this choice has very little influence on the result. "
                            "Sampling uses PyMC's nutpie NUTS sampler, 2,000 draws across 2 chains, "
                            "on the full arm data."
                        ]
                    ),
                    html.P(
                        "The 95% Highest Density Interval is the narrowest range that holds 95% "
                        "of the posterior. Given the model and the data, there's a 95% probability "
                        "the true per-customer effect sits inside it."
                    ),
                    html.P(
                        "The ROPE (Region of Practical Equivalence) is a range around zero that "
                        "you treat as 'not big enough to act on'. Set to the smallest "
                        "per-customer lift that would change a decision, the tab reports how "
                        "much of the posterior sits beyond it."
                    ),
                    html.P(
                        "The posterior predictive check simulates customers from the fitted model "
                        "and compares the simulated distribution to the real one. A good fit shows "
                        "the two distributions sitting on top of each other across the zero spike, "
                        "the positive tail, and the conversion rate."
                    ),
                ],
            ),
        ],
        fluid=True,
        className="py-5"
    )

