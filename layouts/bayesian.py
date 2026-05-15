
from dash import dcc, html
import dash_bootstrap_components as dbc
from dashboard.theme import *
from layouts.components import section_header, methodology_collapse

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
                style={**CARD_STYLE, "borderLeft": f"3px solid {ACCENT}"},
                className="dashboard-card mb-4",
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
                                            html.Code("Bernoulli(p) x LogNormal(μ, σ)"),
                                            " so the spike at ",
                                            html.Code("$0"),
                                            " enters the replicated spend distribution alongside the skewed positives. ",
                                            "Row 2 conditions on converters only versus observed positive amounts, ",
                                            "row 3 redraws each posterior slice at the ",
                                            html.Strong("full observed arm size"),
                                            " so batch conversion noise lines up with empirical rates (rows 1-2 use ",
                                            "a smaller synthetic batch only to keep spend histograms lightweight). ",
                                            html.Strong("What to watch for: "),
                                            "overall alignment in mass at zero (row 1), bulk positive-tail shape ",
                                            "(row 2), and calibrated conversion dispersion (row 3). Discrete catalogue ",
                                            "price ladders still spike the observed converters, judge LogNormal ",
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
                        [
                            "Priors: Beta(1, 1) (uniform) on conversion probability. ",
                            "Normal(μ = mean(log positive spend), σ = 2 x SD(log positive spend)) "
                            "on the log-mean of the amount component, HalfNormal(σ = SD(log positive spend)) "
                            "on the log-sigma. The Normal and HalfNormal priors are weakly informative ",
                            html.Strong("but data-derived"),
                            " (an empirical-Bayes choice): the prior location and scale are read off the ",
                            "pooled positive-spend log-distribution rather than fixed in advance. With ",
                            "~21k positive observations per arm the priors are dominated by the likelihood, ",
                            "but readers comparing to a textbook fully-subjective prior should know the ",
                            "scale was tuned to this dataset. MCMC is run with PyMC via the nutpie NUTS ",
                            "sampler (2,000 draws, 2 chains) on the full arm data — no subsampling.",
                        ]
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
        className="py-5"
    )

