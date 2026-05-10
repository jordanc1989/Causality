
from dash import dcc, html
import dash_bootstrap_components as dbc
from dashboard.theme import *
from layouts.components import section_header

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
                                    "for diagnostics and observational-method comparison. ",
                                    "Tab 2 carries the rematch bootstrap sensitivity band; this table shows the ",
                                    "PSM ",
                                    html.Strong("point estimate only"),
                                    " so its interval is not mistaken for an HC3/HDI-style summary.",
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

