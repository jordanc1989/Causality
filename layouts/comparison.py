
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
                                    "Each row is one method's estimate of the average lift in spend "
                                    "per recipient, with its confidence range. The Overview, Bayesian "
                                    "and OLS rows are the trustworthy headline numbers because they "
                                    "use the random assignment directly. The PSM row is included as a "
                                    "diagnostic, its uncertainty band is shown on the PSM tab and "
                                    "deliberately left blank here so it isn't read as comparable to the "
                                    "other intervals.",
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
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.Span(
                                        "Agreement threshold ",
                                        className="small text-muted",
                                    ),
                                    html.Span(
                                        " ⓘ",
                                        id="comparison-noise-info",
                                        style={
                                            "cursor": "help",
                                            "color": "var(--muted)",
                                            "fontSize": "0.85em",
                                        },
                                    ),
                                ],
                                className="mb-1",
                            ),
                            dbc.InputGroup(
                                [
                                    dbc.InputGroupText(
                                        "|effect| <",
                                        className="dashboard-input-group-text",
                                    ),
                                    dbc.Input(
                                        id="comparison-noise-input",
                                        type="number",
                                        min=0,
                                        step=0.05,
                                        value=0.10,
                                        debounce=True,
                                        className="dashboard-input",
                                    ),
                                    dbc.InputGroupText(
                                        "treated as zero",
                                        className="dashboard-input-group-text",
                                    ),
                                ],
                                size="sm",
                                style={"maxWidth": "320px"},
                                className="mb-3",
                            ),
                            dbc.Tooltip(
                                "Estimates with absolute value below this threshold are "
                                "treated as 'near zero' and excluded from the cross-method "
                                "directional verdict, so a single noise-zone estimate "
                                "doesn't flip the conclusion.",
                                target="comparison-noise-info",
                                placement="right",
                            ),
                            html.Div(id="key-takeaway-card"),
                        ],
                        md=5,
                    ),
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
        className="py-5"
    )

