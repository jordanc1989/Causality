
from dash import html
import dash_bootstrap_components as dbc
from dashboard.theme import *
from layouts.components import (
    graph, graph_row_ids, kpi_graph_row, methodology_collapse,
    labeled_radio, labeled_input_group, spec_strip, section_header, loading,
)

def tab4_layout(**_kwargs):
    return dbc.Container(
        [
            spec_strip(
                "S- / X-Learner",
                "5-fold cross-fitting"
            ),
            dbc.Row(
                [
                    dbc.Col(
                        labeled_radio(
                            "Campaign arm:",
                            "uplift-arm-selector",
                            [
                                {"label": "Men's Email", "value": "mens"},
                                {"label": "Women's Email", "value": "womens"},
                            ],
                            "mens",
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        labeled_radio(
                            "Model:",
                            "uplift-model-selector",
                            [
                                {"label": "S-Learner", "value": "s"},
                                {"label": "X-Learner", "value": "x"},
                            ],
                            "x",
                        ),
                        md=6,
                    ),
                ]
            ),
            dbc.Row(
                dbc.Col(
                    html.Div(
                        [
                            html.Div(
                                "Which learner should I use?",
                                style={
                                    "fontFamily": SERIF,
                                    "fontWeight": "600",
                                    "color": TEXT,
                                    "marginBottom": "0.35rem",
                                },
                            ),
                            html.P(
                                [
                                    html.Strong("X-Learner:"),
                                    " Start with this, it's a good first view for targeting because it learns "
                                    "from both the mailed and control groups then blends "
                                    "the evidence into a steadier customer ranking.",
                                ],
                                className="small text-muted mb-1",
                            ),
                            html.P(
                                [
                                    html.Strong("S-Learner:"),
                                    " Use as a conservative check. It can pull individual uplift estimates towards zero when the signal "
                                    "is faint so treat as a sanity check.",
                                ],
                                className="small text-muted mb-0",
                            ),
                        ],
                        style={
                            "borderLeft": f"2px solid {BORDER_STRONG}",
                            "paddingLeft": "1rem",
                            "marginBottom": "1.25rem",
                        },
                    ),
                    md=12,
                )
            ),
            kpi_graph_row("uplift-kpi-cards", "uplift-cate-hist"),
            graph_row_ids("uplift-feat-imp", "uplift-decile-chart", locked=True),
            graph_row_ids("uplift-qini", ("uplift-segment-compare", 6, True)),
            html.Div(
                [
                    section_header("Targeting policy"),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Div(
                                        "Combine the uplift ranking with send cost and gross margin "
                                        "to find the percentage of the list worth mailing. The optimal "
                                        "point maximises net incremental contribution.",
                                        className="small text-muted mb-3",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                labeled_input_group(
                                                    "Send cost per email",
                                                    [
                                                        dbc.InputGroupText("$", className="dashboard-input-group-text"),
                                                        dbc.Input(
                                                            id="policy-cost",
                                                            type="number",
                                                            min=0,
                                                            step=0.01,
                                                            value=0.05,
                                                            debounce=True,
                                                            className="dashboard-input",
                                                            persistence=True,
                                                            persistence_type="session",
                                                        ),
                                                    ],
                                                ),
                                                md=12,
                                            ),
                                            dbc.Col(
                                                labeled_input_group(
                                                    "Gross margin on incremental spend",
                                                    [
                                                        dbc.Input(
                                                            id="policy-margin",
                                                            type="number",
                                                            min=0,
                                                            max=1,
                                                            step=0.05,
                                                            value=0.40,
                                                            debounce=True,
                                                            className="dashboard-input",
                                                            persistence=True,
                                                            persistence_type="session",
                                                        ),
                                                        dbc.InputGroupText("(0-1)", className="dashboard-input-group-text"),
                                                    ],
                                                ),
                                                md=12,
                                            ),
                                        ],
                                        className="g-2",
                                    ),
                                ],
                                md=4,
                            ),
                            dbc.Col(
                                loading(html.Div(id="policy-kpi-cards", className="kpi-stack")),
                                md=8,
                            ),
                        ],
                        className="g-3 mb-3",
                    ),
                    graph("policy-curve"),
                    html.Div(id="policy-verdict"),
                ],
                className="mb-4",
            ),
            methodology_collapse(
                "tab4",
                [
                    html.P(
                        "Uplift modelling estimates a customer-by-customer treatment effect: the "
                        "lift in spend the model expects for each individual if they receive the "
                        "email, based on what we know about them. That estimate is called CATE."
                    ),
                    html.P(
                        "The S-Learner fits a single model with treatment as an input feature, "
                        "and reads off the difference between predictions with treatment on and "
                        "off. The X-Learner (Künzel et al. 2019) fits a separate outcome "
                        "model per arm, imputes the missing counterfactual outcome for each "
                        "customer, regresses those imputed treatment effects on covariates and "
                        "combines the two arms with a propensity-weighted average. X-Learner "
                        "handles arm imbalance better than S-Learner when treatment groups "
                        "are uneven. Here the groups are balanced, so we're using it as a stronger default "
                        "view as opposed to a final answer."
                    ),
                    html.P(
                        "Both models use 5-fold cross-fitting and the same tuned random "
                        "forest base learner (depth 8, leaf size 50, feature subsampling). The "
                        "tuning is deliberate: spend is ~99% zeros with a right-skewed positive "
                        "tail, and the default forest will memorise individual converters. "
                        "Each customer's CATE comes from a model that never saw them during "
                        "training, which keeps the estimates honest rather than overfit to "
                        "in-sample noise."
                    ),
                    html.P(
                        "The Qini curve shows how much extra revenue you'd capture if you mailed "
                        "customers in order of predicted uplift, starting with the most promising. "
                        "The dashed diagonal is what random targeting would deliver. The area "
                        "above it is the value the ranking adds. The far left of the curve (the "
                        "first percent or two of the list) rests on very few treated customers, "
                        "so spikes there are noise rather than signal. The targeting-policy "
                        "optimiser below ignores shares under 2% for the same reason. The decile "
                        "chart bucketed view carries error bars from a within-decile bootstrap. "
                        "A model with real targeting value should show higher lift in the top "
                        "deciles than the bottom."
                    ),
                    html.P(
                        [
                            html.Strong("Is the AUUC real? "),
                            "The Qini chart shows a permutation p-value alongside the AUUC. "
                            "It comes from shuffling the treatment labels 500 times while keeping "
                            "the predicted ranking fixed, then asking how often the random AUUC "
                            "reaches the observed one. Small p means the ranking is picking out "
                            "real responders rather than coincidence. This is a refit-free test, "
                            "so it conditions on the trained model. It answers "
                            "\"does this ranking work?\" rather than \"would a fresh model on "
                            "different data also find a ranking?\""
                        ]
                    ),
                    html.P(
                        "Feature importance is recomputed for the selected learner by shuffling "
                        "one feature at a time and checking how much that learner's predicted "
                        "CATE moves. A feature that drives real heterogeneity shifts the "
                        "prediction a lot. A feature that doesn't matter barely moves it. It's "
                        "also harder to fool than the default random-forest importance, which "
                        "inflates continuous features and high-cardinality categoricals."
                    ),
                    html.P(
                        "A known quirk of the S-Learner: when outcome noise is large relative to "
                        "the treatment signal, as it is here, the model shrinks individual CATE "
                        "estimates towards zero. That's why the S-Learner's average CATE is "
                        "typically smaller in magnitude than the X-Learner's."
                    ),
                    html.P(
                        "Because assignment was random, these models focus on ranking and on which "
                        "kinds of customers respond more, rather than on correcting for "
                        "confounding. Individual CATE estimates carry more uncertainty than the "
                        "overall average, so treat them as directional rather than precise "
                        "per-customer numbers."
                    ),
                ],
            ),
        ],
        fluid=True,
        className="py-5"
    )
