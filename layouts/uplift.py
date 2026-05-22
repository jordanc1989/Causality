
from dash import html
import dash_bootstrap_components as dbc
from dashboard.theme import *
from layouts.components import (
    graph, graph_row_ids, kpi_graph_row, methodology_collapse,
    labeled_radio, labeled_input_group,
)

def tab4_layout():
    return dbc.Container(
        [
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
                                {"label": "T-Learner", "value": "t"},
                                {"label": "S-Learner", "value": "s"},
                                {"label": "X-Learner", "value": "x"},
                            ],
                            "x",
                        ),
                        md=6,
                    ),
                ]
            ),
            kpi_graph_row("uplift-kpi-cards", "uplift-cate-hist"),
            graph_row_ids("uplift-feat-imp", "uplift-decile-chart", locked=True),
            graph_row_ids("uplift-qini", ("uplift-segment-compare", 6, True)),
            dbc.Card(
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Div(
                                            "Targeting policy",
                                            style={
                                                **SECTION_HEADER_STYLE,
                                                "borderBottom": "none",
                                                "paddingBottom": 0,
                                                "marginBottom": "0.3rem",
                                                "display": "block",
                                            },
                                        ),
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
                                                            ),
                                                        ],
                                                    ),
                                                    md=6,
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
                                                            ),
                                                            dbc.InputGroupText("(0-1)", className="dashboard-input-group-text"),
                                                        ],
                                                    ),
                                                    md=6,
                                                ),
                                            ],
                                            className="g-2",
                                        ),
                                    ],
                                    md=4,
                                ),
                                dbc.Col(html.Div(id="policy-kpi-cards", className="kpi-stack"), md=8),
                            ],
                            className="g-3 mb-3",
                        ),
                        graph("policy-curve"),
                    ]
                ),
                style=CARD_STYLE,
                className="dashboard-card mb-4",
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
                        "The T-Learner fits two separate models, one on the email group and one "
                        "on control, and then subtracts their predictions. The S-Learner fits a "
                        "single model with treatment as an input feature, and reads off the "
                        "difference between predictions with treatment on and off. The X-Learner "
                        "(Künzel et al. 2019) extends the T-Learner: it imputes the missing "
                        "counterfactual outcome for each customer, regresses those imputed "
                        "treatment effects on covariates, and combines the two arms with a "
                        "propensity-weighted average. X-Learner handles arm imbalance better "
                        "than T- or S-Learners and is the recommended default in modern uplift "
                        "practice. The three often disagree on individual customers, which is "
                        "itself a useful signal."
                    ),
                    html.P(
                        "All three models use 5-fold cross-fitting and the same tuned random "
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
                        "above it is the value the ranking adds. The decile chart bucketed view "
                        "carries error bars from a within-decile bootstrap. A model with real "
                        "targeting value should show higher lift in the top deciles than the bottom."
                    ),
                    html.P(
                        [
                            html.Strong("Is the AUUC real? "),
                            "The Qini chart shows a permutation p-value alongside the AUUC. "
                            "It comes from shuffling the treatment labels 500 times while keeping "
                            "the predicted ranking fixed, then asking how often the random AUUC "
                            "reaches the observed one. Small p means the ranking is picking out "
                            "real responders rather than coincidence. This is a refit-free test, "
                            "so it conditions on the trained model - it's the right null for "
                            "\"does this ranking work?\" rather than \"would a fresh model on "
                            "different data also find a ranking?\""
                        ]
                    ),
                    html.P(
                        [
                            html.Strong("Decile CI caveat. "),
                            "The error bars come from resampling treated and control within each "
                            "fixed decile. That captures within-decile sampling noise but ",
                            html.Em("not"),
                            " uncertainty in the decile boundaries themselves - those boundaries "
                            "depend on the predicted CATE, which is itself noisy. A fully honest "
                            "CI would bootstrap the entire fit → rank → decile loop. Read these "
                            "intervals as a lower bound on the true uncertainty."
                        ]
                    ),
                    html.P(
                        "Feature importance is measured by shuffling one feature at a time and "
                        "checking how much the predicted CATE moves. A feature that drives real "
                        "heterogeneity shifts the prediction a lot. A feature that doesn't matter "
                        "barely moves it. This is more honest than the default random-forest "
                        "importance, which gets fooled by continuous features and high-cardinality "
                        "categoricals."
                    ),
                    html.P(
                        "A known quirk of the S-Learner: when outcome noise is large relative to "
                        "the treatment signal, as it is here, the model shrinks individual CATE "
                        "estimates toward zero. That's why the S-Learner's average CATE is "
                        "typically smaller in magnitude than the T-Learner's."
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
