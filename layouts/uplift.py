
from dash import dcc, html
import dash_bootstrap_components as dbc
from dashboard.theme import *
from layouts.components import methodology_collapse

def tab4_layout():
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Campaign arm:", className="small text-muted"),
                            dbc.RadioItems(
                                id="uplift-arm-selector",
                                options=[
                                    {"label": "Men's Email", "value": "mens"},
                                    {"label": "Women's Email", "value": "womens"}
                                ],
                                value="mens",
                                inline=True,
                                className="dashboard-radio-group mb-2"
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            html.Label("Model:", className="small text-muted"),
                            dbc.RadioItems(
                                id="uplift-model-selector",
                                options=[
                                    {"label": "T-Learner", "value": "t"},
                                    {"label": "S-Learner", "value": "s"}
                                ],
                                value="t",
                                inline=True,
                                className="dashboard-radio-group mb-2"
                            ),
                        ],
                        md=6
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(html.Div(id="uplift-kpi-cards"), md=4),
                    dbc.Col(dcc.Graph(id="uplift-cate-hist", config=GRAPH_CONFIG), md=8)
                ],
                className="mb-3"
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="uplift-feat-imp", config=GRAPH_CONFIG), md=6),
                    dbc.Col(dcc.Graph(id="uplift-decile-chart", config=GRAPH_CONFIG), md=6)
                ],
                className="mb-3"
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="uplift-qini", config=GRAPH_CONFIG), md=6),
                    dbc.Col(dcc.Graph(id="uplift-segment-compare", config=GRAPH_CONFIG), md=6)
                ],
                className="mb-3"
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
                        "difference between predictions with treatment on and off. The two often "
                        "disagree on individual customers, which is itself a useful signal."
                    ),
                    html.P(
                        "Both models use 5-fold cross-fitting. Each customer's CATE comes from a "
                        "model that never saw them during training, which keeps the estimates "
                        "honest rather than overfit to in-sample noise."
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

