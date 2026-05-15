
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
                        "Uplift modelling estimates Conditional Average Treatment Effects (CATE): the expected "
                        "causal effect for each individual customer, given their characteristics."
                    ),
                    html.P(
                        "T-Learner trains two separate models: one on treated customers & one on control, "
                        "and computes CATE as the difference in predictions. "
                        "S-Learner trains a single model with treatment as a feature and computes CATE "
                        "by differencing predictions under treatment vs non-treatment."
                    ),
                    html.P(
                        "Both models use 5-fold stratified cross-fitting (stratified on treatment, "
                        "so every fold has both arms represented). Each observation's CATE is "
                        "predicted by a model trained on the other four folds. This avoids "
                        "in-sample overfitting and gives honest out-of-sample estimates."
                    ),
                    html.P(
                        "The Qini curve uses the canonical Radcliffe (2007) definition for continuous "
                        "outcomes: at rank k, cumulative net revenue captured equals the cumulative "
                        "treated spend minus the cumulative control spend re-weighted by the "
                        "treated/control ratio. The chart overlays a random-targeting baseline "
                        "line from zero to the full-population gain, ranking quality is the "
                        "excess area above that baseline. The decile chart shows actual spend lift for "
                        "customers ranked by predicted uplift: good models show declining lift."
                    ),
                    html.P(
                        "Feature importance is reported as permutation importance on the predicted "
                        "CATE surface. For each held-out fold we permute one feature column at a "
                        "time (5 shuffle repeats), re-predict CATE from the T-Learner, and record "
                        "the mean absolute change vs the un-permuted prediction. Features that drive "
                        "heterogeneity show large CATE shifts, irrelevant features show small ones. "
                        "This is a model-agnostic, fold-honest measure that avoids the high-cardinality "
                        "/ continuous-feature bias of raw random-forest impurity importance."
                    ),
                    html.P(
                        "S-Learner with a RandomForest and 'treatment x covariate' interactions is known "
                        "to shrink CATE toward zero when outcome variance is large relative to the "
                        "treatment signal: a pattern we see here, where the T-Learner's average "
                        "CATE tends to be larger in magnitude than the S-Learner's."
                    ),
                    html.P(
                        "Because assignment is randomised, these models focus on treatment-effect heterogeneity "
                        "and policy ranking rather than confounding control. HTE estimates typically carry "
                        "more sampling uncertainty than ATE estimates and should be treated as directional "
                        "unless accompanied by explicit uncertainty intervals."
                    ),
                ],
            ),
        ],
        fluid=True,
        className="py-5"
    )

