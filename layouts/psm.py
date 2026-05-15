
from dash import dcc, html
import dash_bootstrap_components as dbc
from dashboard.theme import *
from layouts.components import methodology_collapse
from content.methodology import psm_intro_copy

def tab2_layout():
    return dbc.Container(
        [
            dbc.Card(
                dbc.CardBody(
                    psm_intro_copy()
                ),
                style={**CARD_STYLE, "borderLeft": f"3px solid {ACCENT}"},
                className="dashboard-card mb-3"
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Campaign arm:", className="small text-muted"),
                            dbc.RadioItems(
                                id="psm-arm-selector",
                                options=[
                                    {
                                        "label": "Men's Email vs Control",
                                        "value": "mens"
                                    },
                                    {
                                        "label": "Women's Email vs Control",
                                        "value": "womens"
                                    }
                                ],
                                value="mens",
                                inline=True,
                                className="dashboard-radio-group mb-3"
                            ),
                        ]
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(html.Div(id="psm-kpi-cards"), md=4),
                    dbc.Col(dcc.Graph(id="psm-ps-dist", config=GRAPH_CONFIG), md=8)
                ],
                className="mb-3"
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="psm-love-plot", config=GRAPH_CONFIG), md=7),
                    dbc.Col(dcc.Graph(id="psm-stats-chart", config=GRAPH_CONFIG), md=5)
                ],
                className="mb-3"
            ),
            methodology_collapse(
                "tab2",
                [
                    html.P(
                        [
                            html.Strong("Statistical machinery (matching): "),
                            "A logistic regression uses nine pre-treatment covariates—recency, history, men's and "
                            "women's catalogue interest flags, two zip-type indicators (suburban/rural vs urban), "
                            "two channel indicators (web / multichannel vs phone), and the newbie flag—to estimate "
                            "each customer's ",
                            html.Em("propensity score"),
                            ": the modelled probability they would appear in the email arm given those traits.",
                            " Matching is ",
                            html.Strong("1:1 nearest neighbour in logit(propensity-score) metric"),
                            " with replacement.",
                            " A ",
                            html.Strong("caliper"),
                            " trims pairs where the ",
                            html.Em("absolute gap in logit(propensity)"),
                            " exceeds ",
                            html.Strong("0.2 times the pooled standard deviation"),
                            " of logit(PS) across all rows in that arm-vs-control subset.",
                            " Unmatched treated customers are omitted from ATT and balance diagnostics on the ",
                            "\"after\" cohort."
                        ]
                    ),
                    html.P(
                        "Because Hillstrom is randomised, treatment assignment is exogenous by design and "
                        "the primary causal estimand is an ITT/ATE contrast (see Overview bootstrap and Tab 3 "
                        "population-weighted OLS). "
                        "PSM illustrates observational workflows: overlap tightening, pruning, and covariance "
                        "balance on the matched cohort—not the project's headline estimator."
                    ),
                    html.P(
                        "Uncertainty is a stratified re-fit/re-match bootstrap (200 replicates): each "
                        "replicate resamples the treated and control pools "
                        "separately at their observed sizes (preserving the arm ratio), re-estimates "
                        "propensity, reapplies NN + caliper, and recomputes ATT on surviving pairs. "
                        "Interpret as a heuristic sensitivity envelope, not textbook nearest-neighbour CIs."
                    ),
                    html.P(
                        "The Love plot contrasts all treated covariates to pooled controls ",
                        "(before pairing) versus caliper-kept treated units to their ",
                        "matched controls."
                    ),
                ],
            ),
        ],
        fluid=True,
        className="py-5"
    )

