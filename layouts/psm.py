
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
                            html.Strong("How the matching works. "),
                            "A logistic regression on nine pre-treatment attributes (recency, history, "
                            "the two catalogue-interest flags, two zip-type flags, two channel flags, and "
                            "the new-customer flag) produces a ",
                            html.Em("propensity score"),
                            " for each customer. That score is the modelled probability that the customer "
                            "would end up in the email arm given those attributes. We then pair each email "
                            "recipient with the control customer whose score is closest to theirs. A caliper "
                            "trims any pair where the gap on the logit of the propensity score is more than "
                            "0.2 standard deviations. Unmatched recipients are dropped from the post-match "
                            "comparison rather than forced into a poor pairing."
                        ]
                    ),
                    html.P(
                        "Because the original assignment was random, the trustworthy headline estimates "
                        "live on the Overview and the OLS tab. PSM is included here as a sanity check that "
                        "would also work in settings where there was no random assignment to lean on. The "
                        "value of this tab is the diagnostic view (overlap, balance, matched cohort), "
                        "not the point estimate."
                    ),
                    html.P(
                        "The confidence band on the bar chart comes from redoing the propensity fit and "
                        "the matching 200 times on resampled data. Each replicate redraws treated and "
                        "control separately at their observed sizes so the arm ratio stays fixed. Read "
                        "the band as a stress test on the matching choices, not as a textbook nearest-"
                        "neighbour confidence interval."
                    ),
                    html.P(
                        "The Love plot compares every attribute in the email arm against the control arm "
                        "twice. Once across the full groups (before pairing) and once across just the "
                        "matched pairs (after). Closer to zero means more comparable groups."
                    ),
                ],
            ),
        ],
        fluid=True,
        className="py-5"
    )

