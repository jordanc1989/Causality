
from dash import dcc, html
import dash_bootstrap_components as dbc
from dashboard.theme import *
from layouts.components import section_header, methodology_collapse

def tab5_layout():
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            section_header(
                                "OLS Coefficient Plot: Treatment Effects & Interactions"
                            ),
                            dcc.Graph(id="ols-coef-plot", config=GRAPH_CONFIG)
                        ]
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            section_header("Marginal Effects by Subgroup"),
                            html.Div(id="ols-marginal-table")
                        ]
                    ),
                ],
                className="mb-3"
            ),
            dbc.Row(
                [
                    dbc.Col(section_header("Subgroup Heatmap: Predicted Spend Lift"))
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="ols-heatmap-mens", config=GRAPH_CONFIG), md=6),
                    dbc.Col(dcc.Graph(id="ols-heatmap-womens", config=GRAPH_CONFIG), md=6)
                ]
            ),
            methodology_collapse(
                "tab5",
                [
                    html.P(
                        "Linear regression estimates the average effect of the email while adjusting "
                        "for a few customer attributes (recency, spend history, channel, postcode type, "
                        "and whether the customer is new). Because assignment was random to begin with, "
                        "this adjustment isn't fixing bias. It tightens the confidence range. All three "
                        "arms are fitted in one model using treatment indicators."
                    ),
                    html.P(
                        "Interaction terms let the effect of the email vary by customer type "
                        "(new vs existing, channel, postcode). They surface subgroup patterns "
                        "that back up the uplift models on the previous tab."
                    ),
                    html.P(
                        "Standard errors use the HC3 robust estimator. Spend is right-skewed and "
                        "its variance grows with the treatment mean, so the default standard "
                        "errors would be too narrow. HC3 is the standard fix for this kind of "
                        "outcome."
                    ),
                    html.P(
                        "The subgroup heatmap collapses the (customer type) by (channel) by "
                        "(postcode) view down to (customer type) by (channel), averaging over "
                        "postcode using each cell's actual customer share. An unweighted average "
                        "would give rare postcode categories too much weight."
                    ),
                    html.P(
                        "Key assumption: the relationship between attributes and spend is roughly "
                        "linear. Coefficient colouring on the chart above uses a Holm-Bonferroni "
                        "adjustment to α = 0.05, so a coefficient only lights up if it survives "
                        "the correction for testing many terms at once. Raw p-values are still in "
                        "the hover for reference but shouldn't be read as confirmatory without "
                        "that adjustment."
                    ),
                ],
            ),
        ],
        fluid=True,
        className="py-5"
    )

