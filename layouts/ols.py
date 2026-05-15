
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
                        "OLS regression estimates population-average treatment effects while controlling for "
                        "observed baseline covariates. In this randomised experiment, covariate adjustment "
                        "is primarily for precision. All three arms (Men, Women, Control) are estimated in "
                        "one model via treatment indicators."
                    ),
                    html.P(
                        "Interaction terms (treatment x newbie, treatment x channel, treatment x zip code) capture "
                        "treatment effect heterogeneity at the subgroup level, complementing the "
                        "non-parametric uplift models in Tab 4."
                    ),
                    html.P(
                        "Standard errors use the HC3 heteroscedasticity-robust (White) estimator. "
                        "Spend is right-skewed and its variance scales with the treatment means, so "
                        "default OLS SEs would be biased. HC3 is the recommended small-sample "
                        "correction for this kind of outcome."
                    ),
                    html.P(
                        "The subgroup heatmap collapses the 3-way (newbie x channel x zip) marginal "
                        "effects to a 2-way (newbie x channel) view by averaging over zip, weighted "
                        "by the actual customer counts in each (newbie, channel, zip) cell. An "
                        "unweighted mean would over-represent rare zip categories."
                    ),
                    html.P(
                        "Key assumption: linearity of the conditional expectation function. The model is "
                        "estimated by OLS (not IV or 2SLS). Coefficient colouring on the chart above is "
                        "thresholded against a Holm-Bonferroni-adjusted α=0.05 across all displayed terms "
                        "to control family-wise error. Raw p-values remain in the hover for inspection but "
                        "shouldn't be read as confirmatory without the multiplicity adjustment."
                    ),
                ],
            ),
        ],
        fluid=True,
        className="py-5"
    )

