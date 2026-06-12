
from dash import html
import dash_bootstrap_components as dbc
from dashboard.theme import *
from layouts.components import (
    graph_row_ids, kpi_graph_row, methodology_collapse, labeled_radio, spec_strip,
)
from content.methodology import psm_intro_copy

def tab2_layout(**_kwargs):
    return dbc.Container(
        [
            spec_strip(
                "Logistic propensity",
                "9 covariates",
                "0.2 SD caliper",
                "200-iter bootstrap",
            ),
            html.Div(
                psm_intro_copy(),
                className="psm-intro mb-4",
                style={
                    "borderLeft": f"2px solid {BORDER_STRONG}",
                    "paddingLeft": "1rem",
                    "color": MUTED,
                },
            ),
            dbc.Row(
                [
                    dbc.Col(
                        labeled_radio(
                            "Campaign group:",
                            "psm-arm-selector",
                            [
                                {"label": "Men's Email vs Control", "value": "mens"},
                                {"label": "Women's Email vs Control", "value": "womens"},
                            ],
                            "mens",
                            className="dashboard-radio-group mb-3",
                        )
                    ),
                ]
            ),
            kpi_graph_row("psm-kpi-cards", "psm-ps-dist"),
            graph_row_ids(("psm-love-plot", 7), ("psm-stats-chart", 5)),
            methodology_collapse(
                "tab2",
                [
                    html.P(
                        [
                            html.Strong("Pedagogical framing. "),
                            "On a randomised experiment, propensity matching isn't a separate "
                            "estimator. It produces a noisier version of the simple "
                            "difference-in-means and discards observations along the way. "
                            "This tab is included to show ",
                            html.Em("how the workflow would look on observational data"),
                            ", where you're unable to use random assignment."
                        ]
                    ),
                    html.P(
                        [
                            html.Strong("How the matching works. "),
                            "A logistic regression on nine pre-treatment attributes (recency, history, "
                            "the two catalogue-interest flags, two postcode-type flags, two channel flags, and "
                            "the new-customer flag) produces a ",
                            html.Em("propensity score"),
                            " for each customer. That score is the modelled probability that the customer "
                            "would end up in the email group given those attributes. We then pair each email "
                            "recipient with the control customer whose score is closest to theirs. A caliper "
                            "trims any pair where the gap on the logit of the propensity score is more than "
                            "0.2 standard deviations. Unmatched recipients are dropped from the post-match "
                            "comparison rather than forced into a poor pairing."
                        ]
                    ),
                    html.P(
                        "The main ATT band uses a simple matched-pair standard error for the retained "
                        "pairs. Matching is done with replacement, so one control customer can serve "
                        "as the lookalike for several recipients. The matched-pair SE treats pairs as "
                        "independent, which makes it slightly optimistic. The bootstrap band beside it "
                        "refits the propensity model and rematches 200 times on resampled data. Read "
                        "that second band as a stress test on the matching choices, not as an exact "
                        "nearest-neighbour matching interval."
                    ),
                    html.P(
                        [
                            "The Love plot compares every attribute in the email group against the control group "
                            "twice. Once across the full groups (before pairing) and once across just the "
                            "matched pairs (after). In an observational study, a big drop from before to "
                            "after is the headline result. On this randomised dataset, the ",
                            html.Em("before"),
                            " plot is already near zero, so the randomisation is working as expected.",
                        ]
                    ),
                ],
            ),
        ],
        fluid=True,
        className="py-5"
    )
