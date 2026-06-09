
from dash import html
import dash_bootstrap_components as dbc
from dashboard.theme import *
from layouts.components import graph_row_ids, section_col

def tab6_layout(**_kwargs):
    return dbc.Container(
        [
            dbc.Row(
                [
                    section_col(
                        "All methods summary",
                        html.P(
                            [
                                "Each row is one method's estimate of the average lift in spend "
                                "per recipient, with its confidence range. The Overview, Bayesian "
                                "and OLS rows are the headline numbers because they "
                                "use the random assignment directly. PSM is left out of this comparison "
                                "because it is included as a teaching example, not as a headline method "
                                "for this randomised dataset.",
                            ],
                            className="text-muted small mb-2",
                        ),
                        html.Div(id="comparison-table"),
                    ),
                ],
                className="mb-4",
            ),
            graph_row_ids(
                "forest-plot-mens", "forest-plot-womens",
                className="mb-4", locked=True,
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    html.Span(
                                        "Agreement threshold",
                                        id="comparison-noise-info",
                                        className="small text-muted info-term",
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
                                                "The PSM tab pairs each email recipient with a lookalike "
                                                "control customer. That's useful as a demo of what you'd do "
                                                "without random assignment, but this dataset was randomised, "
                                                "so PSM is intentionally omitted from the headline comparison."
                                            ),
                                        ],
                                        title="Why PSM is omitted"
                                    ),
                                    dbc.AccordionItem(
                                        [
                                            html.P(
                                                "Fits a probabilistic model to the spend data and "
                                                "produces a full posterior over the per-customer effect. "
                                                "Best for talking about uncertainty (HDI, P(effect > 0)) "
                                                "and for setting a threshold of practical significance "
                                                "via the ROPE control on that tab."
                                            ),
                                        ],
                                        title="Bayesian A/B"
                                    ),
                                    dbc.AccordionItem(
                                        [
                                            html.P(
                                                "Estimates the lift for each individual customer. "
                                                "Best for ranking, so you can "
                                                "target the most responsive shoppers. Individual "
                                                "estimates are noisier than the overall average, so "
                                                "treat them as directional."
                                            ),
                                        ],
                                        title="Uplift (T-, S-, X-Learner)"
                                    ),
                                    dbc.AccordionItem(
                                        [
                                            html.P(
                                                "Linear regression with interaction terms. Best for a "
                                                "clean, auditable read of the average effect and how "
                                                "it varies by customer type. Assumes the relationship "
                                                "between attributes and spend is roughly linear."
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
