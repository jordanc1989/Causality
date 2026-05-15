
import numpy as np
from dash import dcc, html
import dash_bootstrap_components as dbc
from dashboard.theme import *
from dashboard.data import DF
from layouts.components import section_header, segment_overview_card
from figures.overview import _fig_spend_box, _fig_covariate_balance
import causal_utils as cu

def tab1_layout():
    seg_counts = DF["segment"].value_counts()
    n_mens = seg_counts.get("Mens E-Mail", 0)
    n_womens = seg_counts.get("Womens E-Mail", 0)
    n_control = seg_counts.get("No E-Mail", 0)

    conv_mens = DF[DF["segment"] == "Mens E-Mail"]["conversion"].mean() * 100
    conv_womens = DF[DF["segment"] == "Womens E-Mail"]["conversion"].mean() * 100
    conv_control = DF[DF["segment"] == "No E-Mail"]["conversion"].mean() * 100

    spend_mens = DF[DF["segment"] == "Mens E-Mail"]["spend"]
    spend_womens = DF[DF["segment"] == "Womens E-Mail"]["spend"]
    spend_control = DF[DF["segment"] == "No E-Mail"]["spend"]
    avg_mens, avg_womens, avg_control = spend_mens.mean(), spend_womens.mean(), spend_control.mean()
    lift_mens = avg_mens - avg_control
    lift_womens = avg_womens - avg_control

    def _ci95(a, b, n_boot=2000, seed=cu.RANDOM_SEED):
        # Simple difference-of-means bootstrap — not the causal bootstrap used in PSM.
        # Valid here because the randomised design makes the raw difference unbiased.
        rng = np.random.default_rng(seed)
        diffs = np.array([
            rng.choice(a, size=len(a), replace=True).mean() -
            rng.choice(b, size=len(b), replace=True).mean()
            for _ in range(n_boot)
        ])
        return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

    mens_lo, mens_hi = _ci95(spend_mens, spend_control)
    wom_lo, wom_hi = _ci95(spend_womens, spend_control)
    mens_sig = mens_lo > 0
    wom_sig = wom_lo > 0
    proj_mens = lift_mens * n_mens
    proj_womens = lift_womens * n_womens
    proj_mens_lo, proj_mens_hi = mens_lo * n_mens, mens_hi * n_mens
    proj_wom_lo, proj_wom_hi = wom_lo * n_womens, wom_hi * n_womens

    if mens_sig and wom_sig:
        headline = "Both campaigns drove statistically significant incremental revenue."
        headline_color = SUCCESS
    elif wom_sig and not mens_sig:
        headline = (
            "Women's campaign drove significant lift. "
            "Men's result is inconclusive at 95% confidence."
        )
        headline_color = SUCCESS
    elif mens_sig and not wom_sig:
        headline = (
            "Men's campaign drove significant lift. "
            "Women's result is inconclusive at 95% confidence."
        )
        headline_color = SUCCESS
    else:
        headline = (
            "Neither campaign's effect is distinguishable from zero at 95% confidence "
            "in the raw difference-of-means — see subsequent tabs for precision-adjusted "
            "and probabilistic readings."
        )
        headline_color = WARNING

    rev_pct_mens = (lift_mens / avg_control * 100) if avg_control else None
    rev_pct_womens = (lift_womens / avg_control * 100) if avg_control else None
    conv_pct_mens = ((conv_mens - conv_control) / conv_control * 100) if conv_control else None
    conv_pct_womens = ((conv_womens - conv_control) / conv_control * 100) if conv_control else None

    def _hl_col(label, color, sig, proj, proj_lo, proj_hi):
        """Projected total-revenue extrapolation for one arm, with bootstrap CI band."""
        ci_id = f"hl-ci-{label.lower().replace(' ', '-').replace(chr(39), '')}"
        tooltip_text = (
            "Projected total incremental revenue across the campaign cohort: per-recipient "
            "lift × number of recipients. Range is a 2,000-resample percentile bootstrap "
            "on the per-recipient difference of means."
        )
        return dbc.Col(
            [
            html.Div(
                [
                    html.Div(label, style={"fontSize": "0.72rem", "fontFamily": "Ubuntu, sans-serif",
                                           "fontWeight": "600", "letterSpacing": "0.01em",
                                           "color": MUTED, "marginBottom": "0.5rem"}),
                    html.Div("Projected total lift",
                             style={"fontSize": "0.72rem", "fontFamily": "Ubuntu, sans-serif",
                                    "fontWeight": "500", "letterSpacing": "0.01em",
                                    "color": MUTED, "marginBottom": "0.3rem"}),
                    html.Div(f"${proj:,.0f}",
                             style={"fontSize": "1.7rem", "fontFamily": "Ubuntu, sans-serif",
                                    "fontWeight": "700", "letterSpacing": "-0.02em",
                                    "color": color if sig else MUTED,
                                    "lineHeight": "1", "marginBottom": "0.35rem"}),
                    html.Div(
                        [
                            html.Span("95% CI  ", style={"color": MUTED}),
                            html.Span(
                                f"${proj_lo:,.0f} – ${proj_hi:,.0f}",
                                id=ci_id,
                                style={"color": MUTED, "cursor": "help"},
                            ),
                        ],
                        style={"fontSize": "0.7rem", "fontFamily": "Ubuntu Mono, monospace"},
                    ),
                ],
                style={"borderLeft": f"3px solid {color if sig else BORDER}",
                       "paddingLeft": "0.85rem"},
            ),
            dbc.Tooltip(tooltip_text, target=ci_id, placement="bottom"),
            ],
            md=4,
        )

    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        segment_overview_card(
                            name="Men's Email",
                            color=MENS_COLOUR,
                            n=n_mens,
                            revenue_per=avg_mens,
                            conversion_rate=conv_mens,
                            rev_lift=lift_mens,
                            rev_pct=rev_pct_mens,
                            rev_sig=mens_sig,
                            conv_lift_pp=conv_mens - conv_control,
                            conv_pct=conv_pct_mens,
                        ),
                        md=4,
                        className="mb-3",
                    ),
                    dbc.Col(
                        segment_overview_card(
                            name="Women's Email",
                            color=WOMENS_COLOUR,
                            n=n_womens,
                            revenue_per=avg_womens,
                            conversion_rate=conv_womens,
                            rev_lift=lift_womens,
                            rev_pct=rev_pct_womens,
                            rev_sig=wom_sig,
                            conv_lift_pp=conv_womens - conv_control,
                            conv_pct=conv_pct_womens,
                        ),
                        md=4,
                        className="mb-3",
                    ),
                    dbc.Col(
                        segment_overview_card(
                            name="Control",
                            color=CTRL_COLOUR,
                            n=n_control,
                            revenue_per=avg_control,
                            conversion_rate=conv_control,
                            is_control=True,
                        ),
                        md=4,
                        className="mb-3",
                    ),
                ],
                className="mb-4 g-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Headline Finding"),
                                dbc.CardBody(
                                    [
                                        html.P(
                                            headline,
                                            style={"fontSize": "0.82rem", "color": MUTED, "marginBottom": "1.25rem"},
                                        ),
                                        dbc.Row(
                                            [
                                                _hl_col(
                                                    "Men's Email", MENS_COLOUR, mens_sig,
                                                    proj_mens, proj_mens_lo, proj_mens_hi,
                                                ),
                                                _hl_col(
                                                    "Women's Email", WOMENS_COLOUR, wom_sig,
                                                    proj_womens, proj_wom_lo, proj_wom_hi,
                                                ),
                                            ],
                                            className="g-3 mb-3",
                                        ),
                                        html.P(
                                            "Per-recipient lift and SIG/n.s. status are shown on the segment "
                                            "cards above, this card extrapolates that lift to the full campaign "
                                            "cohort. Confidence intervals use a percentile bootstrap (2k resamples) "
                                            "on the difference of means. Subsequent tabs test the per-recipient "
                                            "effect with matching, Bayesian and uplift methods.",
                                            className="text-muted small mb-0",
                                        ),
                                    ]
                                ),
                            ],
                            style={**CARD_STYLE, "borderLeft": f"3px solid {headline_color}"},
                            className="dashboard-card h-100"
                        ),
                        md=6,
                        className="mb-3",
                    ),
                    dbc.Col(
                        [
                            section_header("About this Dataset"),
                            html.P(
                                "The Hillstrom MineThatData dataset captures a randomised email marketing "
                                "experiment across 64k US retail customers. Two treatment groups received "
                                "targeted email campaigns (Men's or Women's catalogue), while the control "
                                "group received nothing. The groups were split around 33% each, and the "
                                "experiment ran for 30 days. The dataset includes customer attributes "
                                "(recency, frequency, monetary value, channel) and outcomes (conversion, spend).",
                                className="small text-muted"
                            ),
                            html.P(
                                "The causal question: does receiving an email cause customers to spend more? "
                                "Even in a randomised experiment, causal analysis adds value by quantifying "
                                "uncertainty, identifying which customers respond most (HTE), and stress-testing "
                                "results across different methodologies.",
                                className="small text-muted"
                            ),
                        ],
                        md=6,
                        className="mb-3",
                    ),
                ],
                className="mb-4 g-3 align-items-start"
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            section_header("Spend Distribution by Segment (among spenders)"),
                            dcc.Graph(id="tab1-box", figure=_fig_spend_box(), config=GRAPH_CONFIG)
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            section_header("Did the randomisation work?"),
                            html.P(
                                [
                                    "Each dot compares the treatment and control groups on a customer attribute "
                                    "(recency, spend history, channel, etc). ",
                                    html.Strong("Dots inside the dashed band = groups are balanced"),
                                    ", meaning any difference in outcomes can be attributed to the email itself "
                                    "rather than pre-existing differences between customers. "
                                    "A randomised experiment should produce this pattern."
                                ],
                                className="small text-muted mb-2"
                            ),
                            dcc.Graph(
                                id="tab1-balance", figure=_fig_covariate_balance(), config=GRAPH_CONFIG
                            ),
                        ],
                        md=6,
                    ),
                ],
                className="mb-4 g-3 align-items-start"
            ),
        ],
        fluid=True,
        className="py-5"
    )

