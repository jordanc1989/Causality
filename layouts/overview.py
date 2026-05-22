
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

    # Minimum detectable effect at 80% power, two-sided α = 0.05. Uses pooled SE
    # of the difference of means: z_{α/2} + z_β = 1.96 + 0.84 = 2.80. The MDE
    # is the smallest per-recipient lift the experiment could reliably detect.
    # Worth surfacing because it sets the floor for how to interpret a null
    # result and contextualises the observed effects.
    def _mde_80(treated, control):
        se = float(np.sqrt(
            treated.var(ddof=1) / len(treated)
            + control.var(ddof=1) / len(control)
        ))
        return 2.80 * se

    mde_mens = _mde_80(spend_mens, spend_control)
    mde_womens = _mde_80(spend_womens, spend_control)
    mens_sig = mens_lo > 0
    wom_sig = wom_lo > 0
    proj_mens = lift_mens * n_mens
    proj_womens = lift_womens * n_womens
    proj_mens_lo, proj_mens_hi = mens_lo * n_mens, mens_hi * n_mens
    proj_wom_lo, proj_wom_hi = wom_lo * n_womens, wom_hi * n_womens

    if mens_sig and wom_sig:
        headline = "Both campaigns produced a lift in spend that's very unlikely to be chance."
        headline_color = SUCCESS
    elif wom_sig and not mens_sig:
        headline = (
            "The Women's campaign produced a clear lift in spend. "
            "The Men's result is too close to call."
        )
        headline_color = SUCCESS
    elif mens_sig and not wom_sig:
        headline = (
            "The Men's campaign produced a clear lift in spend. "
            "The Women's result is too close to call."
        )
        headline_color = SUCCESS
    else:
        headline = (
            "Neither campaign's effect is large enough to separate from chance on the raw "
            "averages. The later tabs use stronger methods to check the same question."
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
            "Per-recipient lift multiplied by the number of recipients in this arm. "
            "The range is a 95% confidence interval from a 2,000-resample bootstrap."
        )
        return dbc.Col(
            [
            html.Div(
                [
                    html.Div(label, style={"fontSize": "0.72rem", "fontFamily": SERIF,
                                           "fontWeight": "600", "letterSpacing": "0.01em",
                                           "color": MUTED, "marginBottom": "0.5rem"}),
                    html.Div("Projected total lift",
                             style={"fontSize": "0.72rem", "fontFamily": SERIF,
                                    "fontWeight": "500", "letterSpacing": "0.01em",
                                    "color": MUTED, "marginBottom": "0.3rem"}),
                    html.Div(f"${proj:,.0f}",
                             style={"fontSize": "1.7rem", "fontFamily": SERIF,
                                    "fontWeight": "700", "letterSpacing": "-0.02em",
                                    "color": color if sig else MUTED,
                                    "lineHeight": "1", "marginBottom": "0.35rem"}),
                    html.Div(
                        [
                            html.Span("95% CI  ", style={"color": MUTED}),
                            html.Span(
                                f"${proj_lo:,.0f} – ${proj_hi:,.0f}",
                                id=ci_id,
                                className="info-term",
                                style={"color": MUTED},
                            ),
                        ],
                        style={"fontSize": "0.7rem", "fontFamily": MONO},
                    ),
                ],
                style={"borderTop": f"2px solid {color if sig else BORDER_STRONG}",
                       "paddingTop": "0.7rem"},
            ),
            dbc.Tooltip(tooltip_text, target=ci_id, placement="bottom"),
            ],
            md=4,
        )

    return dbc.Container(
        [
            # Lede — the finding stated up front, like the standfirst of an article.
            html.Div(
                [
                    html.Div(
                        "Headline finding",
                        style={"fontFamily": MONO, "fontSize": "0.74rem",
                               "textTransform": "uppercase", "letterSpacing": "0.08em",
                               "color": MUTED, "marginBottom": "0.5rem"},
                    ),
                    html.H2(
                        headline,
                        style={"fontFamily": SERIF, "fontWeight": "600",
                               "fontSize": "1.7rem", "lineHeight": "1.3",
                               "maxWidth": "52rem", "color": TEXT,
                               "borderLeft": f"3px solid {headline_color}",
                               "paddingLeft": "1rem", "marginBottom": "1.5rem"},
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
                        className="g-4 mb-2",
                    ),
                    html.P(
                        "Per-recipient lift scaled to the full arm. Ranges are 95% bootstrap "
                        "intervals (2,000 resamples). The later sections re-test the same lift "
                        "with matching, Bayesian and uplift methods.",
                        className="text-muted small mb-0",
                    ),
                ],
                style={"marginBottom": "2.5rem"},
            ),
            section_header("By campaign arm"),
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
                        [
                            section_header("About the data"),
                            html.P(
                                "The Hillstrom dataset is a real email experiment across 64k US "
                                "retail customers: a third got a Men's catalogue email, a third a "
                                "Women's, a third nothing. Spend was recorded over the following "
                                "two weeks, alongside a handful of customer attributes.",
                                className="small text-muted",
                            ),
                            html.P(
                                "The question is whether the email itself caused extra spend, and "
                                "where. Later sections add the confidence around that, which "
                                "customers respond most, and whether different methods agree.",
                                className="small text-muted mb-0",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        html.P(
                            [
                                html.Strong("Detection sensitivity. "),
                                f"With ~{n_mens // 1000}k recipients per arm, the experiment can "
                                "reliably detect a per-recipient lift of ",
                                html.Strong(f"${mde_mens:.2f}"),
                                " (Men's) and ",
                                html.Strong(f"${mde_womens:.2f}"),
                                f" (Women's) at 80% power, two-sided α = 5%. The observed lifts "
                                f"(${lift_mens:.2f} and ${lift_womens:.2f}) clear those floors — "
                                "which is why the intervals above exclude zero.",
                            ],
                            className="small text-muted",
                        ),
                        md=6,
                    ),
                ],
                className="mb-4 g-4 align-items-start",
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

