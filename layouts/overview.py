
from functools import lru_cache

import numpy as np
from dash import html
import dash_bootstrap_components as dbc
from dashboard.theme import *
from dashboard.data import DF
from dashboard.format import money
from layouts.components import (
    graph, section_col, section_header, segment_overview_card,
    spec_strip, page_lede, headline_tile,
)
from figures.overview import _fig_spend_box, _fig_covariate_balance
import causal_utils as cu

def tab1_layout(**_kwargs):
    # Dash Pages calls the layout function on every request, but everything on
    # this page derives from the static cached DF. Build once and reuse so the
    # 2,000 resample bootstrap CIs don't rerun per page view.
    return _build_tab1()


@lru_cache(maxsize=1)
def _build_tab1():
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
        # Simple difference-of-means bootstrap - not the causal bootstrap used in PSM.
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
        headline = "Both campaigns produced a lift in spend that's very unlikely to be chance."
    elif wom_sig and not mens_sig:
        headline = (
            "The Women's campaign produced a clear lift in spend. "
            "The Men's result is too close to call."
        )
    elif mens_sig and not wom_sig:
        headline = (
            "The Men's campaign produced a clear lift in spend. "
            "The Women's result is too close to call."
        )
    else:
        headline = (
            "Neither campaign's effect is large enough to separate from chance on the raw "
            "averages. The later tabs use stronger methods to check the same question."
        )

    rev_pct_mens = (lift_mens / avg_control * 100) if avg_control else None
    rev_pct_womens = (lift_womens / avg_control * 100) if avg_control else None
    conv_pct_mens = ((conv_mens - conv_control) / conv_control * 100) if conv_control else None
    conv_pct_womens = ((conv_womens - conv_control) / conv_control * 100) if conv_control else None

    def _hl_col(label, color, sig, proj, proj_lo, proj_hi):
        """Projected total-revenue extrapolation for one arm w/ bootstrap CI band."""
        ci_id = f"hl-ci-{label.lower().replace(' ', '-').replace(chr(39), '')}"
        tooltip_text = (
            "Per-recipient lift multiplied by the number of recipients in this arm. "
            "The range is a 95% confidence interval from a 2k resample bootstrap."
        )
        meta = html.Span(
            [
                html.Span("95% CI", className="label"),
                html.Span(
                    f"{money(proj_lo, 0)} - {money(proj_hi, 0)}",
                    id=ci_id,
                    className="info-term",
                ),
            ]
        )
        return dbc.Col(
            [
                headline_tile(
                    kicker=label,
                    value=money(proj, 0),
                    label="Projected total lift",
                    meta=meta,
                    accent=color,
                    significant=sig,
                ),
                dbc.Tooltip(tooltip_text, target=ci_id, placement="bottom"),
            ],
            md=4,
        )

    return dbc.Container(
        [
            spec_strip(
                ("Customers", f"{len(DF):,}"),
                "3 arms",
                "14-day spend window",
            ),
            page_lede(
                headline=headline,
                caveat=None,
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
                className="text-muted small mb-5",
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
                    section_col(
                        "About the data",
                        html.P(
                            "The Hillstrom dataset is a real email experiment across 64k US "
                            "retail customers: a third got a Men's catalogue email, a third a "
                            "Women's and a third nothing (the control). Spend was recorded over the following "
                            "two weeks, alongside a handful of customer attributes.",
                            className="small text-muted",
                        ),
                        html.P(
                            "The question is whether the email itself caused extra spend, and "
                            "where. Later sections add the confidence around that, which "
                            "customers respond most, and whether different methods agree.",
                            className="small text-muted mb-0",
                        ),
                        md=12,
                    ),
                ],
                className="mb-4 g-4 align-items-start",
            ),
            dbc.Row(
                [
                    section_col(
                        "Spend distribution by segment (among spenders)",
                        graph("tab1-box", figure=_fig_spend_box()),
                        md=6,
                    ),
                    section_col(
                        "Did the randomisation work?",
                        html.P(
                            [
                                "Each dot compares the treatment and control groups on a customer attribute "
                                "(recency, spend history, channel, etc). ",
                                html.Strong("Dots inside the dashed band mean groups are balanced"),
                                ", so any difference in outcomes can be attributed to the email itself "
                                "rather than pre-existing differences between customers. "
                                "A randomised experiment should produce this pattern."
                            ],
                            className="small text-muted mb-2"
                        ),
                        graph("tab1-balance", figure=_fig_covariate_balance()),
                        md=6,
                    ),
                ],
                className="mb-4 g-3 align-items-start"
            ),
        ],
        fluid=True,
        className="py-5"
    )
