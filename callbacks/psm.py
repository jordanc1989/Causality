
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html, Output, Input
from dashboard.theme import *
from dashboard.data import PSM
from layouts.components import kpi_card
from dashboard.theme import COVARIATE_LABELS

def update_psm(arm):
    p = PSM[arm]
    arm_label, arm_color = ARM_META[arm]

    att_pt = p.get("att_point", float("nan"))
    # Default to matched-pair (Abadie-Imbens style) CI when available, the
    # nonparametric bootstrap is inconsistent for NN matching, so we report it
    # only as a sensitivity band lower in the page.
    ci_lo = p.get("att_ci_lo_matched", p.get("att_ci_lo", float("nan")))
    ci_hi = p.get("att_ci_hi_matched", p.get("att_ci_hi", float("nan")))
    att_ok = bool(np.isfinite(att_pt))
    ci_ok = bool(np.isfinite(ci_lo) and np.isfinite(ci_hi))
    ci_str = (
        f"95% CI: ${ci_lo:.2f} – ${ci_hi:.2f} (matched-pair SE)"
        if ci_ok
        else "95% CI unavailable"
    )
    ps_dist = p.get("avg_logit_ps_distance", p.get("avg_ps_distance"))
    cal_w = p.get("caliper_width")
    pct_mt = float(p.get("pct_matched", 100.0))
    n_drop = int(p.get("n_dropped_no_caliper", 0))
    n_tot = int(p.get("n_treated_total", p.get("n_matched", 0)))

    if ps_dist is not None and np.isfinite(ps_dist) and cal_w is not None:
        dist_str = (
            f"{pct_mt:.1f}% paired - mean |Δlogit(P)| = {ps_dist:.4f} (caliper ≤ {cal_w:.4f}) - "
            f"{n_drop:,} treated dropped"
        )
    else:
        dist_str = "Matching statistics unavailable"
    cs_str = f"{p['cs_lower']:.3f} - {p['cs_upper']:.3f}"
    pct_label = f"{pct_mt:.1f}% matched"

    kpis = html.Div(
        [
            kpi_card(
                f"${att_pt:.2f}" if att_ok else "-",
                f"ATT, {arm_label}",
                ci_str,
                (att_pt > 0) if att_ok else None,
                color=arm_color,
                accent=arm_color,
                info=(
                    "Average lift in spend across the matched pairs. Each email recipient that "
                    "found a close enough lookalike control is compared to that control, and we "
                    "report the mean difference."
                ),
                info_id="psm-info-att"
            ),
            kpi_card(
                pct_label,
                "% treated retained after caliper",
                f"{p.get('n_matched', 0):,} paired of {n_tot:,} treated",
                None,
                accent=ACCENT,
                info=(
                    "How many email recipients found a close enough lookalike in the control "
                    "group. The rest are dropped from this view rather than forced into a "
                    "poor pairing."
                ),
                info_id="psm-info-pct"
            ),
            kpi_card(
                f"{p.get('n_matched', 0):,}",
                "Matched pairs",
                dist_str,
                None,
                accent=MUTED,
                info=(
                    "Number of pairs kept after matching. The distance figure shows how tight "
                    "the average match is, measured on the propensity-score scale."
                ),
                info_id="psm-info-pairs"
            ),
            kpi_card(
                cs_str,
                "Common support range",
                info=(
                    "The propensity-score range where both groups have customers. A wide range "
                    "means the two groups overlap well, which is expected when assignment was "
                    "random."
                ),
                info_id="psm-info-cs"
            ),
        ]
    )

    # Propensity score distribution (full cohorts)
    ps_fig = go.Figure()
    ps_fig.add_trace(
        go.Histogram(
            x=p["propensity_treated"],
            name=f"{arm_label} (all treated)",
            opacity=0.7,
            nbinsx=50,
            marker_color=arm_color
        )
    )
    ps_fig.add_trace(
        go.Histogram(
            x=p["propensity_control"],
            name="Control",
            opacity=0.7,
            nbinsx=50,
            marker_color=CTRL_COLOUR
        )
    )
    ps_fig.update_layout(
        barmode="overlay",
        template=PLOTLY_TEMPLATE,
        title=(
            "Propensity score distribution<br>"
            f"<sup>Overlap between the two groups. {pct_mt:.1f}% of recipients found "
            "a close enough lookalike to keep.</sup>"
        ),
        xaxis_title="Propensity score",
        yaxis_title="Count",
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5
        ),
        margin=dict(t=50, b=70)
    )

    # Love plot: “after” = caliper-kept treated vs their matched controls
    covs = list(p["smd_before"].keys())
    cov_labels = [COVARIATE_LABELS.get(c, c) for c in covs]
    smd_before = [p["smd_before"][c] for c in covs]
    sad = p.get("smd_after", {})
    if isinstance(sad, dict) and sad:
        smd_after_plot = []
        for c in covs:
            v = sad.get(c, float("nan"))
            smd_after_plot.append(float(v) if pd.notna(v) else float("nan"))
    else:
        smd_after_plot = [float("nan")] * len(covs)

    love_fig = go.Figure()
    love_fig.add_trace(
        go.Scatter(
            x=smd_before,
            y=cov_labels,
            mode="markers",
            name="Before (all treated vs controls)",
            marker=dict(color=DANGER, size=10, symbol="circle"),
            hovertemplate="%{y}<br>SMD: %{x:.3f}<extra>Before matching</extra>",
        )
    )
    love_fig.add_trace(
        go.Scatter(
            x=smd_after_plot,
            y=cov_labels,
            mode="markers",
            name="After (retained cohort vs matched ctrl)",
            marker=dict(color=SUCCESS, size=10, symbol="diamond"),
            hovertemplate="%{y}<br>SMD: %{x:.3f}<extra>After caliper NN</extra>",
        )
    )
    love_fig.add_vline(
        x=0.1, line_dash="dash", line_color=WARNING, annotation_text="0.1 threshold"
    )
    love_fig.add_vline(x=-0.1, line_dash="dash", line_color=WARNING)
    love_fig.add_vline(x=0, line_color=BORDER)
    love_fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=(
            "Love plot: covariate balance<br>"
            f"<sup>{p.get('n_matched', 0):,} caliper-kept treated units</sup>"
        ),
        xaxis_title="Standardised mean difference",
        yaxis=dict(automargin=True),
        margin=dict(t=50, b=70, l=195, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )

    # Side-by-side comparison: matched-pair analytical CI (default, defensible)
    # vs the rematch bootstrap CI (sensitivity stress test). The two should
    # roughly agree if the bootstrap behaviour is well-behaved on this data.
    boot_lo = p.get("att_ci_lo", float("nan"))
    boot_hi = p.get("att_ci_hi", float("nan"))
    pair_lo = p.get("att_ci_lo_matched", float("nan"))
    pair_hi = p.get("att_ci_hi_matched", float("nan"))
    has_pair = bool(np.isfinite(pair_lo) and np.isfinite(pair_hi))
    has_boot = bool(np.isfinite(boot_lo) and np.isfinite(boot_hi))

    stats_fig = go.Figure()
    if att_ok:
        stats_fig.add_trace(
            go.Scatter(
                x=["ATT (matched-pair CI)"],
                y=[att_pt],
                mode="markers+text",
                marker=dict(color=arm_color, size=14),
                error_y=(
                    dict(
                        type="data",
                        symmetric=False,
                        array=[pair_hi - att_pt],
                        arrayminus=[att_pt - pair_lo],
                        color=arm_color,
                        thickness=2,
                        width=8,
                    )
                    if has_pair
                    else None
                ),
                text=[f"${att_pt:.2f}"],
                textposition="top center",
                hovertemplate=(
                    "ATT: $%{y:.2f}"
                    + (f"<br>95% CI: ${pair_lo:.2f} – ${pair_hi:.2f}" if has_pair else "")
                    + "<extra></extra>"
                ),
                name="Matched-pair SE",
            )
        )
    if att_ok and has_boot:
        stats_fig.add_trace(
            go.Scatter(
                x=["ATT (rematch-bootstrap band)"],
                y=[att_pt],
                mode="markers+text",
                marker=dict(color=BORDER, size=14),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[boot_hi - att_pt],
                    arrayminus=[att_pt - boot_lo],
                    color=BORDER,
                    thickness=2,
                    width=8,
                ),
                text=[f"${att_pt:.2f}"],
                textposition="top center",
                hovertemplate=(
                    f"ATT: $%{{y:.2f}}<br>Bootstrap band: ${boot_lo:.2f} - ${boot_hi:.2f}<extra></extra>"
                ),
                name="Rematch bootstrap (sensitivity)",
            )
        )
    stats_fig.add_hline(y=0, line_color=BORDER, line_dash="dot")
    stats_fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="ATT estimate - analytical CI vs bootstrap sensitivity band",
        yaxis_title="Effect on spend ($)",
        showlegend=False,
        margin=dict(t=50, b=30),
    )

    return kpis, ps_fig, love_fig, stats_fig


def register_psm_callbacks(app):
    # Methodology-collapse toggle is wired centrally in callbacks/__init__.py.
    app.callback(
        Output("psm-kpi-cards", "children"),
        Output("psm-ps-dist", "figure"),
        Output("psm-love-plot", "figure"),
        Output("psm-stats-chart", "figure"),
        Input("psm-arm-selector", "value"),
    )(update_psm)
