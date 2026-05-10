
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html, Output, Input, State
from dashboard.theme import *
from dashboard.data import PSM
from layouts.components import kpi_card
from dashboard.theme import COVARIATE_LABELS

def update_psm(arm):
    p = PSM[arm]
    arm_label = "Men's Email" if arm == "mens" else "Women's Email"
    arm_color = MENS_COLOUR if arm == "mens" else WOMENS_COLOUR

    att_pt = p.get("att_point", float("nan"))
    ci_lo = p.get("att_ci_lo", float("nan"))
    ci_hi = p.get("att_ci_hi", float("nan"))
    att_ok = bool(np.isfinite(att_pt))
    ci_ok = bool(np.isfinite(ci_lo) and np.isfinite(ci_hi))
    ci_str = (
        f"95% CI: ${ci_lo:.2f} – ${ci_hi:.2f}"
        if ci_ok
        else "95% CI unavailable (insufficient bootstrap samples)"
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
                f"${att_pt:.2f}" if att_ok else "—",
                f"ATT — {arm_label}",
                ci_str,
                (att_pt > 0) if att_ok else None,
                color=arm_color,
                accent=arm_color,
                info=(
                    "Average Treatment Effect on the Treated (ATT): mean spend difference between "
                    "each caliper-retained treated row and its nearest matched control."
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
                    "1:1 nearest-neighbour pairing in logit(propensity-score) distance with replacement. "
                    "Pairs outside a 0.2× pooled-SD logit caliper are pruned."
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
                    "Matched rows only. Mean logit-distance to the paired control summarises tightness "
                    "(same metric as NN search + caliper)."
                ),
                info_id="psm-info-pairs"
            ),
            kpi_card(
                cs_str,
                "Common support range",
                info=(
                    "Overlap window on raw propensity scores across both groups. Wide overlap is "
                    "expected here (the logit caliper is what enforces pairwise closeness)."
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
            f"<sup>Overlap diagnostic - {pct_mt:.1f}% retained after 0.2 SD logit caliper</sup>"
        ),
        xaxis_title="Propensity score",
        yaxis_title="Count",
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5
        ),
        margin=dict(t=50, b=70)
    )

    # Love plot — “after” = caliper-kept treated vs their matched controls
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

    if att_ok and ci_ok:
        bar_y = [att_pt, ci_lo, ci_hi]
        bar_text = [f"${v:.2f}" for v in bar_y]
    else:
        bar_y = [0.0, 0.0, 0.0]
        bar_text = ["—", "—", "—"]

    stats_fig = go.Figure(
        go.Bar(
            x=["ATT estimate", "CI lower", "CI upper"],
            y=bar_y,
            marker_color=[arm_color, BORDER, BORDER],
            text=bar_text,
            textposition="outside",
            textfont=dict(color=TEXT),
            hovertemplate="%{x}: $%{y:.2f}<extra></extra>",
        )
    )
    stats_fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="ATT with pooled re-fit/rematch bootstrap (200 reps)",
        yaxis_title="Effect on spend ($)",
        margin=dict(t=50, b=30)
    )

    return kpis, ps_fig, love_fig, stats_fig

def toggle_method_tab2(n, is_open):
    return not is_open




def register_psm_callbacks(app):
    app.callback(
        Output("psm-kpi-cards", "children"),
        Output("psm-ps-dist", "figure"),
        Output("psm-love-plot", "figure"),
        Output("psm-stats-chart", "figure"),
        Input("psm-arm-selector", "value"),
    )(update_psm)
    app.callback(
        Output("method-collapse-tab2", "is_open"),
        Input("method-btn-tab2", "n_clicks"),
        State("method-collapse-tab2", "is_open"),
        prevent_initial_call=True,
    )(toggle_method_tab2)
