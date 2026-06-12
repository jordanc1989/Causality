
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html, Output, Input
from causal_utils import AUUC_N_PERM
from dashboard.theme import *
from dashboard.data import UPLIFT
from dashboard.format import money_range
from layouts.components import kpi_card

# Per-model cached results are stored under a `{base}_{model}` key in UPLIFT[arm]
# (e.g. cate_s / cate_x). Keep the UI label and cached result keys together.
MODEL_KEYS = {
    "s": dict(
        label="S-Learner", cate="cate_s", avg="avg_cate_s",
        decile="decile_lift_s", qini_x="qini_x_s", qini_y="qini_y_s",
        auc="qini_auc_s", excess="qini_excess_auc_s", p="qini_p_s",
        feat_imp="feat_imp_s",
    ),
    "x": dict(
        label="X-Learner", cate="cate_x", avg="avg_cate_x",
        decile="decile_lift_x", qini_x="qini_x_x", qini_y="qini_y_x",
        auc="qini_auc_x", excess="qini_excess_auc_x", p="qini_p_x",
        feat_imp="feat_imp_x",
    ),
}
MODEL_LABEL = {key: meta["label"] for key, meta in MODEL_KEYS.items()}

# The audience-scaled Qini value is extremely noisy over the first ~2% of the
# ranking (it rests on a handful of treated customers), so the policy optimiser
# never recommends a targeting share below this floor.
MIN_TARGET_SHARE = 0.02


def update_uplift(arm, model):
    u = UPLIFT[arm]
    arm_label, color = ARM_META[arm]
    keys = MODEL_KEYS[model]
    cate = u[keys["cate"]]
    model_label = keys["label"]

    avg_lo = u.get(f"avg_cate_{model}_lo")
    avg_hi = u.get(f"avg_cate_{model}_hi")
    if avg_lo is not None and avg_hi is not None and np.isfinite(avg_lo) and np.isfinite(avg_hi):
        avg_delta = f"95% CI {money_range(avg_lo, avg_hi)}"
    else:
        avg_delta = f"{arm_label} vs Control"

    kpis = html.Div(
        [
            kpi_card(
                f"${np.mean(cate):.2f}",
                f"Avg CATE ({model_label})",
                avg_delta,
                np.mean(cate) > 0,
                color=color,
                accent=color,
                info=(
                    "The average predicted lift in spend per customer, with a 95% bootstrap "
                    "interval. This should be close to the overall lift on the other tabs. The "
                    "interval is estimation-conditional, so read it as a lower bound on the true "
                    "uncertainty."
                ),
                info_id="uplift-kpi-avg-info",
            ),
            kpi_card(
                f"${np.percentile(cate, 90):.2f}",
                "90th percentile CATE",
                "High-responder threshold",
                accent=ACCENT,
                info=(
                    "The predicted uplift you'd see at the cutoff for the top 10% of customers. "
                    "Useful for sizing a high-value targeting segment."
                ),
                info_id="uplift-kpi-p90-info",
            ),
            kpi_card(
                f"{np.mean(cate > 0):.1%}",
                "% customers with positive uplift",
                info=(
                    "Share of customers the model expects to spend more if mailed. A ceiling on "
                    "the audience worth targeting. Customers with negative predicted uplift "
                    "shouldn't be mailed."
                ),
                info_id="uplift-kpi-ppos-info",
            ),
        ]
    )

    p1, p99 = np.percentile(cate, 1), np.percentile(cate, 99)
    # Display only, full CATE is used for all analysis.
    cate_clipped = cate[(cate >= p1) & (cate <= p99)]
    pct_shown = len(cate_clipped) / len(cate) * 100

    hist_fig = go.Figure(
        go.Histogram(
            x=cate_clipped,
            nbinsx=60,
            marker_color=color,
            opacity=0.8,
            name="CATE",
        )
    )
    hist_fig.add_vline(x=0, line_color=DANGER, line_dash="dash")
    hist_fig.add_vline(
        x=np.mean(cate), line_color=WARNING, line_dash="dot", annotation_text="Mean"
    )
    hist_fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"CATE distribution — {model_label} ({arm_label})",
        xaxis_title=(
            f"Individual uplift ($), p1–p99 ({pct_shown:.0f}% of customers)"
        ),
        yaxis_title="Count",
        margin=FIGURE_MARGIN,
    )

    feat_imp = dict(
        sorted(u[keys["feat_imp"]].items(), key=lambda x: x[1])
    )
    feat_labels = [COVARIATE_LABELS.get(k, k) for k in feat_imp.keys()]
    fi_fig = go.Figure(
        go.Bar(
            x=list(feat_imp.values()),
            y=feat_labels,
            orientation="h",
            marker_color=color,
            hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>",
        )
    )
    fi_fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=u.get(
            f"feat_imp_label_{model}",
            f"Heterogeneity importance ({model_label} CATE permutation)",
        ),
        xaxis_title="Relative importance (normalised)",
        xaxis_fixedrange=True,
        yaxis_fixedrange=True,
        dragmode=False,
        margin=dict(t=50, b=30, l=170),
    )

    dec_df = pd.DataFrame(u[keys["decile"]])
    qini_xd = u[keys["qini_x"]]
    qini_yd = u[keys["qini_y"]]
    mean_decile_lift = dec_df["lift"].mean()

    has_ci = "ci_lo" in dec_df.columns and "ci_hi" in dec_df.columns
    decile_fig = go.Figure()
    decile_fig.add_trace(
        go.Bar(
            x=dec_df["decile"],
            y=dec_df["lift"],
            marker_color=[color if v > 0 else DANGER for v in dec_df["lift"]],
            opacity=0.6,
            showlegend=False,
            hovertemplate=(
                "Decile %{x}<br>Actual lift: $%{y:.2f}"
                + (
                    "<br>95% CI: $%{customdata[0]:.2f}–$%{customdata[1]:.2f}"
                    if has_ci
                    else ""
                )
                + "<extra></extra>"
            ),
            customdata=(
                dec_df[["ci_lo", "ci_hi"]].values if has_ci else None
            ),
            error_y=(
                dict(
                    type="data",
                    symmetric=False,
                    array=(dec_df["ci_hi"] - dec_df["lift"]).values,
                    arrayminus=(dec_df["lift"] - dec_df["ci_lo"]).values,
                    color=MUTED,
                    thickness=1.2,
                    width=4,
                )
                if has_ci
                else None
            ),
        )
    )
    decile_fig.add_trace(
        go.Scatter(
            x=dec_df["decile"],
            y=dec_df["lift"],
            mode="markers+text",
            marker=dict(
                color=[color if v > 0 else DANGER for v in dec_df["lift"]],
                size=9,
                line=dict(color=BG, width=1),
            ),
            text=[f"${v:.2f}" for v in dec_df["lift"]],
            textposition="top center",
            textfont=dict(size=10),
            showlegend=False,
            hovertemplate="Decile %{x}<br>Actual lift: $%{y:.2f}<extra></extra>",
        )
    )
    # The line's label is in the subtitle, annotations here clip at the plot
    # edge or collide with the bars.
    decile_fig.add_hline(
        y=mean_decile_lift,
        line_dash="dash",
        line_color=WARNING,
        line_width=1.5,
    )
    decile_fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=(
            f"Actual spend lift by {model_label} uplift decile<br>"
            f"<sup>Dashed line: mean decile lift ${mean_decile_lift:.2f}</sup>"
        ),
        xaxis=dict(
            title="Decile (1 = highest predicted uplift)",
            tickmode="linear",
            tick0=1,
            dtick=1,
            fixedrange=True,
        ),
        yaxis_title="Actual spend lift ($)",
        yaxis_fixedrange=True,
        dragmode=False,
        margin=FIGURE_MARGIN
    )

    # Plotly fill colours don't accept hex+alpha directly so convert to rgba string
    qini_fill = hex_to_rgba(color, 0.15)
    qini_fig = go.Figure()
    qini_fig.add_trace(
        go.Scatter(
            x=qini_xd,
            y=qini_yd,
            mode="lines",
            name=f"{model_label} Qini",
            line=dict(color=color, width=2),
            fill="tozeroy",
            fillcolor=qini_fill,
            hovertemplate="Top %{x:.0%} targeted<br>Cumulative incremental spend: $%{y:,.0f}<extra>%{fullData.name}</extra>",
        )
    )
    qini_fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, qini_yd[-1] if len(qini_yd) else 0],
            mode="lines",
            name="Random baseline",
            line=dict(color=BORDER, dash="dash"),
            hoverinfo="skip",
        )
    )
    qini_p = u.get(keys["p"])
    qini_auc = u.get(keys["auc"], 0.0)
    qini_excess = u.get(keys["excess"], 0.0)
    # The permutation test can't resolve below 1/(n_perm + 1), so a value at
    # that floor means "no permutation reached the observed AUUC" - report it
    # as an upper bound, not an exact p.
    p_floor = 1.0 / (AUUC_N_PERM + 1)
    if qini_p is None:
        p_str = ""
    elif qini_p <= p_floor:
        p_str = f" · permutation p < {p_floor:.3f}"
    else:
        p_str = f" · permutation p = {qini_p:.3f}"
    qini_note = (
        f"AUUC ${qini_auc:,.0f} · excess vs random ${qini_excess:,.0f}{p_str}"
    )
    qini_fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"Qini curve — {model_label}",
        xaxis_title="Fraction of population targeted",
        yaxis_title="Cumulative incremental spend ($)",
        margin=dict(t=72, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        annotations=[
            dict(
                text=qini_note,
                x=0,
                y=1.08,
                xref="paper",
                yref="paper",
                showarrow=False,
                xanchor="left",
                font=dict(family=MONO, size=10, color=MUTED),
            )
        ],
    )

    seg_fig = go.Figure()
    for a, lbl, col in [
        ("mens", "Men's Email", MENS_COLOUR),
        ("womens", "Women's Email", WOMENS_COLOUR),
    ]:
        seg_fig.add_trace(
            go.Bar(
                name=lbl,
                x=list(MODEL_LABEL.values()),
                y=[UPLIFT[a].get(meta["avg"], float("nan")) for meta in MODEL_KEYS.values()],
                marker_color=col,
                opacity=0.85,
                hovertemplate="%{x}<br>Avg CATE: $%{y:.2f}<extra>%{fullData.name}</extra>",
            )
        )
    seg_fig.update_layout(
        barmode="group",
        template=PLOTLY_TEMPLATE,
        title="Average CATE — Men's vs Women's campaign",
        yaxis_title="Avg CATE ($)",
        xaxis_fixedrange=True,
        yaxis_fixedrange=True,
        dragmode=False,
        margin=FIGURE_MARGIN_WIDE,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5
        ),
    )

    return kpis, hist_fig, fi_fig, decile_fig, qini_fig, seg_fig


def update_policy(arm, model, cost_per_email, margin):
    """
    Cost-aware targeting policy curve.

    The Qini curve gives audience-scaled cumulative incremental spend `g(p)`
    as a function of the fraction `p` of the list targeted (sorted by
    predicted CATE desc).
    Net contribution at policy `p` is:

        margin * g(p) - cost_per_email * p * N_total

    where `g` and `p` both come from the Qini arrays and `N_total` is the
    eligible audience for this arm. We report the optimal targeting share, the
    expected net contribution there, and a curve that sweeps `p` from 0 to 1 so
    users can see the tradeoff.

    Cost defaults to $0.05 / send and margin defaults to 0.40. Both are
    user-tunable from the layout's input controls.
    """
    cost_per_email = 0.05 if cost_per_email is None else max(0.0, float(cost_per_email))
    margin = 0.40 if margin is None else float(margin)
    margin = max(0.0, min(1.0, margin))
    u = UPLIFT[arm]
    arm_label, color = ARM_META[arm]
    keys = MODEL_KEYS[model]

    qini_xd = np.asarray(u[keys["qini_x"]], dtype=float)
    qini_yd = np.asarray(u[keys["qini_y"]], dtype=float)
    n_total = len(u[keys["cate"]])

    if len(qini_xd) < 2:
        empty = go.Figure()
        empty.update_layout(template=PLOTLY_TEMPLATE, title="Policy curve unavailable")
        return html.Div("Policy curve unavailable."), empty, html.Div()

    # The Qini curve is scaled to the full ranked audience, so gross revenue
    # and send cost are both expressed for the same targeted customer count.
    n_targeted = qini_xd * n_total
    gross_contribution = margin * qini_yd
    total_cost = cost_per_email * n_targeted
    net_contribution = gross_contribution - total_cost

    # Optimal policy: pick p* where net_contribution is maximised, ignoring
    # shares below MIN_TARGET_SHARE where the audience-scaled curve is too
    # noisy to trust (a low send cost can otherwise put the "optimum" on a
    # spurious early spike). Guard the all-negative case by always also
    # reporting "target nobody = $0".
    eligible = qini_xd >= MIN_TARGET_SHARE
    if eligible.any():
        eligible_idx = np.flatnonzero(eligible)
        idx_opt = int(eligible_idx[np.argmax(net_contribution[eligible])])
    else:
        idx_opt = int(np.argmax(net_contribution))
    p_opt = float(qini_xd[idx_opt])
    net_opt = float(net_contribution[idx_opt])
    n_opt = int(round(qini_xd[idx_opt] * n_total))
    rev_opt = float(qini_yd[idx_opt])
    cost_opt = float(total_cost[idx_opt])
    margin_per_targeted = (gross_contribution[idx_opt] / max(n_opt, 1)) if n_opt > 0 else 0.0

    # If targeting nobody is the best choice (e.g. cost > gross), surface that
    # and zero out the per-optimum figures so the KPI cards don't show the
    # gross/cost/margin of the optimum we just rejected.
    if net_opt <= 0:
        p_opt = 0.0
        net_opt = 0.0
        n_opt = 0
        rev_opt = 0.0
        cost_opt = 0.0
        margin_per_targeted = 0.0
        verdict = "Do not target. At this cost and margin, no portion of the list returns a positive net contribution."
        verdict_color = WARNING
    elif p_opt >= 0.999:
        verdict = "Target the entire list. Margin per send exceeds cost at every point on the ranking."
        verdict_color = SUCCESS
    else:
        verdict = (
            f"Target the top {p_opt:.0%} of the {arm_label} list (ranked by {keys['label']} uplift). "
            f"Net contribution peaks at ${net_opt:,.0f}."
        )
        verdict_color = SUCCESS

    kpis = html.Div(
        [
            kpi_card(
                f"{p_opt:.0%}",
                "Optimal target share",
                f"≈ {n_opt:,} customers",
                color=verdict_color,
                accent=verdict_color,
                info=(
                    "The fraction of the ranked list to mail that maximises net "
                    "incremental contribution. Above this fraction, additional sends cost "
                    "more than the incremental margin they bring in. Shares below 2% of "
                    "the list are never recommended: estimates there rest on too few "
                    "customers to be reliable."
                ),
                info_id="policy-kpi-opt-info",
            ),
            kpi_card(
                f"${net_opt:,.0f}",
                "Net incremental contribution",
                f"Gross ${margin * rev_opt:,.0f} - cost ${cost_opt:,.0f}",
                accent=color,
                info=(
                    "Gross incremental margin from the targeted segment minus the cost of sending. "
                    "Calculated from the Qini curve at the optimal targeting share."
                ),
                info_id="policy-kpi-net-info",
            ),
            kpi_card(
                f"${margin_per_targeted:.2f}",
                "Avg margin per email sent",
                "At the optimal threshold",
                info=(
                    "Average gross margin per email at the optimal targeting share. "
                    "Useful for justifying spend on richer creative or paid touchpoints "
                    "for the same audience."
                ),
                info_id="policy-kpi-marg-info",
            ),
        ]
    )

    # Sweep the Qini grid and plot net contribution vs fraction targeted, with
    # the optimal point marked and a verdict below the chart.
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=qini_xd,
            y=net_contribution,
            mode="lines",
            line=dict(color=color, width=2),
            fill="tozeroy",
            fillcolor=hex_to_rgba(color, 0.12),
            name="Net contribution",
            hovertemplate=(
                "Target top %{x:.0%}<br>"
                "Net contribution: $%{y:,.0f}"
                "<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=qini_xd,
            y=gross_contribution,
            mode="lines",
            line=dict(color=MUTED, width=1, dash="dot"),
            name=f"Gross margin (x{margin:.0%})",
            hovertemplate="Target top %{x:.0%}<br>Gross margin: $%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=qini_xd,
            y=-total_cost,
            mode="lines",
            line=dict(color=DANGER, width=1, dash="dot"),
            name="Send cost",
            customdata=total_cost,
            hovertemplate="Target top %{x:.0%}<br>Cost: $%{customdata:,.0f}<extra></extra>",
        )
    )
    if 0 < p_opt < 1:
        fig.add_vline(
            x=p_opt,
            line_color=SUCCESS,
            line_dash="dash",
            line_width=1.5,
            annotation_text=f"Optimal: top {p_opt:.0%}",
            # Near 100% a right-side label clips off the canvas.
            annotation_position="top right" if p_opt < 0.8 else "top left",
            annotation_font_color=SUCCESS,
        )
    fig.add_hline(y=0, line_color=BORDER, line_width=1)
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"Targeting policy curve — {arm_label} ({keys['label']})",
        xaxis=dict(title="Fraction of list targeted", tickformat=".0%"),
        yaxis_title="Net incremental contribution ($)",
        margin=dict(t=50, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
    )
    # Page content rather than a figure annotation: anything placed below the
    # legend gets clipped off the canvas.
    verdict_note = html.Div(
        verdict,
        className="small",
        style={"color": verdict_color, "fontStyle": "italic", "marginTop": "0.4rem"},
    )
    return kpis, fig, verdict_note


def register_uplift_callbacks(app):
    # Methodology-collapse toggle is wired centrally in callbacks/__init__.py.
    app.callback(
        Output("uplift-kpi-cards", "children"),
        Output("uplift-cate-hist", "figure"),
        Output("uplift-feat-imp", "figure"),
        Output("uplift-decile-chart", "figure"),
        Output("uplift-qini", "figure"),
        Output("uplift-segment-compare", "figure"),
        Input("uplift-arm-selector", "value"),
        Input("uplift-model-selector", "value"),
    )(update_uplift)
    app.callback(
        Output("policy-kpi-cards", "children"),
        Output("policy-curve", "figure"),
        Output("policy-verdict", "children"),
        Input("uplift-arm-selector", "value"),
        Input("uplift-model-selector", "value"),
        Input("policy-cost", "value"),
        Input("policy-margin", "value"),
    )(update_policy)
