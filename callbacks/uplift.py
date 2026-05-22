
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html, Output, Input
from dashboard.theme import *
from dashboard.theme import COVARIATE_LABELS, hex_to_rgba
from dashboard.data import UPLIFT
from layouts.components import kpi_card

def update_uplift(arm, model):
    u = UPLIFT[arm]
    arm_label, color = ARM_META[arm]
    cate = u[_uplift_key("cate", model)]
    model_label = MODEL_LABEL[model]

    kpis = html.Div(
        [
            kpi_card(
                f"${np.mean(cate):.2f}",
                f"Avg CATE ({model_label})",
                f"{arm_label} vs Control",
                np.mean(cate) > 0,
                color=color,
                accent=color,
                info=(
                    "The average predicted lift in spend per customer. This should be close to the "
                    "overall lift on the other tabs. A big gap usually means heavy tails or a model "
                    "quirk."
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
    cate_clipped = cate[(cate >= p1) & (cate <= p99)]  # display only, full CATE used for all analysis
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
        title=f"CATE Distribution - {model_label} ({arm_label})",
        xaxis_title=f"Individual Uplift ($):  showing p1-p99 ({pct_shown:.0f}% of customers)",
        yaxis_title="Count",
        margin=dict(t=50, b=30),
    )

    feat_imp = dict(sorted(u["feat_imp"].items(), key=lambda x: x[1]))
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
        title="Heterogeneity importance (CATE permutation, T-Learner)",
        xaxis_title="Relative importance (normalised)",
        margin=dict(t=50, b=30, l=130),
    )

    decile_key = {"t": "decile_lift", "s": "decile_lift_s", "x": "decile_lift_x"}[model]
    qini_x_key = {"t": "qini_x", "s": "qini_x_s", "x": "qini_x_x"}[model]
    qini_y_key = {"t": "qini_y", "s": "qini_y_s", "x": "qini_y_x"}[model]
    dec_df = pd.DataFrame(u.get(decile_key, u["decile_lift"]))
    qini_xd = u.get(qini_x_key, u["qini_x"])
    qini_yd = u.get(qini_y_key, u["qini_y"])
    overall_ate = dec_df["lift"].mean()

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
                    "<br>95% CI: $%{customdata[0]:.2f} – $%{customdata[1]:.2f}"
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
    decile_fig.add_hline(
        y=overall_ate,
        line_dash="dash",
        line_color=WARNING,
        line_width=1.5,
        annotation_text=f"Avg lift ${overall_ate:.2f}",
        annotation_position="right",
        annotation_font_color=WARNING
    )
    decile_fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"Actual Spend Lift by {model_label} Uplift Decile",
        xaxis=dict(
            title="Decile (1 = highest predicted uplift)",
            tickmode="linear",
            tick0=1,
            dtick=1
        ),
        yaxis_title="Actual Spend Lift ($)",
        margin=dict(t=50, b=30)
    )

    # Plotly fill colours don't accept hex+alpha directly so convert to rgba string
    qini_fill = (
        hex_to_rgba(color, 0.15) if color.startswith("#") else hex_to_rgba(MENS_COLOUR, 0.15)
    )
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
    qini_auc_key = {"t": "qini_auc_t", "s": "qini_auc_s", "x": "qini_auc_x"}[model]
    qini_excess_key = {
        "t": "qini_excess_auc_t",
        "s": "qini_excess_auc_s",
        "x": "qini_excess_auc_x",
    }[model]
    qini_p_key = {"t": "qini_p_t", "s": "qini_p_s", "x": "qini_p_x"}[model]
    qini_p = u.get(qini_p_key)
    qini_auc = u.get(qini_auc_key, 0.0)
    qini_excess = u.get(qini_excess_key, 0.0)
    p_str = (
        f", permutation p = {qini_p:.3f}" if qini_p is not None else ""
    )
    qini_fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=(
            f"Qini Curve · {model_label} (AUUC = ${qini_auc:,.0f}, "
            f"excess vs random = ${qini_excess:,.0f}{p_str})"
        ),
        xaxis_title="Fraction of population targeted",
        yaxis_title="Cumulative incremental spend ($)",
        margin=dict(t=50, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )

    seg_fig = go.Figure()
    for a, lbl, col in [
        ("mens", "Men's Email", MENS_COLOUR),
        ("womens", "Women's Email", WOMENS_COLOUR),
    ]:
        avg_t = UPLIFT[a]["avg_cate_t"]
        avg_s = UPLIFT[a]["avg_cate_s"]
        avg_x = UPLIFT[a].get("avg_cate_x", float("nan"))
        seg_fig.add_trace(
            go.Bar(
                name=lbl,
                x=["T-Learner", "S-Learner", "X-Learner"],
                y=[avg_t, avg_s, avg_x],
                marker_color=col,
                opacity=0.85,
                hovertemplate="%{x}<br>Avg CATE: $%{y:.2f}<extra>%{fullData.name}</extra>",
            )
        )
    seg_fig.update_layout(
        barmode="group",
        template=PLOTLY_TEMPLATE,
        title="Average CATE: Men's vs Women's Campaign",
        yaxis_title="Avg CATE ($)",
        margin=dict(t=50, b=70),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5
        ),
    )

    return kpis, hist_fig, fi_fig, decile_fig, qini_fig, seg_fig


def update_policy(arm, model, cost_per_email, margin):
    """
    Cost-aware targeting policy curve.

    The Qini curve gives cumulative incremental spend `g(p)` as a function of
    the fraction `p` of the list targeted (sorted by predicted CATE desc).
    Net contribution at policy `p` is:

        N_total * [ margin * g(p) - cost_per_email * p ]

    where `g` and the population fraction `p` both come from the Qini arrays
    and `N_total` is the eligible audience for this arm. We report the optimal
    targeting share, the expected net contribution there, and a curve that
    sweeps `p` from 0 to 1 so users can see the tradeoff.

    Cost defaults to $0.05 / send and margin defaults to 0.40. Both are
    user-tunable from the layout's input controls.
    """
    cost_per_email = 0.0 if cost_per_email is None else float(cost_per_email)
    margin = 0.40 if margin is None else float(margin)
    margin = max(0.0, min(1.0, margin))
    u = UPLIFT[arm]
    arm_label = "Men's Email" if arm == "mens" else "Women's Email"
    color = MENS_COLOUR if arm == "mens" else WOMENS_COLOUR

    qini_x_key = {"t": "qini_x", "s": "qini_x_s", "x": "qini_x_x"}[model]
    qini_y_key = {"t": "qini_y", "s": "qini_y_s", "x": "qini_y_x"}[model]
    qini_xd = np.asarray(u.get(qini_x_key, u["qini_x"]), dtype=float)
    qini_yd = np.asarray(u.get(qini_y_key, u["qini_y"]), dtype=float)
    cate_key = {"t": "cate_t", "s": "cate_s", "x": "cate_x"}[model]
    n_total = len(u[cate_key])

    if len(qini_xd) < 2:
        empty = go.Figure()
        empty.update_layout(template=PLOTLY_TEMPLATE, title="Policy curve unavailable")
        return html.Div("Policy curve unavailable."), empty

    # The Qini curve already aggregates over the matched-evaluation cohort.
    # Convert to per-arm totals: gross revenue = qini_y (already in $), and
    # cost = cost_per_email * fraction_targeted * n_total.
    n_targeted = qini_xd * n_total
    gross_contribution = margin * qini_yd
    total_cost = cost_per_email * n_targeted
    net_contribution = gross_contribution - total_cost

    # Optimal policy: pick p* where net_contribution is maximised. Guard the
    # all-negative case by always also reporting "target nobody = $0".
    idx_opt = int(np.argmax(net_contribution))
    p_opt = float(qini_xd[idx_opt])
    net_opt = float(net_contribution[idx_opt])
    n_opt = int(round(qini_xd[idx_opt] * n_total))
    rev_opt = float(qini_yd[idx_opt])
    cost_opt = float(total_cost[idx_opt])
    margin_per_targeted = (gross_contribution[idx_opt] / max(n_opt, 1)) if n_opt > 0 else 0.0

    # If targeting nobody is the best choice (e.g. cost > gross), surface that.
    if net_opt <= 0:
        p_opt = 0.0
        net_opt = 0.0
        n_opt = 0
        verdict = "Do not target. At this cost and margin, no portion of the list returns a positive net contribution."
        verdict_color = WARNING
    elif p_opt >= 0.999:
        verdict = "Target the entire list. Margin per send exceeds cost at every point on the ranking."
        verdict_color = SUCCESS
    else:
        verdict = (
            f"Target the top {p_opt:.0%} of the {arm_label} list (ranked by {model.upper()}-Learner uplift). "
            f"Net contribution peaks at ${net_opt:,.0f}; mailing more dilutes margin, mailing less leaves money on the table."
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
                    "more than the incremental margin they bring in."
                ),
                info_id="policy-kpi-opt-info",
            ),
            kpi_card(
                f"${net_opt:,.0f}",
                "Net incremental contribution",
                f"Gross ${margin * rev_opt:,.0f} − cost ${cost_opt:,.0f}",
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
            name=f"Gross margin (×{margin:.0%})",
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
            hovertemplate="Target top %{x:.0%}<br>Cost: −$%{y:,.0f}<extra></extra>",
        )
    )
    if 0 < p_opt < 1:
        fig.add_vline(
            x=p_opt,
            line_color=SUCCESS,
            line_dash="dash",
            line_width=1.5,
            annotation_text=f"Optimal: top {p_opt:.0%}",
            annotation_position="top right",
            annotation_font_color=SUCCESS,
        )
    fig.add_hline(y=0, line_color=BORDER, line_width=1)
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"Targeting Policy Curve — {arm_label} ({model.upper()}-Learner)",
        xaxis=dict(title="Fraction of list targeted", tickformat=".0%"),
        yaxis_title="Net incremental contribution ($)",
        margin=dict(t=50, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        annotations=[
            dict(
                x=0.5,
                y=-0.42,
                xref="paper",
                yref="paper",
                text=verdict,
                showarrow=False,
                font=dict(size=11, color=verdict_color),
                xanchor="center",
            )
        ],
    )
    return kpis, fig


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
        Input("uplift-arm-selector", "value"),
        Input("uplift-model-selector", "value"),
        Input("policy-cost", "value"),
        Input("policy-margin", "value"),
    )(update_policy)
