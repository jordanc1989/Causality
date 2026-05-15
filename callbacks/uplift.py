
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html, Output, Input, State
from dashboard.theme import *
from dashboard.theme import COVARIATE_LABELS, hex_to_rgba
from dashboard.data import UPLIFT
from layouts.components import kpi_card

def update_uplift(arm, model):
    u = UPLIFT[arm]
    arm_label = "Men's Email" if arm == "mens" else "Women's Email"
    cate_key = {"t": "cate_t", "s": "cate_s", "x": "cate_x"}[model]
    cate = u[cate_key]
    model_label = {"t": "T-Learner", "s": "S-Learner", "x": "X-Learner"}[model]
    color = MENS_COLOUR if arm == "mens" else WOMENS_COLOUR

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
    qini_auc = u.get(qini_auc_key, 0.0)
    qini_excess = u.get(qini_excess_key, 0.0)
    qini_fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"Qini Curve · {model_label} (AUUC = ${qini_auc:,.0f}, excess vs random = ${qini_excess:,.0f})",
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

def toggle_method_tab4(n, is_open):
    return not is_open




def register_uplift_callbacks(app):
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
        Output("method-collapse-tab4", "is_open"),
        Input("method-btn-tab4", "n_clicks"),
        State("method-collapse-tab4", "is_open"),
        prevent_initial_call=True,
    )(toggle_method_tab4)
