
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html, dash_table, Output, Input, State
import dash_bootstrap_components as dbc
from dashboard.theme import *
from dashboard.data import PSM, BAYESIAN, UPLIFT, OLS
from dashboard.theme import TABLE_CELL, TABLE_HEADER, TABLE_SELECTED

def _build_comparison_df():
    """Assemble a tidy DataFrame of point estimates and CIs across all 5 methods x 2 arms."""
    def _rnd_money(v):
        try:
            x = float(v)
            return round(x, 2) if np.isfinite(x) else None
        except (TypeError, ValueError):
            return None

    rows = []
    for arm in ["mens", "womens"]:
        arm_label = ARM_META[arm][0]
        p = PSM[arm]
        rows.append(
            {
                "Method": "PSM (ATT, matched-pair CI)",
                "Arm": arm_label,
                "Estimate ($)": _rnd_money(p.get("att_point")),
                "CI Lower ($)": _rnd_money(p.get("att_ci_lo_matched")),
                "CI Upper ($)": _rnd_money(p.get("att_ci_hi_matched")),
            }
        )

    pair_map = {"mens": "mens_vs_control", "womens": "womens_vs_control"}
    for arm in ["mens", "womens"]:
        arm_label = ARM_META[arm][0]
        b = BAYESIAN[pair_map[arm]]
        rows.append(
            {
                "Method": "Bayesian A/B (posterior mean)",
                "Arm": arm_label,
                "Estimate ($)": round(float(np.mean(b["delta_samples"])), 2),
                "CI Lower ($)": round(b["hdi_lo"], 2),
                "CI Upper ($)": round(b["hdi_hi"], 2)
            }
        )

    for arm in ["mens", "womens"]:
        arm_label = ARM_META[arm][0]
        u = UPLIFT[arm]
        rows.append(
            {
                "Method": "T-Learner (avg CATE)",
                "Arm": arm_label,
                "Estimate ($)": round(u["avg_cate_t"], 2),
                "CI Lower ($)": None,
                "CI Upper ($)": None
            }
        )
        rows.append(
            {
                "Method": "S-Learner (avg CATE)",
                "Arm": arm_label,
                "Estimate ($)": round(u["avg_cate_s"], 2),
                "CI Lower ($)": None,
                "CI Upper ($)": None
            }
        )
        rows.append(
            {
                "Method": "X-Learner (avg CATE)",
                "Arm": arm_label,
                "Estimate ($)": round(u.get("avg_cate_x", float("nan")), 2),
                "CI Lower ($)": None,
                "CI Upper ($)": None
            }
        )

    # OLS: report the *population-weighted ATE* (average marginal effect over
    # the sample's actual covariate distribution) with its HC3 delta-method CI.
    # The raw `mens_email` / `womens_email` coefficients are only the effect
    # for the reference subgroup (Existing + Phone + Urban) and are not
    # directly comparable to PSM's ATT or the Bayesian delta.
    for arm, ate_key, lo_key, hi_key in [
        ("mens", "ate_mens", "ate_mens_lo", "ate_mens_hi"),
        ("womens", "ate_womens", "ate_womens_lo", "ate_womens_hi"),
    ]:
        rows.append(
            {
                "Method": "OLS (avg marginal effect, HC3)",
                "Arm": ARM_META[arm][0],
                "Estimate ($)": round(OLS.get(ate_key, 0.0), 2),
                "CI Lower ($)": round(OLS.get(lo_key, 0.0), 2),
                "CI Upper ($)": round(OLS.get(hi_key, 0.0), 2),
            }
        )

    return pd.DataFrame(rows)

def update_comparison(noise_eps):
    noise_eps = float(noise_eps) if noise_eps is not None else 0.10
    if noise_eps < 0:
        noise_eps = 0.0

    comp_df = _build_comparison_df()

    table = dash_table.DataTable(
        data=comp_df.fillna("-").to_dict("records"),
        columns=[{"name": c, "id": c} for c in comp_df.columns],
        export_format="csv",
        export_headers="display",
        style_table={"overflowX": "auto"},
        style_cell=TABLE_CELL,
        style_header=TABLE_HEADER,
        style_data_conditional=[
            {
                "if": {
                    "filter_query": "{Estimate ($)} > 0",
                    "column_id": "Estimate ($)"
                },
                "color": SUCCESS
            },
            {
                "if": {
                    "filter_query": "{Estimate ($)} < 0",
                    "column_id": "Estimate ($)"
                },
                "color": DANGER
            },
            *TABLE_SELECTED,
        ],
    )

    # Short labels for the forest-plot y-axis. The full method name (kept as
    # the table label) is too long to render cleanly at narrow viewport widths
    # and forces a hard-coded left margin that clips on mobile. yaxis
    # automargin handles whatever remains.
    short_label_map = {
        "PSM (ATT, matched-pair CI)": "PSM (matched-pair)",
        "Bayesian A/B (posterior mean)": "Bayesian A/B",
        "T-Learner (avg CATE)": "T-Learner",
        "S-Learner (avg CATE)": "S-Learner",
        "X-Learner (avg CATE)": "X-Learner",
        "OLS (avg marginal effect, HC3)": "OLS (HC3)",
    }

    def forest_plot(arm_label, color):
        sub = comp_df[comp_df["Arm"] == arm_label].copy()
        sub["short"] = sub["Method"].map(lambda m: short_label_map.get(m, m))
        fig = go.Figure()
        for i, row in sub.iterrows():
            est_cell = row["Estimate ($)"]
            if pd.isna(est_cell) or est_cell is None:
                continue
            has_ci = pd.notna(row["CI Lower ($)"]) and pd.notna(row["CI Upper ($)"])
            method_name = row["Method"]
            is_psm = isinstance(method_name, str) and method_name.startswith("PSM")
            hover_tail = (
                f"95% CI: ${row['CI Lower ($)']:.2f} – ${row['CI Upper ($)']:.2f}"
                if has_ci
                else (
                    "Heuristic rematch band shown on the PSM tab"
                    if is_psm
                    else "No CI available"
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[row["Estimate ($)"]],
                    y=[row["short"]],
                    mode="markers",
                    marker=dict(
                        color=color,
                        size=12,
                        symbol="diamond",
                        line=dict(color=BG, width=1)
                    ),
                    name=method_name,
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{method_name}</b><br>"
                        f"Estimate: ${row['Estimate ($)']:.2f}<br>"
                        + hover_tail
                        + "<extra></extra>"
                    ),
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=[row["CI Upper ($)"] - row["Estimate ($)"]]
                        if has_ci
                        else [0],
                        arrayminus=[row["Estimate ($)"] - row["CI Lower ($)"]]
                        if has_ci
                        else [0],
                        color=MUTED
                    )
                    if has_ci
                    else None,
                )
            )
        fig.add_vline(x=0, line_color=DANGER, line_dash="dash")
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            title=f"Forest Plot - {arm_label}",
            xaxis_title="Effect on Spend ($)",
            yaxis=dict(automargin=True),
            margin=dict(t=50, b=30, l=20, r=20),
            height=350,
            autosize=True,
        )
        return fig

    mens_fig = forest_plot("Men's Email", MENS_COLOUR)
    womens_fig = forest_plot("Women's Email", WOMENS_COLOUR)

    mens_estimates = comp_df[comp_df["Arm"] == "Men's Email"]["Estimate ($)"].values
    womens_estimates = comp_df[comp_df["Arm"] == "Women's Email"]["Estimate ($)"].values
    mens_valid = [float(v) for v in mens_estimates if pd.notna(v)]
    womens_valid = [float(v) for v in womens_estimates if pd.notna(v)]
    mens_min = min(mens_valid) if mens_valid else 0.0
    mens_max = max(mens_valid) if mens_valid else 0.0
    womens_min = min(womens_valid) if womens_valid else 0.0
    womens_max = max(womens_valid) if womens_valid else 0.0

    # Robust verdict: don't flip on a single near-zero estimate. The "noise
    # zone" threshold is user-controllable via the agreement-threshold input,
    # the default ($0.10) is smaller than any plausible action threshold in
    # this dataset.
    def _verdict(estimates):
        material = [v for v in estimates if abs(v) >= noise_eps]
        near_zero = [v for v in estimates if abs(v) < noise_eps]
        if not material:
            return "Every method sits close to zero."
        pos = sum(1 for v in material if v > 0)
        neg = sum(1 for v in material if v < 0)
        if pos > 0 and neg == 0:
            tail = (
                f" ({len(near_zero)} sat near zero.)" if near_zero else ""
            )
            return "Every method points to a positive lift." + tail
        if neg > 0 and pos == 0:
            tail = (
                f" ({len(near_zero)} sat near zero.)" if near_zero else ""
            )
            return "Every method points to a negative effect." + tail
        return (
            f"Methods disagree on direction ({pos} positive, {neg} negative). "
            "Worth a closer look at the assumptions behind each one."
        )

    mens_verdict = _verdict(mens_valid)
    womens_verdict = _verdict(womens_valid)

    takeaway = dbc.Card(
        [
            dbc.CardHeader("Key Takeaway"),
            dbc.CardBody(
                [
                    html.P(
                        [
                            html.Strong(
                                "Men's campaign: ", style={"color": MENS_COLOUR}
                            ),
                            f"Estimated spend uplift ranges from ${mens_min:.2f} to ${mens_max:.2f} across "
                            f"{len(mens_valid)} methods. {mens_verdict}"
                        ]
                    ),
                    html.P(
                        [
                            html.Strong(
                                "Women's campaign: ", style={"color": WOMENS_COLOUR}
                            ),
                            f"Estimated spend uplift ranges from ${womens_min:.2f} to ${womens_max:.2f} across "
                            f"{len(womens_valid)} methods. {womens_verdict}"
                        ]
                    ),
                    html.P(
                        "When the methods agree, you can lean on the headline number with confidence. "
                        "When they disagree, the gap usually points to where the assumptions of one "
                        "method are doing more work. The Overview and OLS tabs lean directly on the "
                        "random assignment. The Bayesian tab adds a richer picture of uncertainty. "
                        "The Uplift tab focuses on who responds most, rather than the average.",
                        className="text-muted small mb-0"
                    ),
                ]
            ),
        ],
        style={
            **CARD_STYLE,
            "borderLeft": f"3px solid {SUCCESS if ('points to' in mens_verdict and 'points to' in womens_verdict) else WARNING}"
        },
        className="dashboard-card",
    )

    return table, mens_fig, womens_fig, takeaway




def register_comparison_callbacks(app):
    # With Dash Pages, the comparison layout only mounts when the user navigates
    # to /comparison, so the tab-active gate that used to short-circuit this
    # callback is no longer needed.
    app.callback(
        Output("comparison-table", "children"),
        Output("forest-plot-mens", "figure"),
        Output("forest-plot-womens", "figure"),
        Output("key-takeaway-card", "children"),
        Input("comparison-noise-input", "value"),
    )(update_comparison)
