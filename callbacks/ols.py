
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dash_table
from dashboard.theme import *
from dashboard.data import OLS, DF


def build_ols_figures():
    """Compute the OLS-tab figures from cached results.

    This is intentionally a plain function, not a Dash callback: the outputs
    only depend on the static `OLS` cache plus `DF`, so the figures can be
    embedded directly in the page layout when the user navigates to /ols.
    """
    coef_df = OLS["coef_df"].copy()
    subgroup_df = OLS["subgroup_df"].copy()

    keep_terms = [t for t in coef_df["term"] if t not in ["Intercept"]]
    plot_df = coef_df[coef_df["term"].isin(keep_terms)].copy()
    plot_df = plot_df.sort_values("coef")

    # Holm-Bonferroni step-down adjustment across all displayed terms. 
    alpha = 0.05
    pvals = plot_df["pvalue"].values.astype(float)
    order = np.argsort(pvals)
    m = len(pvals)
    holm_sig = np.zeros(m, dtype=bool)
    survived = True
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if survived and pvals[idx] <= threshold:
            holm_sig[idx] = True
        else:
            survived = False
    plot_df = plot_df.assign(holm_sig=holm_sig)

    colors = [
        SUCCESS if v > 0 and sig else DANGER if v < 0 and sig else MUTED
        for v, sig in zip(plot_df["coef"], plot_df["holm_sig"])
    ]

    coef_fig = go.Figure()
    coef_fig.add_trace(
        go.Scatter(
            x=plot_df["coef"],
            y=plot_df["term"],
            mode="markers",
            error_x=dict(
                type="data",
                symmetric=False,
                array=plot_df["ci_hi"] - plot_df["coef"],
                arrayminus=plot_df["coef"] - plot_df["ci_lo"],
                color=MUTED
            ),
            marker=dict(color=colors, size=10),
            name="Coefficient",
            customdata=plot_df[["ci_lo", "ci_hi", "pvalue"]].values,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Coef: $%{x:.2f}<br>"
                "95% CI: $%{customdata[0]:.2f} - $%{customdata[1]:.2f}<br>"
                "p-value: %{customdata[2]:.3f}"
                "<extra></extra>"
            ),
        )
    )
    coef_fig.add_vline(x=0, line_color=DANGER, line_dash="dash")
    coef_fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=(
            f"OLS Coefficients (n={OLS['n_obs']:,}, R²={OLS['r_squared']:.4f})<br>"
            "<sup>Colour marks Holm-adjusted significance (α = 0.05), error bars "
            "are unadjusted 95% CIs, so the two can disagree near the cutoff.</sup>"
        ),
        xaxis_title="Effect on Spend ($)",
        xaxis_fixedrange=True,
        yaxis_fixedrange=True,
        dragmode=False,
        margin=dict(t=70, b=30, l=260)
    )

    # Weight the zip-level marginal effects by actual population shares when collapsing to (newbie, channel). 
    zip_spell = {"Urban": "Urban", "Surburban": "Suburban", "Rural": "Rural"}  # raw data misspells "Suburban"
    _cell_counts = (
        DF.assign(
            _newbie=DF["newbie"].map({0: "Existing", 1: "New"}),
            _zip=DF["zip_code"].map(zip_spell),
        )
        .groupby(["_newbie", "channel", "_zip"])
        .size()
        .rename("n")
        .reset_index()
        .rename(columns={"_newbie": "newbie", "_zip": "zip_code"})
    )
    _weighted = subgroup_df.merge(
        _cell_counts, on=["newbie", "channel", "zip_code"], how="left"
    )
    _weighted["n"] = _weighted["n"].fillna(0)

    def _wavg(g):
        """Population-weighted average of marginal effects within a (newbie, channel) group."""
        w = g["n"].values
        if w.sum() == 0:
            return pd.Series(
                {"me_mens": g["me_mens"].mean(), "me_womens": g["me_womens"].mean()}
            )
        return pd.Series(
            {
                "me_mens": float(np.average(g["me_mens"], weights=w)),
                "me_womens": float(np.average(g["me_womens"], weights=w)),
            }
        )

    weighted_sub = (
        _weighted.groupby(["newbie", "channel"], group_keys=False)
        .apply(_wavg)
        .reset_index()
    )

    disp_df = weighted_sub.rename(
        columns={
            "newbie": "Customer type",
            "channel": "Channel",
            "me_mens": "Men's Email ($)",
            "me_womens": "Women's Email ($)"
        }
    )
    disp_df["Men's Email ($)"] = disp_df["Men's Email ($)"].round(2)
    disp_df["Women's Email ($)"] = disp_df["Women's Email ($)"].round(2)

    table = dash_table.DataTable(
        data=disp_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in disp_df.columns],
        style_table={"overflowX": "auto"},
        style_cell=TABLE_CELL,
        style_header=TABLE_HEADER,
        style_data_conditional=[
            {
                "if": {"filter_query": f"{{{col}}} {op} 0", "column_id": col},
                "color": style_color,
            }
            for col in ("Men's Email ($)", "Women's Email ($)")
            for op, style_color in ((">", SUCCESS), ("<", DANGER))
        ] + TABLE_SELECTED,
        page_size=12
    )

    all_vals = pd.concat([weighted_sub["me_mens"], weighted_sub["me_womens"]])
    zmax = max(abs(all_vals.min()), abs(all_vals.max()))
    zmin = -zmax
    heat_colorscale = [
        [0.0, "#7E5A86"],
        [0.5, "#F4F1EA"],
        [1.0, "#2F6E8F"],
    ]

    def make_heatmap(arm_col):
        heat_pivot = weighted_sub.pivot(
            index="newbie", columns="channel", values=arm_col
        )
        return go.Figure(
            go.Heatmap(
                z=heat_pivot.values,
                x=heat_pivot.columns.tolist(),
                y=heat_pivot.index.tolist(),
                colorscale=heat_colorscale,
                zmin=zmin,
                zmax=zmax,
                zmid=0,
                text=[[f"${v:.2f}" for v in row] for row in heat_pivot.values],
                texttemplate="%{text}",
                hovertemplate="%{y} / %{x}<br>Marginal effect: $%{z:.2f}<extra></extra>",
                colorbar=dict(
                    title=dict(text="$ lift", font=dict(color=MUTED)),
                    tickfont=dict(color=MUTED)
                ),
            )
        )

    # Wide bottom margin so the x-axis title clears the tick labels.
    mens_heat = make_heatmap("me_mens")
    mens_heat.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Men's Email: Marginal Effect ($)",
        xaxis_title="Channel",
        yaxis_title="Customer type",
        xaxis_fixedrange=True,
        yaxis_fixedrange=True,
        dragmode=False,
        margin=FIGURE_MARGIN_WIDE
    )

    womens_heat = make_heatmap("me_womens")
    womens_heat.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Women's Email: Marginal Effect ($)",
        xaxis_title="Channel",
        yaxis_title="Customer type",
        xaxis_fixedrange=True,
        yaxis_fixedrange=True,
        dragmode=False,
        margin=FIGURE_MARGIN_WIDE
    )

    return coef_fig, table, mens_heat, womens_heat


def register_ols_callbacks(app):
    pass
