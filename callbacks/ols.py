
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dash_table, Output, Input, State
from dashboard.theme import *
from dashboard.data import OLS, DF
from dashboard.theme import TABLE_CELL, TABLE_HEADER, TABLE_SELECTED

def update_ols(tab):
    if tab != "tab-5":
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    coef_df = OLS["coef_df"].copy()
    subgroup_df = OLS["subgroup_df"].copy()

    keep_terms = [t for t in coef_df["term"] if t not in ["Intercept"]]
    plot_df = coef_df[coef_df["term"].isin(keep_terms)].copy()
    plot_df = plot_df.sort_values("coef")

    colors = [
        SUCCESS if v > 0 and p < 0.05 else DANGER if v < 0 and p < 0.05 else MUTED
        for v, p in zip(plot_df["coef"], plot_df["pvalue"])
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
        title=f"OLS Coefficients (n={OLS['n_obs']:,}, R²={OLS['r_squared']:.4f})",
        xaxis_title="Effect on Spend ($)",
        margin=dict(t=50, b=30, l=260)
    )

    # Weight the zip-level marginal effects by actual population shares when
    # collapsing to (newbie, channel). The raw dataset has wildly uneven zip
    # distributions; an unweighted mean would over-represent rare zip cells.
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
                "if": {
                    "filter_query": "{Men's Email ($)} > 0",
                    "column_id": "Men's Email ($)",
                },
                "color": SUCCESS
            },
            {
                "if": {
                    "filter_query": "{Men's Email ($)} < 0",
                    "column_id": "Men's Email ($)",
                },
                "color": DANGER
            },
            {
                "if": {
                    "filter_query": "{Women's Email ($)} > 0",
                    "column_id": "Women's Email ($)",
                },
                "color": SUCCESS
            },
            {
                "if": {
                    "filter_query": "{Women's Email ($)} < 0",
                    "column_id": "Women's Email ($)",
                },
                "color": DANGER
            },
            *TABLE_SELECTED,
        ],
        page_size=12
    )

    all_vals = pd.concat([weighted_sub["me_mens"], weighted_sub["me_womens"]])
    zmax = max(abs(all_vals.min()), abs(all_vals.max()))
    zmin = -zmax

    def make_heatmap(arm_col):
        heat_pivot = weighted_sub.pivot(
            index="newbie", columns="channel", values=arm_col
        )
        return go.Figure(
            go.Heatmap(
                z=heat_pivot.values,
                x=heat_pivot.columns.tolist(),
                y=heat_pivot.index.tolist(),
                colorscale="RdYlGn",
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

    mens_heat = make_heatmap("me_mens")
    mens_heat.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Men's Email: Marginal Effect ($)",
        xaxis_title="Channel",
        yaxis_title="Customer type",
        margin=dict(t=50, b=30)
    )

    womens_heat = make_heatmap("me_womens")
    womens_heat.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Women's Email: Marginal Effect ($)",
        xaxis_title="Channel",
        yaxis_title="Customer type",
        margin=dict(t=50, b=30)
    )

    return coef_fig, table, mens_heat, womens_heat

def toggle_method_tab5(n, is_open):
    return not is_open




def register_ols_callbacks(app):
    app.callback(
        Output("ols-coef-plot", "figure"),
        Output("ols-marginal-table", "children"),
        Output("ols-heatmap-mens", "figure"),
        Output("ols-heatmap-womens", "figure"),
        Input("main-tabs", "active_tab"),
    )(update_ols)
    app.callback(
        Output("method-collapse-tab5", "is_open"),
        Input("method-btn-tab5", "n_clicks"),
        State("method-collapse-tab5", "is_open"),
        prevent_initial_call=True,
    )(toggle_method_tab5)
