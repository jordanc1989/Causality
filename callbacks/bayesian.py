
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html, Output, Input
import dash_bootstrap_components as dbc
from dashboard.theme import *
from dashboard.data import BAYESIAN
from layouts.components import kpi_card

def update_bayesian(pair_key, rope_val):
    # None = cleared input; restore the layout default. A zero ROPE reads as
    # 100% outside and claims strong evidence.
    rope_val = 1.0 if rope_val is None else max(0.0, float(rope_val))
    b = BAYESIAN[pair_key]
    delta = b["delta_samples"]

    hdi_str = f"95% HDI: ${b['hdi_lo']:.2f} - ${b['hdi_hi']:.2f}"
    p_pos = b["p_positive"]
    # Near 0% is strong evidence of a negative effect, not uncertainty.
    p_pos_color = SUCCESS if p_pos > 0.95 else DANGER if p_pos < 0.05 else WARNING
    arm_color = MENS_COLOUR if pair_key.startswith("mens") else WOMENS_COLOUR

    kpis = html.Div(
        [
            kpi_card(
                hdi_str,
                f"Treatment effect: {b['arm_a_label']} vs {b['arm_b_label']}",
                accent=arm_color,
                info=(
                    "The 95% Highest Density Interval is the narrowest range that holds 95% of "
                    "the posterior. Under this model and the data, there's a 95% probability the "
                    "per-customer effect sits inside it."
                ),
                info_id="bayes-kpi-hdi-info",
            ),
            kpi_card(
                f"{p_pos:.1%}",
                "P(effect > 0)",
                color=p_pos_color,
                accent=p_pos_color,
                info=(
                    "Under the model, the probability that the per-customer effect is "
                    "positive. Above 95% is strong evidence of a lift, below 5% is "
                    "equally strong evidence the effect runs the other way. Near 50% "
                    "means the data can't tell which way the effect points."
                ),
                info_id="bayes-kpi-ppos-info",
            ),
            kpi_card(
                f"${b['mean_a']:.2f}",
                f"Mean spend, {b['arm_a_label']}",
                accent=arm_color,
                info=(
                    "The model's estimate of the average per-customer spend for this arm, "
                    "combining the chance of any spend with the size of the spend when it happens."
                ),
                info_id="bayes-kpi-meana-info",
            ),
            kpi_card(
                f"${b['mean_b']:.2f}",
                f"Mean spend, {b['arm_b_label']}",
                accent=CTRL_COLOUR,
                info=(
                    "The model's estimate of the average per-customer spend for this arm, "
                    "combining the chance of any spend with the size of the spend when it happens."
                ),
                info_id="bayes-kpi-meanb-info",
            ),
        ]
    )

    # A binned outline reads as sampler noise the posterior doesn't have.
    from scipy.stats import gaussian_kde

    grid = np.linspace(float(delta.min()), float(delta.max()), 400)
    density = gaussian_kde(delta)(grid)

    posterior_fig = go.Figure()

    rope_mask = (grid >= -rope_val) & (grid <= rope_val)
    if rope_mask.any():
        posterior_fig.add_trace(
            go.Scatter(
                x=np.concatenate(
                    [
                        [grid[rope_mask][0]],
                        grid[rope_mask],
                        [grid[rope_mask][-1]],
                    ]
                ),
                y=np.concatenate([[0], density[rope_mask], [0]]),
                fill="toself",
                fillcolor=hex_to_rgba(WARNING, 0.18),
                line=dict(color="rgba(0,0,0,0)"),
                name=f"ROPE ±${rope_val}",
                showlegend=True,
                hoverinfo="skip",
            )
        )

    line_color = MENS_COLOUR if pair_key.startswith("mens") else WOMENS_COLOUR
    fill_rgba = hex_to_rgba(line_color, 0.15)

    posterior_fig.add_trace(
        go.Scatter(
            x=grid,
            y=density,
            mode="lines",
            fill="tozeroy",
            fillcolor=fill_rgba,
            line=dict(color=line_color, width=2),
            name="Posterior δ",
            hovertemplate="Effect: $%{x:.2f}<extra>Posterior δ</extra>",
        )
    )

    posterior_fig.add_vline(
        x=b["hdi_lo"],
        line_dash="dash",
        line_color=MUTED,
        annotation_text="95% HDI lower",
        annotation_position="top left",
        annotation_font_color=MUTED
    )
    posterior_fig.add_vline(
        x=b["hdi_hi"],
        line_dash="dash",
        line_color=MUTED,
        annotation_text="95% HDI upper",
        # An outside-right label clips at the canvas edge.
        annotation_position="top left",
        annotation_font_color=MUTED
    )
    posterior_fig.add_vline(
        x=0,
        line_color=DANGER,
        line_width=1.5,
        line_dash="dot"
    )

    posterior_fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"{b['arm_a_label']} vs {b['arm_b_label']}",
        xaxis_title="Effect on Spend ($)",
        yaxis_title="Density",
        margin=dict(t=40, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )

    p_outside_rope = float(np.mean((delta > rope_val) | (delta < -rope_val)))
    rope_color = SUCCESS if p_outside_rope > 0.9 else WARNING
    verdict = (
        "Strong evidence the effect is big enough to matter."
        if p_outside_rope > 0.9
        else "Not decisive yet. The effect could still be too small to act on."
    )
    rope_card = html.Div(
        [
            html.Div(
                [
                    html.Span(f"{p_outside_rope:.1%}",
                              style={"fontSize": "2rem", "fontFamily": MONO,
                                     "fontWeight": "600", "letterSpacing": "-0.01em",
                                     "color": rope_color, "lineHeight": "1"}),
                    html.Span(
                        f"P(|δ| > ${rope_val})",
                        id="bayes-rope-info",
                        className="info-term",
                        style={"fontSize": "0.82rem", "color": MUTED,
                               "fontFamily": SERIF,
                               "marginLeft": "10px", "verticalAlign": "middle"},
                    ),
                ],
                style={"display": "flex", "alignItems": "baseline", "marginBottom": "0.3rem"},
            ),
            html.Div(verdict, className="small text-muted",
                     style={"fontStyle": "italic", "marginBottom": 0}),
            dbc.Tooltip(
                f"How much of the posterior sits outside ±${rope_val} per customer. Values "
                f"inside that range are treated as practically equivalent to zero, so the "
                f"probability outside it is the evidence the effect is big enough to act on.",
                target="bayes-rope-info",
                placement="top",
            ),
        ],
        style={"borderTop": f"2px solid {rope_color}", "paddingTop": "0.7rem"},
    )

    return kpis, posterior_fig, rope_card

def update_ppc_figure(pair_key):
    b = BAYESIAN[pair_key]
    lab_a = b["arm_a_label"]
    lab_b = b["arm_b_label"]
    arm_colours = {
        "Mens E-Mail": MENS_COLOUR,
        "Womens E-Mail": WOMENS_COLOUR,
        "No E-Mail": CTRL_COLOUR,
    }
    colour_a = arm_colours.get(lab_a, MENS_COLOUR)
    colour_b = arm_colours.get(lab_b, CTRL_COLOUR)
    pack = b.get("ppc_pack")

    if pack is None:
        fig = go.Figure()
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            title="Posterior predictive: rebuild `.cache/results.pkl` after upgrading causal_utils",
            height=420,
            margin=dict(t=60, b=50),
        )
        return fig

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            f"Full spend - {lab_a}",
            f"Full spend - {lab_b}",
            f"Amount given spend > 0 - {lab_a}",
            f"Amount given spend > 0 - {lab_b}",
            f"Conversion rate - {lab_a}",
            f"Conversion rate - {lab_b}",
        ),
        vertical_spacing=0.11,
        horizontal_spacing=0.08,
    )
    # Subplot titles render as annotations, quieten them so they don't fight the
    # data or the figure title. (Set before traces/vlines add their own.)
    fig.update_annotations(font=dict(family=SERIF, size=12.5, color=MUTED))

    obs_sa = b.get("observed_spend_a")
    obs_sb = b.get("observed_spend_b")
    ppc_sa = pack.get("ppc_spend_display_a")
    ppc_sb = pack.get("ppc_spend_display_b")
    obs_pa = b.get("observed_amount_a")
    obs_pb = b.get("observed_amount_b")
    ppc_pa = pack.get("ppc_amount_pos_a")
    ppc_pb = pack.get("ppc_amount_pos_b")
    ocra = b.get("obs_conv_rate_a")
    ocrb = b.get("obs_conv_rate_b")
    pcrma = pack.get("ppc_conv_rep_mean_a")
    pcrmb = pack.get("ppc_conv_rep_mean_b")

    # Clip the spend / amount x-axes to the 99th percentile of observed positive
    # spend. The raw positive-spend distribution has a long right tail that
    # squashes the bulk of the data into a thin spike at the left, so the
    # row-1 zero spike + row-2 mass become unreadable. 
    _pos_pool = np.concatenate([
        a for a in (obs_pa, obs_pb) if a is not None and len(a) > 0
    ]) if any(a is not None and len(a) > 0 for a in (obs_pa, obs_pb)) else None
    if _pos_pool is not None and len(_pos_pool) > 0:
        spend_xmax = float(np.percentile(_pos_pool, 99))
    else:
        spend_xmax = 300.0

    def _add_hist_pair(row, col, obs_x, ppc_x, obs_name, ppc_name, color_o, color_p):
        if obs_x is None or ppc_x is None or len(obs_x) == 0 or len(ppc_x) == 0:
            return
        fig.add_trace(
            go.Histogram(
                x=obs_x,
                name=obs_name,
                histnorm="probability density",
                marker_color=color_o,
                opacity=0.42,
                nbinsx=70,
                # One legend entry per arm's observed colour (row 1 has both
                # columns), so every swatch matches what's on screen.
                showlegend=(row == 1),
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Histogram(
                x=ppc_x,
                name=ppc_name,
                histnorm="probability density",
                marker_color=color_p,
                opacity=0.42,
                nbinsx=70,
                showlegend=(row == 1 and col == 1),
            ),
            row=row,
            col=col,
        )

    # Observed is coloured by its arm, the model (PPC) is the accent
    # everywhere, so the legend is three entries: one per observed arm plus
    # the model.
    _add_hist_pair(
        1, 1, obs_sa, ppc_sa, f"Observed - {lab_a}", "Model (PPC)",
        colour_a, ACCENT,
    )
    _add_hist_pair(
        1, 2, obs_sb, ppc_sb, f"Observed - {lab_b}", "Model (PPC)",
        colour_b, ACCENT,
    )
    _add_hist_pair(
        2, 1, obs_pa, ppc_pa, f"Observed - {lab_a}", "Model (PPC)",
        colour_a, ACCENT,
    )
    _add_hist_pair(
        2, 2, obs_pb, ppc_pb, f"Observed - {lab_b}", "Model (PPC)",
        colour_b, ACCENT,
    )

    if pcrma is not None and len(pcrma):
        fig.add_trace(
            go.Histogram(
                x=pcrma,
                nbinsx=45,
                name="PPC draw mean conversion",
                marker_color=ACCENT,
                opacity=0.55,
                showlegend=False,
            ),
            row=3,
            col=1,
        )
    if ocra is not None and np.isfinite(ocra):
        fig.add_vline(
            x=ocra,
            line_dash="dash",
            line_color=TEXT,
            annotation_text=f"Observed {ocra * 100:.2f}%",
            annotation_position="bottom right",
            row=3,
            col=1,
        )

    if pcrmb is not None and len(pcrmb):
        fig.add_trace(
            go.Histogram(
                x=pcrmb,
                nbinsx=45,
                name="PPC draw mean conversion",
                marker_color=ACCENT,
                opacity=0.55,
                showlegend=False,
            ),
            row=3,
            col=2,
        )
    if ocrb is not None and np.isfinite(ocrb):
        fig.add_vline(
            x=ocrb,
            line_dash="dash",
            line_color=TEXT,
            annotation_text=f"Observed {ocrb * 100:.2f}%",
            annotation_position="bottom right",
            row=3,
            col=2,
        )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        barmode="overlay",
        height=1000,
        title=dict(
            text="Posterior predictive check - observed vs model-simulated data",
            font=dict(size=14),
        ),
        margin=dict(t=58, b=92, l=54, r=36),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.07,
            x=0.5,
            xanchor="center",
        ),
    )

    for r in (1, 2):
        for c in (1, 2):
            fig.update_xaxes(
                title_text="Spend ($)",
                range=[0, spend_xmax],
                row=r,
                col=c,
            )
            fig.update_yaxes(title_text="Density", row=r, col=c)

    fig.update_xaxes(title_text="Conversion fraction", tickformat=".1%", row=3, col=1)
    fig.update_xaxes(title_text="Conversion fraction", tickformat=".1%", row=3, col=2)
    fig.update_yaxes(title_text="Count", row=3, col=1)
    fig.update_yaxes(title_text="Count", row=3, col=2)

    return fig


def update_diagnostics_table(pair_key):
    b = BAYESIAN[pair_key]
    diag_table = b.get("diagnostics_table", [])

    if not diag_table:
        return "No diagnostics available"

    header = html.Tr(
        [
            html.Th(
                "Parameter",
                style={"fontFamily": MONO, "fontSize": "0.75rem"},
            ),
            html.Th(
                "R̂",
                style={"fontFamily": MONO, "fontSize": "0.75rem"},
            ),
            html.Th(
                "Bulk ESS",
                style={"fontFamily": MONO, "fontSize": "0.75rem"},
            ),
            html.Th(
                "Tail ESS",
                style={"fontFamily": MONO, "fontSize": "0.75rem"},
            ),
        ]
    )

    rows = []
    for row in diag_table:
        rhat_color = (
            SUCCESS
            if row["r_hat"] < 1.05
            else WARNING
            if row["r_hat"] < 1.1
            else DANGER
        )
        rows.append(
            html.Tr(
                [
                    html.Td(
                        row["parameter"],
                        style={
                            "fontFamily": MONO,
                            "fontSize": "0.8rem",
                        },
                    ),
                    html.Td(
                        f"{row['r_hat']:.3f}",
                        style={
                            "fontFamily": MONO,
                            "fontSize": "0.8rem",
                            "color": rhat_color,
                        },
                    ),
                    html.Td(
                        f"{row['ess_bulk']:.0f}",
                        style={
                            "fontFamily": MONO,
                            "fontSize": "0.8rem",
                        },
                    ),
                    html.Td(
                        f"{row['ess_tail']:.0f}",
                        style={
                            "fontFamily": MONO,
                            "fontSize": "0.8rem",
                        },
                    ),
                ]
            )
        )

    table = dbc.Table(
        [html.Thead(header), html.Tbody(rows)],
        bordered=False,
        size="sm",
        style={"marginBottom": 0},
    )

    return table


def register_bayesian_callbacks(app):
    app.callback(
        Output("bayes-kpi-cards", "children"),
        Output("bayes-posterior-plot", "figure"),
        Output("rope-result-card", "children"),
        Input("bayes-pair-selector", "value"),
        Input("rope-input", "value"),
    )(update_bayesian)

    # Figure / table updates fire on pair-selector change so the content inside
    # each collapse stays current whether the panel is open or closed. The
    # button-click toggles for the collapses on this tab are registered
    # centrally in `callbacks/__init__.py`.
    app.callback(
        Output("bayes-ppc-plot", "figure"),
        Input("bayes-pair-selector", "value"),
    )(update_ppc_figure)
    app.callback(
        Output("bayes-diagnostics-table", "children"),
        Input("bayes-pair-selector", "value"),
    )(update_diagnostics_table)
