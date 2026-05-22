
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html, Output, Input
import dash_bootstrap_components as dbc
from dashboard.theme import *
from dashboard.theme import hex_to_rgba
from dashboard.data import BAYESIAN
from layouts.components import kpi_card

def update_bayesian(pair_key, rope_val):
    rope_val = rope_val if rope_val is not None else 0  # Dash returns None for empty number inputs
    b = BAYESIAN[pair_key]
    delta = b["delta_samples"]

    hdi_str = f"95% HDI: ${b['hdi_lo']:.2f} - ${b['hdi_hi']:.2f}"
    p_pos = b["p_positive"]
    arm_color = MENS_COLOUR if pair_key.startswith("mens") else WOMENS_COLOUR

    kpis = html.Div(
        [
            kpi_card(
                hdi_str,
                f"Treatment effect: {b['arm_a_label']} vs {b['arm_b_label']}",
                accent=arm_color,
                info=(
                    "The 95% Highest Density Interval is the narrowest range that holds 95% of "
                    "the posterior. Given the model and the data, there's a 95% probability the "
                    "true per-customer effect sits inside it."
                ),
                info_id="bayes-kpi-hdi-info",
            ),
            kpi_card(
                f"{p_pos:.1%}",
                "P(effect > 0)",
                color=SUCCESS if p_pos > 0.9 else WARNING,
                accent=SUCCESS if p_pos > 0.9 else WARNING,
                info=(
                    "The probability that the true per-customer effect is positive. Above 95% is "
                    "strong evidence of a lift. Near 50% means the data can't tell which way the "
                    "effect points."
                ),
                info_id="bayes-kpi-ppos-info",
            ),
            kpi_card(
                f"${b['mean_a']:.2f}",
                f"Mean spend - {b['arm_a_label']}",
                accent=arm_color,
                info=(
                    "The model's estimate of the average per-customer spend for this arm, "
                    "combining the chance of any spend with the size of the spend when it happens."
                ),
                info_id="bayes-kpi-meana-info",
            ),
            kpi_card(
                f"${b['mean_b']:.2f}",
                f"Mean spend - {b['arm_b_label']}",
                accent=CTRL_COLOUR,
                info=(
                    "The model's estimate of the average per-customer spend for this arm, "
                    "combining the chance of any spend with the size of the spend when it happens."
                ),
                info_id="bayes-kpi-meanb-info",
            ),
        ]
    )

    counts, bin_edges = np.histogram(delta, bins=120, density=True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    posterior_fig = go.Figure()

    rope_mask = (bin_centres >= -rope_val) & (bin_centres <= rope_val)
    if rope_mask.any():
        posterior_fig.add_trace(
            go.Scatter(
                x=np.concatenate(
                    [
                        [bin_centres[rope_mask][0]],
                        bin_centres[rope_mask],
                        [bin_centres[rope_mask][-1]],
                    ]
                ),
                y=np.concatenate([[0], counts[rope_mask], [0]]),
                fill="toself",
                fillcolor="rgba(217,119,6,0.2)",
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
            x=bin_centres,
            y=counts,
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
        annotation_text="HDI 2.5%",
        annotation_position="top left",
        annotation_font_color=MUTED
    )
    posterior_fig.add_vline(
        x=b["hdi_hi"],
        line_dash="dash",
        line_color=MUTED,
        annotation_text="HDI 97.5%",
        annotation_position="top right",
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
                              style={"fontSize": "2rem", "fontFamily": "Ubuntu, sans-serif",
                                     "fontWeight": "700", "letterSpacing": "-0.02em",
                                     "color": rope_color, "lineHeight": "1"}),
                    html.Span(
                        [
                            f"  P(|δ| > ${rope_val})",
                            html.Span(" ⓘ", id="bayes-rope-info",
                                      style={"cursor": "help", "color": MUTED,
                                             "fontSize": "0.85em", "marginLeft": "4px"}),
                        ],
                        style={"fontSize": "0.78rem", "color": MUTED,
                               "fontFamily": "Ubuntu, sans-serif",
                               "letterSpacing": "0.01em",
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
                style={"fontFamily": "Ubuntu, sans-serif", "fontSize": "0.8rem"},
            ),
        ],
        style={"borderLeft": f"3px solid {rope_color}", "paddingLeft": "0.9rem"},
    )

    return kpis, posterior_fig, rope_card

def update_trace_figure(pair_key):
    b = BAYESIAN[pair_key]
    delta_chains = b["delta_chains"]

    fig = go.Figure()
    colors = [MENS_COLOUR, WOMENS_COLOUR]
    for i, chain in enumerate(delta_chains):
        fig.add_trace(
            go.Scatter(
                x=list(range(len(chain))),
                y=chain,
                mode="lines",
                name=f"Chain {i + 1}",
                line=dict(color=colors[i % len(colors)], width=0.8),
                opacity=0.8,
                hovertemplate="Draw %{x}<br>δ: $%{y:.2f}<extra>Chain %{fullData.name}</extra>",
            )
        )

    rhat = b.get("rhat_delta", "N/A")
    bulk_ess = b.get("bulk_ess_delta", "N/A")
    tail_ess = b.get("tail_ess_delta", "N/A")

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="MCMC Trace: δ (treatment effect)",
        xaxis_title="Draw",
        yaxis_title="δ value",
        margin=dict(t=50, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        annotations=[
            dict(
                x=1.02,
                y=1.0,
                xref="paper",
                yref="paper",
                text=f"<b>MCMC Diagnostics</b><br>R̂: {rhat}<br>Bulk ESS: {bulk_ess}<br>Tail ESS: {tail_ess}",
                showarrow=False,
                font=dict(family="Ubuntu Mono, monospace", size=10, color=MUTED),
                align="left",
                bgcolor=SURFACE,
                bordercolor=BORDER,
                borderwidth=1,
                borderpad=6,
            )
        ],
    )
    return fig


def update_ppc_figure(pair_key):
    b = BAYESIAN[pair_key]
    lab_a = b["arm_a_label"]
    lab_b = b["arm_b_label"]
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

    nd = pack.get("ppc_n_draws", "—")
    nf = pack.get("ppc_n_fake_per_draw", "—")
    nca = pack.get("ppc_conv_n_a")
    ncb = pack.get("ppc_conv_n_b")
    nca_s = f"{nca:,}" if nca is not None else "—"
    ncb_s = f"{ncb:,}" if ncb is not None else "—"

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            f"Full spend ·{lab_a} ({nf} synthetic draws / PPC slice)",
            f"Full spend ·{lab_b} ({nf} synthetic draws / PPC slice)",
            f"Amount | spend > 0 ·{lab_a} (same {nf}-draw slices)",
            f"Amount | spend > 0 ·{lab_b} (same {nf}-draw slices)",
            f"Conversion ·{lab_a} (PPC batches of n = {nca_s})",
            f"Conversion ·{lab_b} (PPC batches of n = {ncb_s})",
        ),
        vertical_spacing=0.09,
        horizontal_spacing=0.07,
    )

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
    # row-1 zero spike + row-2 mass become unreadable. Clipping gives the user
    # a view of the body and the model-vs-observed fit there.
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
                showlegend=(row == 1 and col == 1),
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

    _add_hist_pair(
        1, 1, obs_sa, ppc_sa, f"Observed:{lab_a}", f"PPC mimic:{lab_a}",
        MENS_COLOUR, ACCENT,
    )
    _add_hist_pair(
        1, 2, obs_sb, ppc_sb, f"Observed:{lab_b}", f"PPC mimic:{lab_b}",
        WOMENS_COLOUR, CTRL_COLOUR,
    )
    _add_hist_pair(
        2, 1, obs_pa, ppc_pa, f"Obs converters:{lab_a}", f"PPC | spend>0:{lab_a}",
        MENS_COLOUR, ACCENT,
    )
    _add_hist_pair(
        2, 2, obs_pb, ppc_pb, f"Obs converters:{lab_b}", f"PPC | spend>0:{lab_b}",
        WOMENS_COLOUR, CTRL_COLOUR,
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
                marker_color=CTRL_COLOUR,
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
        height=980,
        title=(
            "Posterior predictive check · "
            f"rows 1 and 2 simulate {nf} customers per posterior draw. "
            f"Row 3 simulates at the full arm size "
            f"(n={nca_s}, n={ncb_s}) so the conversion-rate spread matches reality. "
            f"({nd} posterior draws total.)"
        ),
        margin=dict(t=80, b=46, l=54, r=36),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
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

    fig.update_xaxes(title_text="Conversion fraction", tickformat=".0%", row=3, col=1)
    fig.update_xaxes(title_text="Conversion fraction", tickformat=".0%", row=3, col=2)
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
                style={"fontFamily": "Ubuntu Mono, monospace", "fontSize": "0.75rem"},
            ),
            html.Th(
                "R̂",
                style={"fontFamily": "Ubuntu Mono, monospace", "fontSize": "0.75rem"},
            ),
            html.Th(
                "Bulk ESS",
                style={"fontFamily": "Ubuntu Mono, monospace", "fontSize": "0.75rem"},
            ),
            html.Th(
                "Tail ESS",
                style={"fontFamily": "Ubuntu Mono, monospace", "fontSize": "0.75rem"},
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
                            "fontFamily": "Ubuntu Mono, monospace",
                            "fontSize": "0.8rem",
                        },
                    ),
                    html.Td(
                        f"{row['r_hat']:.3f}",
                        style={
                            "fontFamily": "Ubuntu Mono, monospace",
                            "fontSize": "0.8rem",
                            "color": rhat_color,
                        },
                    ),
                    html.Td(
                        f"{row['ess_bulk']:.0f}",
                        style={
                            "fontFamily": "Ubuntu Mono, monospace",
                            "fontSize": "0.8rem",
                        },
                    ),
                    html.Td(
                        f"{row['ess_tail']:.0f}",
                        style={
                            "fontFamily": "Ubuntu Mono, monospace",
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
        Input("rope-slider", "value"),
    )(update_bayesian)

    # Figure / table updates fire on pair-selector change so the content inside
    # each collapse stays current whether the panel is open or closed. The
    # button-click toggles for the four collapses on this tab are registered
    # centrally in `callbacks/__init__.py`.
    app.callback(
        Output("bayes-trace-plot", "figure"),
        Input("bayes-pair-selector", "value"),
    )(update_trace_figure)
    app.callback(
        Output("bayes-ppc-plot", "figure"),
        Input("bayes-pair-selector", "value"),
    )(update_ppc_figure)
    app.callback(
        Output("bayes-diagnostics-table", "children"),
        Input("bayes-pair-selector", "value"),
    )(update_diagnostics_table)
