"""
app.py
------
Causal Inference Dashboard — Hillstrom Email Campaign Dataset
6-tab Dash app with dark CYBORG theme. All heavy computation is cached to disk.
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, dash_table, Input, Output, State, callback
import dash_bootstrap_components as dbc

import causal_utils as cu

# ---------------------------------------------------------------------------
# Initialise app & load data
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    title="Causal Inference Dashboard",
)
server = app.server

print("=" * 60)
print("Causal Inference Dashboard — Hillstrom Dataset")
print("=" * 60)
RESULTS = cu.load_or_build_cache()
DF = RESULTS["df"]
PSM = RESULTS["psm"]
BAYESIAN = RESULTS["bayesian"]
UPLIFT = RESULTS["uplift"]
OLS = RESULTS["ols"]
print("[App] Data loaded. Starting server...")

# ---------------------------------------------------------------------------
# Shared style constants
# ---------------------------------------------------------------------------

PLOTLY_TEMPLATE = "plotly_dark"
CARD_STYLE = {"border": "1px solid #444", "borderRadius": "8px"}
KPI_LABEL_STYLE = {"fontSize": "0.75rem", "color": "#aaa", "textTransform": "uppercase", "letterSpacing": "0.05em"}
KPI_VALUE_STYLE = {"fontSize": "1.8rem", "fontWeight": "700", "color": "#fff"}
KPI_DELTA_STYLE = {"fontSize": "0.85rem"}
SECTION_HEADER = {"borderBottom": "1px solid #444", "paddingBottom": "0.5rem", "marginBottom": "1rem"}

COVARIATE_LABELS = {
    "recency": "Recency (months)",
    "history": "History ($)",
    "mens": "Mens catalogue",
    "womens": "Womens catalogue",
    "zip_suburban": "Zip: Suburban",
    "zip_rural": "Zip: Rural",
    "channel_web": "Channel: Web",
    "channel_multichannel": "Channel: Multichannel",
    "newbie": "New customer",
}


def kpi_card(value, label, delta=None, delta_positive=None, color="#fff"):
    delta_color = "#28a745" if delta_positive else "#dc3545" if delta_positive is False else "#aaa"
    return dbc.Card(
        dbc.CardBody([
            html.P(label, style=KPI_LABEL_STYLE),
            html.P(value, style={**KPI_VALUE_STYLE, "color": color}),
            html.P(delta or "", style={**KPI_DELTA_STYLE, "color": delta_color}),
        ]),
        style=CARD_STYLE,
        className="mb-2",
    )


def methodology_collapse(tab_id, content):
    return html.Div([
        dbc.Button(
            "📖 Methodology & Assumptions",
            id=f"method-btn-{tab_id}",
            color="secondary",
            outline=True,
            size="sm",
            className="mb-2",
        ),
        dbc.Collapse(
            dbc.Card(dbc.CardBody(content, style={"fontSize": "0.88rem", "color": "#ccc"})),
            id=f"method-collapse-{tab_id}",
            is_open=False,
        ),
    ])


# ---------------------------------------------------------------------------
# Tab layouts
# ---------------------------------------------------------------------------

# ── Tab 1: Overview ──────────────────────────────────────────────────────────

def tab1_layout():
    seg_counts = DF["segment"].value_counts()
    n_mens = seg_counts.get("Mens E-Mail", 0)
    n_womens = seg_counts.get("Womens E-Mail", 0)
    n_control = seg_counts.get("No E-Mail", 0)

    conv_mens = DF[DF["segment"] == "Mens E-Mail"]["conversion"].mean() * 100
    conv_womens = DF[DF["segment"] == "Womens E-Mail"]["conversion"].mean() * 100
    conv_control = DF[DF["segment"] == "No E-Mail"]["conversion"].mean() * 100

    return dbc.Container([
        dbc.Row([
            dbc.Col(kpi_card(f"{n_mens:,}", "Men's Email (treated)", color="#7ec8e3"), md=3),
            dbc.Col(kpi_card(f"{n_womens:,}", "Women's Email (treated)", color="#f9a8d4"), md=3),
            dbc.Col(kpi_card(f"{n_control:,}", "Control (no email)", color="#86efac"), md=3),
            dbc.Col(kpi_card(f"{len(DF):,}", "Total customers", color="#fde68a"), md=3),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(kpi_card(f"{conv_mens:.1f}%", "Conversion — Men's", f"vs control: +{conv_mens - conv_control:.1f}pp", conv_mens > conv_control), md=4),
            dbc.Col(kpi_card(f"{conv_womens:.1f}%", "Conversion — Women's", f"vs control: +{conv_womens - conv_control:.1f}pp", conv_womens > conv_control), md=4),
            dbc.Col(kpi_card(f"{conv_control:.1f}%", "Conversion — Control"), md=4),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                html.H5("Spend Distribution by Segment", style=SECTION_HEADER),
                dcc.Graph(id="tab1-violin", figure=_fig_violin()),
            ], md=8),
            dbc.Col([
                html.H5("About this Dataset", style=SECTION_HEADER),
                html.P(
                    "The Hillstrom MineThatData dataset captures a randomised email marketing "
                    "experiment across 64,000 US retail customers. Two treatment arms received "
                    "targeted email campaigns (Men's or Women's catalogue), while the control "
                    "group received nothing.",
                    className="small text-muted",
                ),
                html.P(
                    "The causal question: does receiving an email cause customers to spend more? "
                    "Even in a randomised experiment, causal analysis adds value by quantifying "
                    "uncertainty, identifying which customers respond most (HTE), and stress-testing "
                    "results across methodologies.",
                    className="small text-muted",
                ),
                html.P(
                    "Outcome of interest: spend ($) in the two weeks following the campaign.",
                    className="small text-muted",
                ),
            ], md=4),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                html.H5("Covariate Balance — Standardised Mean Differences (vs Control)", style=SECTION_HEADER),
                html.P(
                    "SMD measures how many pooled standard deviations separate each arm from control. "
                    "Values within ±0.1 (dashed lines) indicate good balance — expected in a randomised experiment.",
                    className="small text-muted mb-2",
                ),
                dcc.Graph(id="tab1-balance", figure=_fig_covariate_balance()),
            ]),
        ]),

        methodology_collapse("tab1", [
            html.P(
                "This tab presents descriptive statistics. Because this is a randomised experiment, "
                "groups should be broadly balanced on observed covariates — the balance table confirms this. "
                "However, minor imbalances can still bias estimates, motivating the matching and regression "
                "adjustments in subsequent tabs."
            ),
        ]),
    ], fluid=True, className="py-4")


def _fig_violin():
    # Most customers have spend=0; show spenders only (spend > 0) so the
    # distribution is visible, plus annotate the zero-spend % per segment.
    spenders = DF[DF["spend"] > 0].copy()
    # Cap at 99th percentile of spenders to remove extreme outliers
    cap = spenders["spend"].quantile(0.99)
    spenders = spenders[spenders["spend"] <= cap]

    seg_order = ["Mens E-Mail", "Womens E-Mail", "No E-Mail"]
    color_map = {
        "Mens E-Mail": "#7ec8e3",
        "Womens E-Mail": "#f9a8d4",
        "No E-Mail": "#86efac",
    }

    fig = go.Figure()
    for seg in seg_order:
        vals = spenders[spenders["segment"] == seg]["spend"].values
        pct_zero = (DF[DF["segment"] == seg]["spend"] == 0).mean() * 100
        fig.add_trace(go.Violin(
            y=vals,
            name=f"{seg}<br><span style='font-size:10px'>{pct_zero:.0f}% zero-spend</span>",
            box_visible=True,
            meanline_visible=True,
            points=False,
            fillcolor=color_map[seg],
            line_color=color_map[seg],
            opacity=0.7,
        ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        showlegend=False,
        margin=dict(t=30, b=20),
        yaxis_title="Spend ($) — among spenders only",
        xaxis_title="",
        violingap=0.3,
    )
    return fig


def _fig_covariate_balance():
    """
    Standardised Mean Difference (SMD) dot plot comparing each treatment arm
    to the control group. SMD = (mean_treated - mean_control) / pooled_std.
    Values near 0 indicate good balance; |SMD| < 0.1 is the conventional threshold.
    """
    covs = cu.COVARIATES
    control = DF[DF["segment"] == "No E-Mail"]
    arms = {
        "Men's Email": DF[DF["segment"] == "Mens E-Mail"],
        "Women's Email": DF[DF["segment"] == "Womens E-Mail"],
    }
    colors = {"Men's Email": "#f97316", "Women's Email": "#22d3ee"}
    symbols = {"Men's Email": "diamond", "Women's Email": "circle"}

    fig = go.Figure()
    for arm_label, arm_df in arms.items():
        smds = []
        labels = []
        for cov in covs:
            a = arm_df[cov].values
            b = control[cov].values
            pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
            smd = (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0
            smds.append(smd)
            labels.append(COVARIATE_LABELS.get(cov, cov))

        fig.add_trace(go.Scatter(
            x=smds,
            y=labels,
            mode="markers",
            name=arm_label,
            marker=dict(color=colors[arm_label], size=11, symbol=symbols[arm_label]),
        ))

    fig.add_vline(x=0.1, line_dash="dash", line_color="#facc15",
                  annotation_text="0.1", annotation_position="top right")
    fig.add_vline(x=-0.1, line_dash="dash", line_color="#facc15",
                  annotation_text="-0.1", annotation_position="top left")
    fig.add_vline(x=0, line_color="#555", line_width=1)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        margin=dict(t=30, b=60, l=160, r=40),
        xaxis_title="Standardised Mean Difference (vs Control)",
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        height=340,
    )
    return fig


# ── Tab 2: PSM ───────────────────────────────────────────────────────────────

def tab2_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Label("Campaign arm:", className="small text-muted"),
                dbc.RadioItems(
                    id="psm-arm-selector",
                    options=[
                        {"label": "Men's Email vs Control", "value": "mens"},
                        {"label": "Women's Email vs Control", "value": "womens"},
                    ],
                    value="mens",
                    inline=True,
                    className="mb-3",
                ),
            ]),
        ]),

        dbc.Row([
            dbc.Col(html.Div(id="psm-kpi-cards"), md=4),
            dbc.Col(dcc.Graph(id="psm-ps-dist"), md=8),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col(dcc.Graph(id="psm-love-plot"), md=7),
            dbc.Col(dcc.Graph(id="psm-stats-chart"), md=5),
        ], className="mb-3"),

        methodology_collapse("tab2", [
            html.P(
                "Propensity Score Matching (PSM) estimates the Average Treatment Effect on the Treated (ATT) "
                "by matching each treated customer to a control customer with a similar probability of "
                "treatment (propensity score). The propensity score is estimated via logistic regression "
                "on all observed covariates."
            ),
            html.P(
                "Key assumptions: (1) Ignorability — all confounders are observed and included in the "
                "propensity model. (2) Common support — treated and control overlap in propensity scores. "
                "(3) SUTVA — no spillover between customers. The Love Plot shows covariate balance "
                "before and after matching; standardised mean differences below 0.1 indicate good balance."
            ),
            html.P(
                "Uncertainty is quantified via 500 bootstrap samples of the matched dataset. "
                "The 95% CI shown is the 2.5th–97.5th percentile of bootstrap ATT estimates."
            ),
        ]),
    ], fluid=True, className="py-4")


# ── Tab 3: Bayesian A/B ───────────────────────────────────────────────────────

def tab3_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Label("Comparison:", className="small text-muted"),
                dbc.RadioItems(
                    id="bayes-pair-selector",
                    options=[
                        {"label": "Men's vs Control", "value": "mens_vs_control"},
                        {"label": "Women's vs Control", "value": "womens_vs_control"},
                        {"label": "Men's vs Women's", "value": "mens_vs_womens"},
                    ],
                    value="mens_vs_control",
                    inline=True,
                    className="mb-3",
                ),
            ]),
        ]),

        dbc.Row([
            dbc.Col(html.Div(id="bayes-kpi-cards"), md=4),
            dbc.Col(dcc.Graph(id="bayes-posterior-plot"), md=8),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([
                html.Label("ROPE — minimum meaningful effect ($):", className="small text-muted"),
                dbc.InputGroup([
                    dbc.InputGroupText("$", style={"backgroundColor": "#2a2a3e", "color": "#aaa", "border": "1px solid #444"}),
                    dbc.Input(
                        id="rope-slider",
                        type="number",
                        min=0, max=10, step=0.5, value=1,
                        debounce=True,
                        style={"backgroundColor": "#1e1e2e", "color": "#e2e8f0", "border": "1px solid #444"},
                    ),
                    dbc.InputGroupText("per customer", style={"backgroundColor": "#2a2a3e", "color": "#aaa", "border": "1px solid #444"}),
                ], size="sm", className="mt-1"),
            ], md=6),
            dbc.Col(html.Div(id="rope-result-card"), md=6),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([
                dbc.Button(
                    "📊 Show MCMC Trace Plots",
                    id="trace-btn",
                    color="secondary",
                    outline=True,
                    size="sm",
                    className="mb-2",
                ),
                dbc.Collapse(
                    dcc.Graph(id="bayes-trace-plot"),
                    id="trace-collapse",
                    is_open=False,
                ),
            ]),
        ]),

        methodology_collapse("tab3", [
            html.P(
                "The Bayesian A/B test models spend as a Normal distribution for each group, "
                "with weakly informative priors centred on the pooled mean. The treatment effect "
                "(delta) is the difference in posterior means. MCMC is run with PyMC (2,000 draws, "
                "2 chains, fixed seed=42)."
            ),
            html.P(
                "The 95% Highest Density Interval (HDI) is the shortest interval containing 95% of the "
                "posterior probability — unlike a frequentist CI, this has a direct probabilistic "
                "interpretation: we believe there is a 95% probability the true effect lies in this range."
            ),
            html.P(
                "The ROPE (Region of Practical Equivalence) lets you define a minimum effect size that "
                "matters for business decisions. The dashboard shows the probability mass outside the ROPE."
            ),
        ]),
    ], fluid=True, className="py-4")


# ── Tab 4: Uplift / HTE ───────────────────────────────────────────────────────

def tab4_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Label("Campaign arm:", className="small text-muted"),
                dbc.RadioItems(
                    id="uplift-arm-selector",
                    options=[
                        {"label": "Men's Email", "value": "mens"},
                        {"label": "Women's Email", "value": "womens"},
                    ],
                    value="mens",
                    inline=True,
                    className="mb-2",
                ),
            ], md=6),
            dbc.Col([
                html.Label("Model:", className="small text-muted"),
                dbc.RadioItems(
                    id="uplift-model-selector",
                    options=[
                        {"label": "T-Learner", "value": "t"},
                        {"label": "S-Learner", "value": "s"},
                    ],
                    value="t",
                    inline=True,
                    className="mb-2",
                ),
            ], md=6),
        ]),

        dbc.Row([
            dbc.Col(html.Div(id="uplift-kpi-cards"), md=4),
            dbc.Col(dcc.Graph(id="uplift-cate-hist"), md=8),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col(dcc.Graph(id="uplift-feat-imp"), md=6),
            dbc.Col(dcc.Graph(id="uplift-decile-chart"), md=6),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col(dcc.Graph(id="uplift-qini"), md=6),
            dbc.Col(dcc.Graph(id="uplift-segment-compare"), md=6),
        ], className="mb-3"),

        methodology_collapse("tab4", [
            html.P(
                "Uplift modelling estimates Conditional Average Treatment Effects (CATE): the expected "
                "causal effect for each individual customer, given their characteristics."
            ),
            html.P(
                "T-Learner trains two separate models — one on treated customers, one on control — "
                "and computes CATE as the difference in predictions. "
                "S-Learner trains a single model with treatment as a feature and computes CATE "
                "by differencing predictions under treatment=1 vs treatment=0."
            ),
            html.P(
                "Both models use 5-fold cross-fitting: each observation's CATE is predicted by a "
                "model trained on the other four folds. This avoids in-sample overfitting and gives "
                "honest out-of-sample estimates — the same principle as Double ML / cross-fitting."
            ),
            html.P(
                "The Qini curve ranks customers by predicted uplift; a higher area under the curve "
                "indicates the model successfully identifies high-responders. "
                "The decile chart shows actual spend lift for customers ranked by predicted uplift — "
                "good models show declining lift across deciles."
            ),
            html.P(
                "Key assumption: the same ignorability assumption as PSM — all relevant confounders "
                "are observed. HTE estimates have wider uncertainty than ATE estimates and should be "
                "treated as directional rather than precise."
            ),
        ]),
    ], fluid=True, className="py-4")


# ── Tab 5: Multi-Arm OLS ─────────────────────────────────────────────────────

def tab5_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H5("OLS Coefficient Plot — Treatment Effects & Interactions", style=SECTION_HEADER),
                dcc.Graph(id="ols-coef-plot"),
            ]),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([
                html.H5("Marginal Effects by Subgroup", style=SECTION_HEADER),
                html.Div(id="ols-marginal-table"),
            ]),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([
                html.H5("Subgroup Heatmap — Predicted Spend Lift", style=SECTION_HEADER),
            ]),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id="ols-heatmap-mens"), md=6),
            dbc.Col(dcc.Graph(id="ols-heatmap-womens"), md=6),
        ]),

        methodology_collapse("tab5", [
            html.P(
                "OLS regression estimates population-average treatment effects while controlling for "
                "observed covariates. All three arms (Men's, Women's, Control) are compared simultaneously "
                "via treatment dummy variables, avoiding the multiple comparison inflation of separate tests."
            ),
            html.P(
                "Interaction terms (treatment × newbie, treatment × channel, treatment × zip_code) capture "
                "treatment effect heterogeneity at the subgroup level — a parametric complement to the "
                "non-parametric uplift models in Tab 4."
            ),
            html.P(
                "Key assumption: linearity of the conditional expectation function. The model is estimated "
                "by OLS (not IV or 2SLS) and relies on the same ignorability assumption as PSM."
            ),
        ]),
    ], fluid=True, className="py-4")


# ── Tab 6: Method Comparison ──────────────────────────────────────────────────

def tab6_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H5("All Methods Summary", style=SECTION_HEADER),
                html.Div(id="comparison-table"),
            ]),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col(dcc.Graph(id="forest-plot-mens"), md=6),
            dbc.Col(dcc.Graph(id="forest-plot-womens"), md=6),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                html.Div(id="key-takeaway-card"),
            ], md=5),
            dbc.Col([
                dbc.Accordion([
                    dbc.AccordionItem([
                        html.P("Estimates the ATT via covariate balancing. Best when you believe all confounders are observed. Sensitive to propensity model misspecification."),
                    ], title="Propensity Score Matching (PSM)"),
                    dbc.AccordionItem([
                        html.P("Provides a full posterior distribution over the treatment effect. Best for communicating uncertainty and making probabilistic decisions. Requires distributional assumptions (Normal likelihood here)."),
                    ], title="Bayesian A/B Test"),
                    dbc.AccordionItem([
                        html.P("Estimates individual-level CATEs — ideal for targeting campaigns to the most responsive customers. Higher variance than ATE estimates; treat as directional."),
                    ], title="Uplift / HTE (T-Learner, S-Learner)"),
                    dbc.AccordionItem([
                        html.P("Efficient parametric estimate with explicit interaction terms. Assumes linearity. Best for understanding which subgroups drive heterogeneity in a transparent, auditable way."),
                    ], title="Multi-Arm OLS"),
                ], start_collapsed=True),
            ], md=7),
        ]),
    ], fluid=True, className="py-4")


# ---------------------------------------------------------------------------
# App Layout
# ---------------------------------------------------------------------------

TABS = [
    dbc.Tab(tab1_layout(), label="1 · Overview", tab_id="tab-1"),
    dbc.Tab(tab2_layout(), label="2 · PSM", tab_id="tab-2"),
    dbc.Tab(tab3_layout(), label="3 · Bayesian A/B", tab_id="tab-3"),
    dbc.Tab(tab4_layout(), label="4 · Uplift / HTE", tab_id="tab-4"),
    dbc.Tab(tab5_layout(), label="5 · Multi-Arm OLS", tab_id="tab-5"),
    dbc.Tab(tab6_layout(), label="6 · Method Comparison", tab_id="tab-6"),
]

app.layout = html.Div([
    # Sticky navbar
    dbc.Navbar(
        dbc.Container([
            html.Span("Causal Inference Dashboard", className="navbar-brand fw-bold"),
            html.Span(["Hillstrom Email Campaign ", html.I("Hillstrom (2008)")], className="text-muted small ms-3"),
        ], fluid=True),
        color="dark",
        dark=True,
        sticky="top",
        className="mb-0 border-bottom border-secondary",
    ),

    dbc.Container([
        dbc.Tabs(TABS, id="main-tabs", active_tab="tab-1"),
    ], fluid=True),
])


# ===========================================================================
# CALLBACKS — Tab 2: PSM
# ===========================================================================

@app.callback(
    Output("psm-kpi-cards", "children"),
    Output("psm-ps-dist", "figure"),
    Output("psm-love-plot", "figure"),
    Output("psm-stats-chart", "figure"),
    Input("psm-arm-selector", "value"),
)
def update_psm(arm):
    p = PSM[arm]
    arm_label = "Men's Email" if arm == "mens" else "Women's Email"

    # KPI cards
    ci_str = f"95% CI: ${p['att_ci_lo']:.2f} — ${p['att_ci_hi']:.2f}"
    pct_str = f"{p['pct_matched']:.1f}% of treated matched"
    cs_str = f"[{p['cs_lower']:.3f}, {p['cs_upper']:.3f}]"

    kpis = html.Div([
        kpi_card(f"${p['att_point']:.2f}", f"ATT — {arm_label}", ci_str, p["att_point"] > 0),
        kpi_card(f"{p['n_matched']:,}", "Matched pairs", pct_str),
        kpi_card(cs_str, "Common support range"),
    ])

    # Propensity score distribution
    ps_fig = go.Figure()
    ps_fig.add_trace(go.Histogram(
        x=p["propensity_treated"], name=arm_label,
        opacity=0.65, nbinsx=50,
        marker_color="#7ec8e3" if arm == "mens" else "#f9a8d4",
    ))
    ps_fig.add_trace(go.Histogram(
        x=p["propensity_control"], name="Control",
        opacity=0.65, nbinsx=50,
        marker_color="#86efac",
    ))
    ps_fig.update_layout(
        barmode="overlay",
        template=PLOTLY_TEMPLATE,
        title="Propensity Score Distribution",
        xaxis_title="Propensity score",
        yaxis_title="Count",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(t=50, b=70),
    )

    # Love plot
    covs = list(p["smd_before"].keys())
    cov_labels = [COVARIATE_LABELS.get(c, c) for c in covs]
    smd_before = [p["smd_before"][c] for c in covs]
    smd_after = [p["smd_after"][c] for c in covs]

    love_fig = go.Figure()
    love_fig.add_trace(go.Scatter(
        x=smd_before, y=cov_labels, mode="markers",
        name="Before matching", marker=dict(color="#f97316", size=10, symbol="circle"),
    ))
    love_fig.add_trace(go.Scatter(
        x=smd_after, y=cov_labels, mode="markers",
        name="After matching", marker=dict(color="#22c55e", size=10, symbol="diamond"),
    ))
    love_fig.add_vline(x=0.1, line_dash="dash", line_color="#facc15", annotation_text="0.1 threshold")
    love_fig.add_vline(x=-0.1, line_dash="dash", line_color="#facc15")
    love_fig.add_vline(x=0, line_color="#666")
    love_fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Love Plot — Standardised Mean Differences",
        xaxis_title="Standardised Mean Difference",
        margin=dict(t=50, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
    )

    # Stats bar chart (before/after ATT context)
    stats_fig = go.Figure(go.Bar(
        x=["ATT Estimate", "CI Lower", "CI Upper"],
        y=[p["att_point"], p["att_ci_lo"], p["att_ci_hi"]],
        marker_color=["#7ec8e3", "#64748b", "#64748b"],
        text=[f"${v:.2f}" for v in [p["att_point"], p["att_ci_lo"], p["att_ci_hi"]]],
        textposition="outside",
    ))
    stats_fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="ATT with 95% Bootstrap CI",
        yaxis_title="Effect on Spend ($)",
        margin=dict(t=50, b=30),
    )

    return kpis, ps_fig, love_fig, stats_fig


@app.callback(
    Output("method-collapse-tab2", "is_open"),
    Input("method-btn-tab2", "n_clicks"),
    State("method-collapse-tab2", "is_open"),
    prevent_initial_call=True,
)
def toggle_method_tab2(n, is_open):
    return not is_open


# ===========================================================================
# CALLBACKS — Tab 3: Bayesian A/B
# ===========================================================================

@app.callback(
    Output("bayes-kpi-cards", "children"),
    Output("bayes-posterior-plot", "figure"),
    Output("rope-result-card", "children"),
    Input("bayes-pair-selector", "value"),
    Input("rope-slider", "value"),
)
def update_bayesian(pair_key, rope_val):
    b = BAYESIAN[pair_key]
    delta = b["delta_samples"]

    hdi_str = f"95% HDI: ${b['hdi_lo']:.2f} — ${b['hdi_hi']:.2f}"
    p_pos = b["p_positive"]

    kpis = html.Div([
        kpi_card(hdi_str, f"Treatment effect: {b['arm_a_label']} vs {b['arm_b_label']}", delta=None),
        kpi_card(f"{p_pos:.1%}", "P(effect > 0)", color="#22c55e" if p_pos > 0.9 else "#fbbf24"),
        kpi_card(f"${b['mean_a']:.2f}", f"Mean spend — {b['arm_a_label']}"),
        kpi_card(f"${b['mean_b']:.2f}", f"Mean spend — {b['arm_b_label']}"),
    ])

    # Posterior KDE using a histogram approximation
    counts, bin_edges = np.histogram(delta, bins=120, density=True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    posterior_fig = go.Figure()

    # Shade ROPE region
    rope_mask = (bin_centres >= -rope_val) & (bin_centres <= rope_val)
    if rope_mask.any():
        posterior_fig.add_trace(go.Scatter(
            x=np.concatenate([[bin_centres[rope_mask][0]], bin_centres[rope_mask], [bin_centres[rope_mask][-1]]]),
            y=np.concatenate([[0], counts[rope_mask], [0]]),
            fill="toself",
            fillcolor="rgba(251,191,36,0.2)",
            line=dict(color="rgba(0,0,0,0)"),
            name=f"ROPE ±${rope_val}",
            showlegend=True,
        ))

    posterior_fig.add_trace(go.Scatter(
        x=bin_centres, y=counts,
        mode="lines", fill="tozeroy",
        fillcolor="rgba(126,200,227,0.25)" if "mens_vs_control" in pair_key else "rgba(249,168,212,0.25)",
        line=dict(color="#7ec8e3" if "mens_vs_control" in pair_key or "mens_vs_womens" in pair_key else "#f9a8d4", width=2),
        name="Posterior δ",
    ))

    # HDI lines
    posterior_fig.add_vline(x=b["hdi_lo"], line_dash="dash", line_color="#94a3b8",
                            annotation_text="HDI 2.5%", annotation_position="top left")
    posterior_fig.add_vline(x=b["hdi_hi"], line_dash="dash", line_color="#94a3b8",
                            annotation_text="HDI 97.5%", annotation_position="top right")
    posterior_fig.add_vline(x=0, line_color="#f87171", line_width=1.5, line_dash="dot",
                            annotation_text="Zero", annotation_position="bottom right",
                            annotation_font_color="#f87171")

    posterior_fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"Posterior Distribution — Treatment Effect: {b['arm_a_label']} vs {b['arm_b_label']}",
        xaxis_title="Effect on Spend ($)",
        yaxis_title="Density",
        margin=dict(t=50, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
    )

    # ROPE card
    p_outside_rope = float(np.mean((delta > rope_val) | (delta < -rope_val)))
    rope_card = kpi_card(
        f"{p_outside_rope:.1%}",
        f"P(effect outside ROPE ±${rope_val})",
        color="#22c55e" if p_outside_rope > 0.9 else "#fbbf24",
    )

    return kpis, posterior_fig, rope_card


@app.callback(
    Output("trace-collapse", "is_open"),
    Output("bayes-trace-plot", "figure"),
    Input("trace-btn", "n_clicks"),
    State("trace-collapse", "is_open"),
    State("bayes-pair-selector", "value"),
    prevent_initial_call=True,
)
def toggle_trace(n, is_open, pair_key):
    b = BAYESIAN[pair_key]
    delta_chains = b["delta_chains"]  # (chains, draws)

    fig = go.Figure()
    colors = ["#7ec8e3", "#f9a8d4"]
    for i, chain in enumerate(delta_chains):
        fig.add_trace(go.Scatter(
            x=list(range(len(chain))),
            y=chain,
            mode="lines",
            name=f"Chain {i+1}",
            line=dict(color=colors[i % len(colors)], width=0.8),
            opacity=0.8,
        ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="MCMC Trace — δ (treatment effect)",
        xaxis_title="Draw",
        yaxis_title="δ value",
        margin=dict(t=50, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
    )
    return not is_open, fig


@app.callback(
    Output("method-collapse-tab3", "is_open"),
    Input("method-btn-tab3", "n_clicks"),
    State("method-collapse-tab3", "is_open"),
    prevent_initial_call=True,
)
def toggle_method_tab3(n, is_open):
    return not is_open


# ===========================================================================
# CALLBACKS — Tab 4: Uplift / HTE
# ===========================================================================

@app.callback(
    Output("uplift-kpi-cards", "children"),
    Output("uplift-cate-hist", "figure"),
    Output("uplift-feat-imp", "figure"),
    Output("uplift-decile-chart", "figure"),
    Output("uplift-qini", "figure"),
    Output("uplift-segment-compare", "figure"),
    Input("uplift-arm-selector", "value"),
    Input("uplift-model-selector", "value"),
)
def update_uplift(arm, model):
    u = UPLIFT[arm]
    arm_label = "Men's Email" if arm == "mens" else "Women's Email"
    cate = u["cate_t"] if model == "t" else u["cate_s"]
    model_label = "T-Learner" if model == "t" else "S-Learner"
    color = "#7ec8e3" if arm == "mens" else "#f9a8d4"

    kpis = html.Div([
        kpi_card(f"${np.mean(cate):.2f}", f"Avg CATE ({model_label})", f"{arm_label} vs Control", np.mean(cate) > 0),
        kpi_card(f"${np.percentile(cate, 90):.2f}", "90th percentile CATE", "High-responder threshold"),
        kpi_card(f"{np.mean(cate > 0):.1%}", "% customers with positive uplift"),
    ])

    # CATE histogram — clip display to p1–p99 to avoid extreme outliers
    # stretching the x-axis (the ~1% tail values are real but sparse noise
    # from high-spend customers in individual cross-fitting folds)
    p1, p99 = np.percentile(cate, 1), np.percentile(cate, 99)
    cate_clipped = cate[(cate >= p1) & (cate <= p99)]
    pct_shown = len(cate_clipped) / len(cate) * 100

    hist_fig = go.Figure(go.Histogram(
        x=cate_clipped, nbinsx=60,
        marker_color=color, opacity=0.8,
        name="CATE",
    ))
    hist_fig.add_vline(x=0, line_color="#dc2626", line_dash="dash", annotation_text="Zero")
    hist_fig.add_vline(x=np.mean(cate), line_color="#fbbf24", line_dash="dot", annotation_text="Mean")
    hist_fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"CATE Distribution — {model_label} ({arm_label})",
        xaxis_title=f"Individual Uplift ($)  ·  showing p1–p99 ({pct_shown:.0f}% of customers)",
        yaxis_title="Count",
        margin=dict(t=50, b=30),
    )

    # Feature importance
    feat_imp = dict(sorted(u["feat_imp"].items(), key=lambda x: x[1]))
    feat_labels = [COVARIATE_LABELS.get(k, k) for k in feat_imp.keys()]
    fi_fig = go.Figure(go.Bar(
        x=list(feat_imp.values()),
        y=feat_labels,
        orientation="h",
        marker_color=color,
    ))
    fi_fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Feature Importance (T-Learner treatment model)",
        xaxis_title="Importance",
        margin=dict(t=50, b=30),
    )

    # Decile chart — bars + overlaid markers so every decile has a visible point
    dec_df = pd.DataFrame(u["decile_lift"])
    overall_ate = dec_df["lift"].mean()  # average across deciles ≈ overall ATE

    decile_fig = go.Figure()
    decile_fig.add_trace(go.Bar(
        x=dec_df["decile"],
        y=dec_df["lift"],
        marker_color=[color if v > 0 else "#dc2626" for v in dec_df["lift"]],
        opacity=0.6,
        showlegend=False,
        hovertemplate="Decile %{x}<br>Actual lift: $%{y:.2f}<extra></extra>",
    ))
    decile_fig.add_trace(go.Scatter(
        x=dec_df["decile"],
        y=dec_df["lift"],
        mode="markers+text",
        marker=dict(color=[color if v > 0 else "#dc2626" for v in dec_df["lift"]], size=9, line=dict(color="#fff", width=1)),
        text=[f"${v:.2f}" for v in dec_df["lift"]],
        textposition="top center",
        textfont=dict(size=10),
        showlegend=False,
        hovertemplate="Decile %{x}<br>Actual lift: $%{y:.2f}<extra></extra>",
    ))
    # Random targeting baseline — flat line at the mean lift across deciles
    decile_fig.add_hline(
        y=overall_ate,
        line_dash="dash", line_color="#facc15", line_width=1.5,
        annotation_text=f"Avg lift ${overall_ate:.2f}",
        annotation_position="right",
        annotation_font_color="#facc15",
    )
    decile_fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Actual Spend Lift by Predicted Uplift Decile",
        xaxis=dict(title="Decile (1 = highest predicted uplift)", tickmode="linear", tick0=1, dtick=1),
        yaxis_title="Actual Spend Lift ($)",
        margin=dict(t=50, b=30),
    )

    # Qini curve
    qini_fig = go.Figure()
    qini_fig.add_trace(go.Scatter(
        x=u["qini_x"], y=u["qini_y"],
        mode="lines", name="Model Qini",
        line=dict(color=color, width=2),
        fill="tozeroy", fillcolor=f"rgba({','.join(str(int(c*255)) for c in px.colors.hex_to_rgb(color))},0.15)" if color.startswith("#") else "rgba(126,200,227,0.15)",
    ))
    qini_fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 0], mode="lines",
        name="Random", line=dict(color="#666", dash="dash"),
    ))
    qini_fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Qini Curve",
        xaxis_title="Fraction of population targeted",
        yaxis_title="Cumulative uplift (normalised)",
        margin=dict(t=50, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
    )

    # Segment comparison (avg CATE: mens vs womens, both models)
    seg_fig = go.Figure()
    for a, lbl, col in [("mens", "Men's Email", "#7ec8e3"), ("womens", "Women's Email", "#f9a8d4")]:
        avg_t = UPLIFT[a]["avg_cate_t"]
        avg_s = UPLIFT[a]["avg_cate_s"]
        seg_fig.add_trace(go.Bar(
            name=lbl,
            x=["T-Learner", "S-Learner"],
            y=[avg_t, avg_s],
            marker_color=col,
            opacity=0.85,
        ))

    seg_fig.update_layout(
        barmode="group",
        template=PLOTLY_TEMPLATE,
        title="Average CATE: Men's vs Women's Campaign",
        yaxis_title="Avg CATE ($)",
        margin=dict(t=50, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5),
    )

    return kpis, hist_fig, fi_fig, decile_fig, qini_fig, seg_fig


@app.callback(
    Output("method-collapse-tab4", "is_open"),
    Input("method-btn-tab4", "n_clicks"),
    State("method-collapse-tab4", "is_open"),
    prevent_initial_call=True,
)
def toggle_method_tab4(n, is_open):
    return not is_open


# ===========================================================================
# CALLBACKS — Tab 5: Multi-Arm OLS
# ===========================================================================

@app.callback(
    Output("ols-coef-plot", "figure"),
    Output("ols-marginal-table", "children"),
    Output("ols-heatmap-mens", "figure"),
    Output("ols-heatmap-womens", "figure"),
    Input("main-tabs", "active_tab"),
)
def update_ols(tab):
    coef_df = OLS["coef_df"].copy()
    subgroup_df = OLS["subgroup_df"].copy()

    # Filter to interesting terms
    keep_terms = [t for t in coef_df["term"] if t not in ["Intercept"]]
    plot_df = coef_df[coef_df["term"].isin(keep_terms)].copy()
    plot_df = plot_df.sort_values("coef")

    colors = [
        "#22c55e" if v > 0 and p < 0.05 else
        "#dc2626" if v < 0 and p < 0.05 else
        "#64748b"
        for v, p in zip(plot_df["coef"], plot_df["pvalue"])
    ]

    coef_fig = go.Figure()
    coef_fig.add_trace(go.Scatter(
        x=plot_df["coef"], y=plot_df["term"],
        mode="markers",
        error_x=dict(
            type="data",
            symmetric=False,
            array=plot_df["ci_hi"] - plot_df["coef"],
            arrayminus=plot_df["coef"] - plot_df["ci_lo"],
            color="#94a3b8",
        ),
        marker=dict(color=colors, size=10),
        name="Coefficient",
    ))
    coef_fig.add_vline(x=0, line_color="#dc2626", line_dash="dash")
    coef_fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"OLS Coefficients (n={OLS['n_obs']:,}, R²={OLS['r_squared']:.4f})",
        xaxis_title="Effect on Spend ($)",
        margin=dict(t=50, b=30, l=180),
        height=500,
    )

    # Marginal effects table — both arms, aggregated by newbie × channel
    disp_df = subgroup_df.groupby(["newbie", "channel"])[["me_mens", "me_womens"]].mean().reset_index()
    disp_df = disp_df.rename(columns={
        "newbie": "Customer type",
        "channel": "Channel",
        "me_mens": "Men's Email ($)",
        "me_womens": "Women's Email ($)",
    })
    disp_df["Men's Email ($)"] = disp_df["Men's Email ($)"].round(2)
    disp_df["Women's Email ($)"] = disp_df["Women's Email ($)"].round(2)

    table = dash_table.DataTable(
        data=disp_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in disp_df.columns],
        style_table={"overflowX": "auto"},
        style_cell={"backgroundColor": "#1a1a2e", "color": "#e2e8f0", "border": "1px solid #334155", "textAlign": "left", "padding": "8px"},
        style_header={"backgroundColor": "#0f0f23", "fontWeight": "bold", "color": "#94a3b8"},
        style_data_conditional=[
            {"if": {"filter_query": "{Men's Email ($)} > 0", "column_id": "Men's Email ($)"}, "color": "#22c55e"},
            {"if": {"filter_query": "{Men's Email ($)} < 0", "column_id": "Men's Email ($)"}, "color": "#dc2626"},
            {"if": {"filter_query": "{Women's Email ($)} > 0", "column_id": "Women's Email ($)"}, "color": "#22c55e"},
            {"if": {"filter_query": "{Women's Email ($)} < 0", "column_id": "Women's Email ($)"}, "color": "#dc2626"},
        ],
        page_size=12,
    )

    # Shared z-range across both arms so colours are directly comparable
    all_vals = pd.concat([
        subgroup_df.groupby(["newbie", "channel"])["me_mens"].mean(),
        subgroup_df.groupby(["newbie", "channel"])["me_womens"].mean(),
    ])
    zmax = max(abs(all_vals.min()), abs(all_vals.max()))
    zmin = -zmax

    def make_heatmap(arm_col):
        heat_pivot = subgroup_df.groupby(["newbie", "channel"])[arm_col].mean().unstack()
        return go.Figure(go.Heatmap(
            z=heat_pivot.values,
            x=heat_pivot.columns.tolist(),
            y=heat_pivot.index.tolist(),
            colorscale="RdYlGn",
            zmin=zmin,
            zmax=zmax,
            zmid=0,
            text=[[f"${v:.2f}" for v in row] for row in heat_pivot.values],
            texttemplate="%{text}",
            colorbar=dict(title="$ lift"),
        ))

    mens_heat = make_heatmap("me_mens")
    mens_heat.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Men's Email — Marginal Effect ($)",
        xaxis_title="Channel",
        yaxis_title="Customer type",
        margin=dict(t=50, b=30),
    )

    womens_heat = make_heatmap("me_womens")
    womens_heat.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Women's Email — Marginal Effect ($)",
        xaxis_title="Channel",
        yaxis_title="Customer type",
        margin=dict(t=50, b=30),
    )

    return coef_fig, table, mens_heat, womens_heat


@app.callback(
    Output("method-collapse-tab5", "is_open"),
    Input("method-btn-tab5", "n_clicks"),
    State("method-collapse-tab5", "is_open"),
    prevent_initial_call=True,
)
def toggle_method_tab5(n, is_open):
    return not is_open


# ===========================================================================
# CALLBACKS — Tab 6: Method Comparison
# ===========================================================================

def _build_comparison_df():
    """Aggregate all method estimates into one DataFrame."""
    rows = []
    for arm in ["mens", "womens"]:
        arm_label = "Men's Email" if arm == "mens" else "Women's Email"
        p = PSM[arm]
        rows.append({
            "Method": "PSM (ATT)",
            "Arm": arm_label,
            "Estimate ($)": round(p["att_point"], 2),
            "CI Lower ($)": round(p["att_ci_lo"], 2),
            "CI Upper ($)": round(p["att_ci_hi"], 2),
        })

    pair_map = {"mens": "mens_vs_control", "womens": "womens_vs_control"}
    for arm in ["mens", "womens"]:
        arm_label = "Men's Email" if arm == "mens" else "Women's Email"
        b = BAYESIAN[pair_map[arm]]
        rows.append({
            "Method": "Bayesian A/B (posterior mean)",
            "Arm": arm_label,
            "Estimate ($)": round(float(np.mean(b["delta_samples"])), 2),
            "CI Lower ($)": round(b["hdi_lo"], 2),
            "CI Upper ($)": round(b["hdi_hi"], 2),
        })

    for arm in ["mens", "womens"]:
        arm_label = "Men's Email" if arm == "mens" else "Women's Email"
        u = UPLIFT[arm]
        rows.append({
            "Method": "T-Learner (avg CATE)",
            "Arm": arm_label,
            "Estimate ($)": round(u["avg_cate_t"], 2),
            "CI Lower ($)": None,
            "CI Upper ($)": None,
        })
        rows.append({
            "Method": "S-Learner (avg CATE)",
            "Arm": arm_label,
            "Estimate ($)": round(u["avg_cate_s"], 2),
            "CI Lower ($)": None,
            "CI Upper ($)": None,
        })

    coef_df = OLS["coef_df"]
    for arm, col, arm_label in [("mens", "mens_email", "Men's Email"), ("womens", "womens_email", "Women's Email")]:
        row_ols = coef_df[coef_df["term"] == col].iloc[0]
        rows.append({
            "Method": "OLS (main effect)",
            "Arm": arm_label,
            "Estimate ($)": round(row_ols["coef"], 2),
            "CI Lower ($)": round(row_ols["ci_lo"], 2),
            "CI Upper ($)": round(row_ols["ci_hi"], 2),
        })

    return pd.DataFrame(rows)


@app.callback(
    Output("comparison-table", "children"),
    Output("forest-plot-mens", "figure"),
    Output("forest-plot-womens", "figure"),
    Output("key-takeaway-card", "children"),
    Input("main-tabs", "active_tab"),
)
def update_comparison(tab):
    if tab != "tab-6":
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    comp_df = _build_comparison_df()

    table = dash_table.DataTable(
        data=comp_df.fillna("—").to_dict("records"),
        columns=[{"name": c, "id": c} for c in comp_df.columns],
        export_format="csv",
        export_headers="display",
        style_table={"overflowX": "auto"},
        style_cell={"backgroundColor": "#1a1a2e", "color": "#e2e8f0", "border": "1px solid #334155", "textAlign": "left", "padding": "8px"},
        style_header={"backgroundColor": "#0f0f23", "fontWeight": "bold", "color": "#94a3b8"},
        style_data_conditional=[
            {"if": {"filter_query": "{Estimate ($)} > 0", "column_id": "Estimate ($)"}, "color": "#22c55e"},
            {"if": {"filter_query": "{Estimate ($)} < 0", "column_id": "Estimate ($)"}, "color": "#dc2626"},
        ],
    )

    def forest_plot(arm_label, color):
        sub = comp_df[comp_df["Arm"] == arm_label].copy()
        fig = go.Figure()
        for i, row in sub.iterrows():
            has_ci = pd.notna(row["CI Lower ($)"]) and pd.notna(row["CI Upper ($)"])
            fig.add_trace(go.Scatter(
                x=[row["Estimate ($)"]],
                y=[row["Method"]],
                mode="markers",
                marker=dict(color=color, size=12, symbol="diamond"),
                name=row["Method"],
                showlegend=False,
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[row["CI Upper ($)"] - row["Estimate ($)"]] if has_ci else [0],
                    arrayminus=[row["Estimate ($)"] - row["CI Lower ($)"]] if has_ci else [0],
                    color="#94a3b8",
                ) if has_ci else None,
            ))
        fig.add_vline(x=0, line_color="#dc2626", line_dash="dash")
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            title=f"Forest Plot — {arm_label}",
            xaxis_title="Effect on Spend ($)",
            margin=dict(t=50, b=30, l=180),
            height=350,
        )
        return fig

    mens_fig = forest_plot("Men's Email", "#7ec8e3")
    womens_fig = forest_plot("Women's Email", "#f9a8d4")

    # Key takeaway — data-driven summary
    mens_estimates   = comp_df[comp_df["Arm"] == "Men's Email"]["Estimate ($)"].values
    womens_estimates = comp_df[comp_df["Arm"] == "Women's Email"]["Estimate ($)"].values
    mens_valid   = [v for v in mens_estimates   if pd.notna(v)]
    womens_valid = [v for v in womens_estimates if pd.notna(v)]
    mens_agree   = all(v > 0 for v in mens_valid)
    womens_agree = all(v > 0 for v in womens_valid)
    mens_min, mens_max     = min(mens_valid),   max(mens_valid)
    womens_min, womens_max = min(womens_valid), max(womens_valid)

    mens_verdict   = "Strong consensus — all methods agree on a positive effect." if mens_agree   else "Methods disagree on direction — inspect assumptions carefully."
    womens_verdict = "All methods agree on a positive effect."                     if womens_agree else "Mixed signals — some methods find no effect. Treat with caution."

    takeaway = dbc.Card([
        dbc.CardHeader("🎯 Key Takeaway", className="fw-bold"),
        dbc.CardBody([
            html.P([
                html.Strong("Men's campaign: "),
                f"Estimated spend uplift ranges from ${mens_min:.2f} to ${mens_max:.2f} across "
                f"{len(mens_valid)} methods. {mens_verdict}",
            ]),
            html.P([
                html.Strong("Women's campaign: "),
                f"Estimated spend uplift ranges from ${womens_min:.2f} to ${womens_max:.2f} across "
                f"{len(womens_valid)} methods. {womens_verdict}",
            ]),
            html.P(
                "Method agreement strengthens causal credibility. Where estimates diverge, "
                "the gap reflects differing assumptions — PSM relies on covariate overlap, "
                "Bayesian A/B on distributional form, and uplift models on out-of-sample generalisation.",
                className="text-muted small mb-0",
            ),
        ]),
    ], style={**CARD_STYLE, "borderColor": "#22c55e" if (mens_agree and womens_agree) else "#fbbf24"})

    return table, mens_fig, womens_fig, takeaway


# ===========================================================================
# CALLBACKS — Tab 1 methodology collapse (no output needed, just toggle)
# ===========================================================================

@app.callback(
    Output("method-collapse-tab1", "is_open"),
    Input("method-btn-tab1", "n_clicks"),
    State("method-collapse-tab1", "is_open"),
    prevent_initial_call=True,
)
def toggle_method_tab1(n, is_open):
    return not is_open


# ===========================================================================
# Run
# ===========================================================================

if __name__ == "__main__":
    app.run(debug=True, port=8050)
