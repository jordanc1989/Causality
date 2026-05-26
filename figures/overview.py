
import numpy as np
import plotly.graph_objects as go
from dashboard.data import DF
from dashboard.theme import (
    PLOTLY_TEMPLATE, MENS_COLOUR, WOMENS_COLOUR, CTRL_COLOUR,
    WARNING, BORDER, TEXT, hex_to_rgba,
)
from dashboard.theme import COVARIATE_LABELS
import causal_utils as cu

def _fig_spend_box():
    spenders = DF[DF["spend"] > 0].copy()

    seg_order = ["Mens E-Mail", "Womens E-Mail", "No E-Mail"]
    color_map = {
        "Mens E-Mail": MENS_COLOUR,
        "Womens E-Mail": WOMENS_COLOUR,
        "No E-Mail": CTRL_COLOUR
    }

    seg_labels = {
        "Mens E-Mail": "Men's Email",
        "Womens E-Mail": "Women's Email",
        "No E-Mail": "Control"
    }

    outlier_rgba = hex_to_rgba(TEXT, 0.35)
    fig = go.Figure()
    for seg in seg_order:
        vals = spenders[spenders["segment"] == seg]["spend"].values
        q1, med, q3 = np.percentile(vals, [25, 50, 75], method='linear')
        mean_val = vals.mean()
        fill = color_map[seg]
        fig.add_trace(
            go.Box(
                y=vals,
                name=seg_labels[seg],
                width=0.2,
                marker=dict(
                    color=fill,
                    outliercolor=outlier_rgba,
                    line=dict(outliercolor=outlier_rgba, outlierwidth=1)
                ),
                line=dict(color=TEXT, width=1.4),
                fillcolor=fill,
                opacity=0.55,
                boxmean=True,
                boxpoints='outliers',
                customdata=[[q1, med, q3, mean_val] for _ in range(len(vals))],
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Median: $%{customdata[1]:.0f}<br>"
                    "Mean: $%{customdata[3]:.0f}<br>"
                    "IQR: $%{customdata[0]:.0f}-$%{customdata[2]:.0f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        showlegend=False,
        margin=dict(t=30, b=20),
        yaxis_title="Spend ($)",
        xaxis_title="",
        boxgap=0.3
    )
    return fig

def _fig_covariate_balance():
    covs = cu.COVARIATES
    control = DF[DF["segment"] == "No E-Mail"]
    arms = {
        "Men's Email": DF[DF["segment"] == "Mens E-Mail"],
        "Women's Email": DF[DF["segment"] == "Womens E-Mail"]
    }
    colors = {"Men's Email": MENS_COLOUR, "Women's Email": WOMENS_COLOUR}
    symbols = {"Men's Email": "diamond", "Women's Email": "circle"}

    fig = go.Figure()
    for arm_label, arm_df in arms.items():
        smds, labels = [], []
        for cov in covs:
            smds.append(cu.smd(arm_df[cov].values, control[cov].values))
            labels.append(COVARIATE_LABELS.get(cov, cov))

        fig.add_trace(
            go.Scatter(
                x=smds,
                y=labels,
                mode="markers",
                name=arm_label,
                marker=dict(
                    color=colors[arm_label], size=11, symbol=symbols[arm_label]
                ),
                hovertemplate="%{y}<br>SMD: %{x:.3f}<extra>%{fullData.name}</extra>"
            )
        )

    fig.add_vline(
        x=0.1,
        line_dash="dash",
        line_color=WARNING
    )
    fig.add_vline(
        x=-0.1,
        line_dash="dash",
        line_color=WARNING
    )
    fig.add_vline(x=0, line_color=BORDER, line_width=1)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        margin=dict(t=30, b=60, l=160, r=40),
        xaxis_title="Standardised Mean Difference (vs Control)",
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        height=340
    )
    return fig

