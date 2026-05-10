
import numpy as np
import plotly.graph_objects as go
from dashboard.data import DF
from dashboard.theme import (
    PLOTLY_TEMPLATE, MENS_COLOUR, WOMENS_COLOUR, CTRL_COLOUR,
    WARNING, BORDER,
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
                    outliercolor='rgba(0,0,0,0.4)',
                    line=dict(outliercolor='rgba(0,0,0,0.4)', outlierwidth=1)
                ),
                line=dict(color='black', width=1.5),
                fillcolor=fill,
                opacity=0.75,
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
            a = arm_df[cov].values
            b = control[cov].values
            pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
            smd = (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0
            smds.append(smd)
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

