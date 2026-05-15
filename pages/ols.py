"""Multi-Arm OLS page — `/ols`."""

import dash

from layouts.ols import tab5_layout

dash.register_page(
    __name__,
    path="/ols",
    name="Multi-Arm OLS",
    title="Causal Inference Dashboard — Multi-Arm OLS",
    order=3,
)

layout = tab5_layout
