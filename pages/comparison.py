"""Method Comparison page — `/comparison`."""

import dash

from layouts.comparison import tab6_layout

dash.register_page(
    __name__,
    path="/comparison",
    name="Method Comparison",
    title="Causal Inference Dashboard — Method Comparison",
    order=6,
)

layout = tab6_layout
