"""Uplift / HTE page - `/uplift`."""

import dash

from layouts.uplift import tab4_layout

dash.register_page(
    __name__,
    path="/uplift",
    name="Uplift / HTE",
    title="Causal Inference Dashboard - Uplift / HTE",
    order=4,
)

layout = tab4_layout
