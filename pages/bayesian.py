"""Bayesian A/B page — `/bayesian`."""

import dash

from layouts.bayesian import tab3_layout

dash.register_page(
    __name__,
    path="/bayesian",
    name="Bayesian A/B",
    title="Causal Inference Dashboard — Bayesian A/B",
    order=2,
)

layout = tab3_layout
