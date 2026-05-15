"""Overview page — landing route at `/`."""

import dash

from layouts.overview import tab1_layout

dash.register_page(
    __name__,
    path="/",
    name="Overview",
    title="Causal Inference Dashboard — Overview",
    order=1,
)

layout = tab1_layout
