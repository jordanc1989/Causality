"""PSM page (pedagogical) — `/psm`."""

import dash

from layouts.psm import tab2_layout

dash.register_page(
    __name__,
    path="/psm",
    name="PSM (pedagogical)",
    title="Causal Inference Dashboard — PSM (pedagogical)",
    order=5,
)

layout = tab2_layout
