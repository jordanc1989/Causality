from dash import Output, Input, State

from .psm import register_psm_callbacks
from .bayesian import register_bayesian_callbacks
from .uplift import register_uplift_callbacks
from .ols import register_ols_callbacks
from .comparison import register_comparison_callbacks

# Pure-clientside toggle: flipping `is_open` doesn't need a server roundtrip.
_TOGGLE_JS = "function(n, is_open) { return !is_open; }"


def _register_collapse_toggles(app, pairs):
    """Wire `methodology_collapse` (and similar) buttons to their collapses
    via a single clientside callback per pair.

    `pairs` is an iterable of `(button_id, collapse_id)`. The button-click
    flips the collapse's `is_open` state with no server work.
    """
    for btn_id, collapse_id in pairs:
        app.clientside_callback(
            _TOGGLE_JS,
            Output(collapse_id, "is_open"),
            Input(btn_id, "n_clicks"),
            State(collapse_id, "is_open"),
            prevent_initial_call=True,
        )


def register_callbacks(app):
    register_psm_callbacks(app)
    register_bayesian_callbacks(app)
    register_uplift_callbacks(app)
    register_ols_callbacks(app)
    register_comparison_callbacks(app)

    # Methodology collapses on the per-page tabs. tab1 (Overview) has no
    # methodology collapse; tab6 (Comparison) doesn't either, verified by
    # grep against the layouts.
    _register_collapse_toggles(
        app,
        [
            ("method-btn-tab2", "method-collapse-tab2"),
            ("method-btn-tab3", "method-collapse-tab3"),
            ("method-btn-tab4", "method-collapse-tab4"),
            ("method-btn-tab5", "method-collapse-tab5"),
            ("trace-btn", "trace-collapse"),
            ("ppc-btn", "ppc-collapse"),
            ("diag-btn", "diag-collapse"),
        ],
    )
