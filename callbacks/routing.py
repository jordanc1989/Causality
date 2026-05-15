"""Two-way sync between the URL hash and the active tab so any view can be
deep-linked by appending e.g. `#tab-3` to the dashboard URL."""

import dash
from dash import Output, Input, State

VALID_TABS = {"tab-1", "tab-2", "tab-3", "tab-4", "tab-5", "tab-6"}


def _url_to_tab(url_hash, active_tab):
    target = (url_hash or "").lstrip("#")
    if target in VALID_TABS and target != active_tab:
        return target
    return dash.no_update


def _tab_to_url(active_tab):
    if active_tab in VALID_TABS:
        return f"#{active_tab}"
    return dash.no_update


def register_routing_callbacks(app):
    app.callback(
        Output("main-tabs", "active_tab"),
        Input("url", "hash"),
        State("main-tabs", "active_tab"),
    )(_url_to_tab)
    app.callback(
        Output("url", "hash"),
        Input("main-tabs", "active_tab"),
    )(_tab_to_url)
