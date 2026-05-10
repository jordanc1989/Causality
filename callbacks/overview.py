
from dash import Output, Input, State
from dashboard.theme import *

def toggle_method_tab1(n, is_open):
    return not is_open




def register_overview_callbacks(app):
    app.callback(
        Output("method-collapse-tab1", "is_open"),
        Input("method-btn-tab1", "n_clicks"),
        State("method-collapse-tab1", "is_open"),
        prevent_initial_call=True,
    )(toggle_method_tab1)
