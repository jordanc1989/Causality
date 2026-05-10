"""
app.py
------
Thin entrypoint for the Causal Inference Dashboard.
"""

import dash
import dash_bootstrap_components as dbc

from dashboard.theme import GOOGLE_FONTS, register_plotly_template
from layouts.shell import build_layout
from callbacks import register_callbacks

register_plotly_template()

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG, GOOGLE_FONTS],
    suppress_callback_exceptions=True,
    title="Causal Inference Dashboard",
)
server = app.server

app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""

app.layout = build_layout()
register_callbacks(app)

if __name__ == "__main__":
    app.run(debug=False, port=8050)
