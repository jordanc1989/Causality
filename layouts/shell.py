"""App shell: an editorial masthead (nameplate + section nav) and a
`dash.page_container` placeholder where the active page renders.

Dash Pages handles URL → layout routing automatically; this file just defines
the chrome and the page-link list driven by `dash.page_registry`.
"""

import dash
import dash_bootstrap_components as dbc
from dash import html
from dashboard.theme import BG


def _page_nav_links():
    """Build NavLinks ordered by each page's `order` value."""
    ordered = sorted(
        (p for p in dash.page_registry.values() if p.get("path")),
        key=lambda p: p.get("order", 99),
    )
    return [
        dbc.NavLink(
            page["name"],
            href=page["relative_path"],
            active="exact",
            className="dashboard-nav-link",
        )
        for page in ordered
    ]


def build_layout():
    masthead = html.Header(
        dbc.Container(
            [
                html.Div(
                    [
                        html.Div("Causal Inference", className="masthead-title"),
                        html.Div(
                            "An analysis of the Hillstrom randomised email experiment",
                            className="masthead-standfirst",
                        ),
                    ],
                    className="masthead-inner",
                ),
                dbc.Nav(
                    _page_nav_links(),
                    pills=False,
                    className="dashboard-tabs",
                    id="page-nav",
                ),
            ],
            fluid=True,
        ),
        className="masthead",
    )

    return html.Div(
        [
            masthead,
            dbc.Container(dash.page_container, fluid=True),
        ],
        className="dashboard-app",
        style={"backgroundColor": BG, "minHeight": "100vh"},
    )
