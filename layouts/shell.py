"""App shell: an editorial masthead (nameplate, standfirst, dateline) and the
section nav. The nav is rendered as a sibling of the masthead - not inside it -
so it can be `position: sticky` to the viewport while the nameplate scrolls
away on long pages.

Dash Pages handles URL → layout routing automatically; this file just defines
the chrome and the page-link list driven by `dash.page_registry`.
"""

import dash
import dash_bootstrap_components as dbc
from dash import html
from dashboard.theme import BG
from layouts.components import masthead_dateline


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
        html.Div(
            [
                html.Div("Causal Inference", className="masthead-title"),
                html.Div(
                    "An analysis of the Hillstrom randomised email experiment",
                    className="masthead-standfirst",
                ),
                masthead_dateline(
                    "Hillstrom (2008) MineThatData email analytics challenge",
                    updated="May 2026",
                ),
            ],
            className="masthead-inner",
        ),
        className="masthead",
    )

    body = dbc.Container(
        [
            dbc.Nav(
                _page_nav_links(),
                pills=False,
                className="dashboard-tabs",
                id="page-nav",
            ),
            dash.page_container,
        ],
        fluid=True,
    )

    return html.Div(
        [
            dbc.Container(masthead, fluid=True),
            body,
        ],
        className="dashboard-app",
        style={"backgroundColor": BG, "minHeight": "100vh"},
    )
