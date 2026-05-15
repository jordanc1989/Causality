"""App shell: navbar (with one nav link per registered page) and a
`dash.page_container` placeholder where the active page renders.

Dash Pages handles URL → layout routing automatically; this file just defines
the chrome and the page-link list driven by `dash.page_registry`.
"""

import dash
import dash_bootstrap_components as dbc
from dash import html
from dashboard.theme import ACCENT, TEXT, MUTED, BG

# Inline SVG mark — a "cause → effect" node-link icon, themed for the dashboard.
# Inlining as a data URI keeps the navbar logo self-contained (no LFS asset,
# no missing-image broken icon when the binary file is unresolved).
_LOGO_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32' "
    "width='32' height='32' role='img' aria-label='Causal inference mark'>"
    f"<circle cx='7' cy='16' r='4' fill='{ACCENT}'/>"
    f"<line x1='11' y1='16' x2='20' y2='16' stroke='{TEXT}' "
    "stroke-width='1.8' stroke-linecap='round'/>"
    f"<polygon points='19,12.6 25,16 19,19.4' fill='{TEXT}'/>"
    f"<circle cx='25' cy='16' r='3' fill='none' stroke='{TEXT}' stroke-width='1.8'/>"
    "</svg>"
)


def _logo_data_uri():
    # url-encode the few characters that would otherwise need %-escaping in a
    # data URI
    encoded = _LOGO_SVG.replace("#", "%23").replace("\n", "")
    return f"data:image/svg+xml;charset=utf-8,{encoded}"


def _page_nav_links():
    """Build NavLinks ordered by each page's `order` value."""
    ordered = sorted(
        (p for p in dash.page_registry.values() if p.get("path")),
        key=lambda p: p.get("order", 99),
    )
    links = []
    for i, page in enumerate(ordered, start=1):
        links.append(
            dbc.NavLink(
                f"{i} {page['name']}",
                href=page["relative_path"],
                active="exact",
                className="dashboard-nav-link",
            )
        )
    return links


def build_layout():
    return html.Div(
        [
            dbc.Navbar(
                dbc.Container(
                    [
                        html.Div(
                            [
                                html.Img(
                                    src=_logo_data_uri(),
                                    alt="Causal inference mark",
                                    style={
                                        "height": "26px",
                                        "width": "26px",
                                        "marginRight": "0.7rem",
                                    },
                                ),
                                html.Span(
                                    "Causal",
                                    style={
                                        "fontFamily": "Ubuntu, sans-serif",
                                        "fontWeight": "700",
                                        "fontSize": "1.05rem",
                                        "color": ACCENT,
                                        "letterSpacing": "-0.02em",
                                    },
                                ),
                                html.Span(
                                    " inference",
                                    style={
                                        "fontFamily": "Ubuntu, sans-serif",
                                        "fontWeight": "500",
                                        "fontSize": "1.05rem",
                                        "color": TEXT,
                                    },
                                ),
                            ],
                            style={"display": "flex", "alignItems": "center"},
                        ),
                        html.Div(
                            [
                                html.Span(
                                    "Hillstrom randomised email test",
                                    style={
                                        "fontFamily": "Ubuntu, sans-serif",
                                        "fontSize": "0.72rem",
                                        "fontWeight": "500",
                                        "color": MUTED,
                                    },
                                )
                            ],
                            className="ms-auto d-none d-md-flex align-items-center",
                        ),
                    ],
                    fluid=True,
                ),
                className="dashboard-navbar mb-0",
                color="dark",
                dark=True,
                sticky="top",
            ),
            dbc.Container(
                dbc.Nav(
                    _page_nav_links(),
                    pills=False,
                    className="dashboard-tabs",
                    id="page-nav",
                ),
                fluid=True,
                className="pt-2",
            ),
            dbc.Container(
                dash.page_container,
                fluid=True,
            ),
        ],
        className="dashboard-app",
        style={"backgroundColor": BG, "minHeight": "100vh"},
    )
