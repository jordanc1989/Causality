from .psm import register_psm_callbacks
from .bayesian import register_bayesian_callbacks
from .uplift import register_uplift_callbacks
from .ols import register_ols_callbacks
from .comparison import register_comparison_callbacks
from .overview import register_overview_callbacks


def register_callbacks(app):
    register_psm_callbacks(app)
    register_bayesian_callbacks(app)
    register_uplift_callbacks(app)
    register_ols_callbacks(app)
    register_comparison_callbacks(app)
    register_overview_callbacks(app)
