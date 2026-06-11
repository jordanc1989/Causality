[![Live Demo](https://img.shields.io/badge/live-demo-purple)](https://jordancheney89-causality.hf.space/)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Spaces-causality-yellow)](https://huggingface.co/spaces/jordancheney89/causality)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.13+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/badge/managed%20by-uv-DE5FE9.svg)](https://docs.astral.sh/uv/)
[![Dash](https://img.shields.io/badge/Dash-4.x-2bb6f6.svg?logo=plotly&logoColor=white)](https://dash.plotly.com/)
[![PyMC](https://img.shields.io/badge/PyMC-6.x-1f77b4.svg)](https://www.pymc.io/)

# Causal Inference Dashboard

Interactive Dash app for comparing a suite of causal measurement approaches using a publicly available randomised marketing experiment as an example (Hillstrom, 2008).

**Live demo:** [Hugging Face Space](https://jordancheney89-causality.hf.space/)

This project provides a dashboard to:
- estimate average treatment effects (ATE) with uncertainty
- inspect heterogeneity and targeting value
- show where each method agrees and disagrees, and what assumptions drive the result

## What this dashboard shows

Teams might ask two different questions:
- "Did the campaign work on average?" (causal effect / ATE)
- "Who should we target next?" (HTE / uplift policy)

This dashboard puts both views side-by-side so the methodological choices and any business implications are easily comparable.

## Methods Covered

| Tab | Method | Role in this project |
|-----|--------|----------------------|
| 1 | Overview | Dataset summary, arm balance, headline effects |
| 2 | Bayesian A/B (PyMC hurdle model) | Probabilistic effect estimation with posterior uncertainty |
| 3 | Multi-Arm OLS with interactions | Precision-adjusted average effects and subgroup patterns |
| 4 | Uplift / HTE (S- and X-Learner) | Ranking customers by estimated incremental value, X-learner is default |
| 5 | PSM (propensity matching + caliper) | Pedagogical: the workflow you'd use on observational data, run on this RCT for comparison |
| 6 | Method Comparison | Side-by-side estimate reconciliation and takeaway |

## Dataset

Source: [MineThatData Email Analytics (Hillstrom)](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html)

Randomised experiment across ~64,000 customers:
- Men's, Women's and Control (split roughly equal-sized three ways)
- Primary outcome: 2-week post-campaign spend (USD)
- Key covariates: recency, history, mens/womens indicators, zip code, newbie, channel

## Quick Start

Dependencies are managed with **[uv](https://docs.astral.sh/uv/)**.

### Requirements
- [uv](https://docs.astral.sh/uv/getting-started/) installed
- Python **3.13+**

### Install

From the project root:

```bash
uv sync
```

This creates `.venv` (if needed) and installs the locked dependency set.

### Run

```bash
uv run app.py
```

Open `http://localhost:8050`.

### First-run behavior
- First run precomputes models and caches results in `.cache/results.pkl`.
- Subsequent runs load from cache and start quickly.
- Depending on machine speed, initial build can take several minutes.

### Force recompute
- Delete `.cache/results.pkl`, or set `USE_CACHE = False` in `causal_utils.py`.
- Restart the app once to rebuild the cache.
- Set `USE_CACHE` back to `True` after a deliberate rebuild (optional, deleting the pickle has the same effect if `USE_CACHE` stays `True`).

## Hugging Face Spaces (Docker)

**Live Space:** [huggingface.co/spaces/jordancheney89/causality](https://huggingface.co/spaces/jordancheney89/causality)

This repo includes a [`Dockerfile`](Dockerfile) configured for the [Docker Spaces SDK](https://huggingface.co/docs/hub/spaces-sdks-docker)

## Methodology Notes and Caveats

- The underlying dataset is randomized, so causal identification of average effects comes from random assignment.
- Covariate-adjusted analyses are included for precision and interpretability; propensity matching is included as a pedagogical workflow.
- Average CATE is reported with a bootstrap interval, though that interval is estimation-conditional and reads as a lower bound on the true uncertainty for high-stakes targeting.
- Propensity matching (PSM) is included as a demo workflow for non-randomised data, and isn't used as a headline estimate for this dashboard.

## Results Snapshot

- Men's email lifts two-week spend by about **$0.77 per recipient** (95% CI $0.50 to $1.05); Women's by about **$0.42** ($0.17 to $0.67).
- The methods agree: Bayesian, OLS and both uplift learners all land between $0.74 and $0.79 (Men's) and $0.42 and $0.45 (Women's).
- The lift is a conversion effect. The emails roughly double the share of customers who buy (1.25% / 0.88% vs 0.57% in control) while spend per buyer stays around $114.
- Both uplift rankings beat random targeting (permutation p < 0.002), and the policy view turns the ranking plus a send cost and margin into an optimal mailing share.

## Project Structure

```text
.
├── Dockerfile             # Hugging Face Spaces (Docker SDK), gunicorn on port 7860
├── .dockerignore          # Smaller build context (excludes .venv, caches of dev tools)
├── app.py                 # Thin entrypoint: Dash app, theme registration, layout, callback wiring
├── causal_utils.py        # Data prep, caching, and all causal estimation logic
├── dashboard/
│   ├── theme.py           # Design tokens, Plotly template, shared style dicts
│   └── data.py            # Loads cache → exposes RESULTS, DF, PSM, BAYESIAN, UPLIFT, OLS
├── pages/                 # Dash Pages route registration
│   ├── overview.py        # /
│   ├── bayesian.py        # /bayesian
│   ├── ols.py             # /ols
│   ├── uplift.py          # /uplift
│   ├── psm.py             # /psm
│   └── comparison.py      # /comparison
├── layouts/
│   ├── shell.py           # Masthead + section nav + Dash Pages container
│   ├── components.py      # Reusable UI helpers
│   ├── overview.py        # Overview layout
│   ├── psm.py             # PSM layout
│   ├── bayesian.py        # Bayesian A/B layout
│   ├── uplift.py          # Uplift / HTE layout
│   ├── ols.py             # Multi-Arm OLS layout
│   └── comparison.py      # Method Comparison layout
├── callbacks/
│   ├── __init__.py        # register_callbacks(app)
│   ├── psm.py             # PSM callbacks
│   ├── bayesian.py        # Bayesian A/B callbacks
│   ├── uplift.py          # Uplift / HTE callbacks
│   ├── ols.py             # Static OLS figure builder
│   └── comparison.py      # Method Comparison callbacks
├── figures/
│   └── overview.py        # Static Plotly helpers for Overview tab
├── content/
│   └── methodology.py     # Long-form copy separated from layout code
├── assets/
│   └── style.css          # Global styles (Dash serves /assets automatically)
├── .cache/                # Precomputed outputs (e.g. results.pkl)
├── pyproject.toml
├── uv.lock
├── .python-version
└── README.md
```

## Roadmap

- Add data ingestion wizard


## License

MIT. See `LICENSE`.
