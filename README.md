---
title: Causal Inference Dashboard
emoji: 📈
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
license: mit
---

# Causal Inference Dashboard

Interactive Dash app for comparing causal measurement approaches on a real randomised marketing experiment (Hillstrom, 2008).

**Live demo:** [Hugging Face Space](https://huggingface.co/spaces/jordancheney89/causality)

This project provides a dashboard to:
- estimate average treatment effects with uncertainty
- inspect heterogeneity and targeting value
- show where each method agrees, where it disagrees and what assumptions drive the result

## Dashboard Preview

![Causal Inference Dashboard preview](assets/screenshot.png)

## Why This Project Matters

Teams might ask two different questions:
- "Did the campaign work on average?" (causal effect / ATE)
- "Who should we target next?" (HTE / uplift policy)

This dashboard puts both views side-by-side so the methodological choices and any business implications are easily comparable.

## Methods Covered

| Tab | Method | Role in this project |
|-----|--------|----------------------|
| 2 | PSM sensitivity (propensity matching + caliper) | Observational-style diagnostic; matched ATT on pruned cohort vs control |
| 3 | Bayesian A/B (PyMC hurdle model) | Probabilistic effect estimation with posterior uncertainty |
| 4 | Uplift / HTE (T-Learner, S-Learner) | Ranking customers by estimated incremental value |
| 5 | Multi-Arm OLS with interactions | Precision-adjusted average effects and subgroup patterns |
| 6 | Method Comparison | Side-by-side estimate reconciliation and takeaway |

## Dataset

Source: [MineThatData Email Analytics (Hillstrom)](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html)

Randomised experiment across ~64,000 customers:
- Men's Email: 21,388
- Women's Email: 21,307
- Control (No Email): 21,305
- Primary outcome: 2-week post-campaign spend (USD)
- Key covariates: recency, history, mens/womens indicators, zip code, newbie, channel

## Quick Start

Dependencies are managed with **[uv](https://docs.astral.sh/uv/)** ([`pyproject.toml`](pyproject.toml) + [`uv.lock`](uv.lock)).

### Requirements
- [uv](https://docs.astral.sh/uv/getting-started/) installed
- Python **3.13+** (see [`requires-python`](pyproject.toml); [`.python-version`](.python-version) pins 3.13 for local development)
- macOS/Linux/Windows supported

### Install

From the project root:

```bash
uv sync
```

This creates `.venv` (if needed) and installs the locked dependency set.

### Run

```bash
uv run python app.py
```

Open `http://localhost:8050`.

### First-run behavior
- First run precomputes models and caches results in `.cache/results.pkl`.
- Subsequent runs load from cache and start quickly.
- Depending on machine speed, initial build can take several minutes.

### Force recompute
- Delete `.cache/results.pkl`, or set `USE_CACHE = False` in `causal_utils.py`.
- Restart the app once to rebuild the cache.
- Set `USE_CACHE` back to `True` after a deliberate rebuild (optional; deleting the pickle has the same effect if `USE_CACHE` stays `True`).

## Hugging Face Spaces (Docker)

**Live Space:** [huggingface.co/spaces/jordancheney89/causality](https://huggingface.co/spaces/jordancheney89/causality)

This repo includes a [`Dockerfile`](Dockerfile) configured for the [Docker Spaces SDK](https://huggingface.co/docs/hub/spaces-sdks-docker)


## Reproducibility Notes

- If you change estimation logic in `causal_utils.py`, delete `.cache/results.pkl` or use `USE_CACHE = False`, then rerun the app once.
- **`uv.lock`** pins transitive versions; run `uv lock` after changing dependencies in `pyproject.toml`.

## Methodology Notes and Caveats

- The underlying dataset is randomized, so causal identification of average effects comes from random assignment.
- Covariate-adjusted and matched analyses are included as precision, sensitivity, and interpretability tools.
- Uplift metrics are useful for ranking policy decisions but should ideally be reported with uncertainty intervals when used for high-stakes targeting.
- Subgroup interaction findings are exploratory unless multiplicity is explicitly controlled.

## Results Snapshot

- Agreement and disagreement across methods for Mens vs Control and Womens vs Control.
- Posterior probability and HDI width in Bayesian A/B (effect magnitude + uncertainty).
- Whether uplift curves and decile lift indicate actionable ranking value beyond random targeting.
- Consistency between OLS interaction patterns and uplift heterogeneity signals.

## Project Structure

```text
.
├── Dockerfile             # Hugging Face Spaces (Docker SDK); gunicorn on port 7860
├── .dockerignore          # Smaller build context (excludes .venv, caches of dev tools)
├── app.py                 # Thin entrypoint: Dash app, theme registration, layout, callback wiring
├── causal_utils.py        # Data prep, caching, and all causal estimation logic
├── dashboard/
│   ├── theme.py           # Design tokens, Plotly template, shared style dicts
│   └── data.py            # Loads cache → exposes RESULTS, DF, PSM, BAYESIAN, UPLIFT, OLS
├── layouts/
│   ├── shell.py           # Navbar + tab container; imports per-tab layouts
│   ├── components.py      # Reusable UI helpers (KPI cards, section headers, methodology collapse)
│   ├── overview.py        # Tab 1 layout
│   ├── psm.py             # Tab 2 layout
│   ├── bayesian.py        # Tab 3 layout
│   ├── uplift.py          # Tab 4 layout
│   ├── ols.py             # Tab 5 layout
│   └── comparison.py      # Tab 6 layout
├── callbacks/
│   ├── __init__.py        # register_callbacks(app)
│   ├── overview.py        # Tab 1 callbacks
│   ├── psm.py             # Tab 2 callbacks
│   ├── bayesian.py        # Tab 3 callbacks
│   ├── uplift.py          # Tab 4 callbacks
│   ├── ols.py             # Tab 5 callbacks
│   └── comparison.py      # Tab 6 callbacks
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
