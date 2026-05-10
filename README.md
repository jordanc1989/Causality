# Causal Inference Dashboard

Interactive Dash app for comparing causal measurement approaches on a real randomised marketing experiment (Hillstrom, 2008).

This project demonstrates how to:
- estimate average treatment effects with uncertainty
- inspect heterogeneity and targeting value
- communicate assumptions and modeling trade-offs in a decision-friendly format

## Dashboard Preview

![Causal Inference Dashboard preview](assets/screenshot.png)

## Why This Project Matters

Teams often ask two different questions:
- "Did the campaign work on average?" (causal effect / ATE)
- "Who should we target next?" (HTE / uplift policy)

This dashboard puts both views side-by-side so methodological choices and business implications are transparent.

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

## Reproducibility Notes

- If you change estimation logic in `causal_utils.py`, delete `.cache/results.pkl` or use `USE_CACHE = False`, then rerun the app once.
- **`uv.lock`** pins transitive versions; run `uv lock` after changing dependencies in `pyproject.toml`.

## Methodology Notes and Caveats

- The underlying dataset is randomized, so causal identification of average effects comes from random assignment.
- Covariate-adjusted and matched analyses are included as precision, sensitivity, and interpretability tools.
- Uplift metrics are useful for ranking policy decisions but should ideally be reported with uncertainty intervals when used for high-stakes targeting.
- Subgroup interaction findings are exploratory unless multiplicity is explicitly controlled.

## Results Snapshot (What reviewers should look for)

- Agreement and disagreement across methods for Men's vs Control and Women's vs Control.
- Posterior probability and HDI width in Bayesian A/B (effect magnitude + uncertainty).
- Whether uplift curves and decile lift indicate actionable ranking value beyond random targeting.
- Consistency between OLS interaction patterns and uplift heterogeneity signals.

## Project Structure

```text
.
├── app.py              # Dash UI, figures, callbacks, interpretation copy
├── causal_utils.py     # Data prep and causal estimation logic
├── pyproject.toml      # Project metadata and dependencies
├── uv.lock             # Locked versions (committed for reproducibility)
├── .python-version     # Hint for Python 3.13 (optional; used by uv/pyenv)
└── README.md
```

## Roadmap

- Add deployment and public demo link.
- Add benchmark tests for uplift policy value with uncertainty intervals.
- Add formal multiplicity-adjusted subgroup reporting mode.


## License

MIT. See `LICENSE`.
