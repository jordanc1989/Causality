# Causal Inference Dashboard — Agent Brief

## Project Overview
A portfolio-grade interactive causal inference dashboard using Dash (Python).
Uses the Hillstrom dataset (64k customers, two email campaigns + control) to demonstrate
multiple causal methods side-by-side with uncertainty estimates throughout.

## Target Audience
Senior hiring managers and data science peers. Must look polished, be self-explanatory,
and demonstrate deep methodological understanding.

## Tech Stack
- **Framework:** Dash + dash-bootstrap-components (CYBORG base theme, heavily overridden)
- **Causal methods:** sklearn (PSM), pymc (Bayesian A/B), scikit-uplift (T-Learner/S-Learner,
  Qini), statsmodels (Multi-Arm OLS)
- **Visualisation:** Plotly only (no matplotlib). Custom template `"enterprise_dark"` registered
  via `pio.templates` — apply to every figure via `template=PLOTLY_TEMPLATE`
- **Data:** `from sklift.datasets import fetch_hillstrom`
- **Styling:** Enterprise dark theme — Ubuntu/Oswald/Ubuntu Mono fonts via Google Fonts,
  deep teal palette (`BG=#041818`, `SURFACE=#072C2C`), orange accent `ACCENT=#FF5F03`.
  All design tokens defined at the top of `app.py`. Custom CSS injected via `app.index_string`.

## Data Notes
- Segments: `"Mens E-Mail"`, `"Womens E-Mail"`, `"No E-Mail"` (control)
- Key covariates: `recency`, `history`, `mens`, `womens`, `zip_code`, `newbie`, `channel`
- Outcomes: `visit`, `conversion`, `spend`
- **`zip_code` is misspelled `"Surburban"` in the raw dataset** — all code matches this spelling
- Randomised experiment; no time dimension
- Fixed random seed (`RANDOM_SEED = 42`) throughout for reproducibility

## Data Preprocessing (in `causal_utils.load_data`)
- No missing values or duplicates
- One-hot encode `zip_code` → `zip_suburban`, `zip_rural` (reference: Urban)
- One-hot encode `channel` → `channel_web`, `channel_multichannel` (reference: Phone)
- Binary treatment indicators: `is_mens`, `is_womens`, `is_control`
- `mens`, `womens`, `newbie` already binary — no transformation needed
- Raw `history` and `spend` kept untransformed for display

## Caching
All causal results are cached to `.cache/results.pkl` on first run and reloaded thereafter.
**Delete `.cache/results.pkl` and restart to rebuild after any change to `causal_utils.py`.**
When adding new keys to the results dict, use `.get("new_key", fallback)` in `app.py` so
the old cache still works until it's rebuilt.

## App Structure

### Tab 1: Overview
- KPI cards: n per segment, conversion rates with deltas vs control
- Violin plot of spend distribution by segment (among spenders, capped at p99)
- Covariate balance SMD dot plot (both arms vs control)

### Tab 2: PSM
- 1:1 nearest-neighbour matching on propensity score, no caliper (all treated units matched)
- KPI cards: ATT with 95% bootstrap CI, matched pairs + avg PS distance, common support range
- Propensity score distribution (overlapping histograms), Love Plot, ATT bar chart
- Love plot left margin: `l=195, yaxis=dict(automargin=True)` to prevent label clipping

### Tab 3: Bayesian A/B
- Normal likelihood model in PyMC (acknowledged simplification for zero-inflated spend)
- Posterior distribution with ROPE shading; HDI lines; P(effect > 0)
- ROPE input capped at `maxWidth: 220px`; guard `rope_val = rope_val if rope_val is not None else 0`
- Multi-arm: Men's vs Control, Women's vs Control, Men's vs Women's
- Collapsible MCMC trace plots

### Tab 4: Uplift / HTE
- T-Learner (`TwoModels`) and S-Learner (`SoloModel`) via scikit-uplift
- 5-fold cross-fitting to avoid in-sample CATE bias
- Decile lift and Qini curve are computed separately for each model's ranking
  (keys: `decile_lift` / `decile_lift_s`, `qini_x/y` / `qini_x_s/y_s`)
- Feature importance from T-Learner treatment model only

### Tab 5: Multi-Arm OLS
- statsmodels formula with treatment dummies + interaction terms (newbie, channel, zip)
- Marginal effects computed manually from params, averaged over zip for heatmap display
- Callback guards with `if tab != "tab-5": return dash.no_update × 4`

### Tab 6: Method Comparison
- Summary table + forest plots for Men's and Women's arms
- Key Takeaway card with data-driven verdict
- Callback guards with `if tab != "tab-6": return dash.no_update × 4`

## Component Conventions
- `kpi_card(value, label, delta, delta_positive, color, accent, info, info_id)` — `info`/`info_id`
  add an `ⓘ` tooltip; follow the ROPE info icon pattern
- `section_header(text)` — Oswald uppercase label above chart sections
- `methodology_collapse(tab_id, content)` — `html.Button` with class `btn-methodology`
- All chart colours use named tokens (`MENS_COLOR`, `WOMENS_COLOR`, `CTRL_COLOR`, etc.)

## What This Demonstrates
- Selecting and applying the right causal method for a given business question
- Understanding assumptions (common support, SUTVA, ignorability, Normal likelihood limitation)
- Communicating uncertainty — CIs, HDI, posterior distributions, not just point estimates
- Individual-level causal thinking via HTE/uplift
- Translation of technical results into plain-English business language
