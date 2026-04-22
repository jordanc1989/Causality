# Causal Inference Dashboard — Agent Brief

## Project Overview
Build a professional portfolio-grade interactive causal inference dashboard using Dash (Python).
The app uses the Hillstrom dataset (64k customers across two email campaigns — Men's and Women's,
plus a control group) and allows the user to run and compare multiple causal inference methods
side-by-side, with rich visualisations and uncertainty estimates throughout.

## Target Audience
Senior hiring managers and data science peers reviewing a portfolio. The app must look polished,
be self-explanatory and demonstrate deep methodological understanding.

## Tech Stack
- **Framework:** Dash (Plotly) with dash-bootstrap-components
- **Causal methods:** sklearn (PSM), pymc (Bayesian A/B), scikit-uplift (T-Learner/S-Learner,
Qini evaluation), dowhy (causal graph, refutation tests), statsmodels (Multi-Arm OLS)
- **Visualisation:** Plotly (no matplotlib)
- **Data:** Use scikit-uplift to load the Hillstrom dataset via `fetch_hillstrom`
- **Styling:** Dark theme (dash-bootstrap-components DARKLY or CYBORG), professional and
  minimal — no generic "AI slop" aesthetics
  
## Data Notes
- Load via: `from sklift.datasets import fetch_hillstrom`
- The dataset has 64,000 customers with three segments: `"Mens E-Mail"`, `"Womens E-Mail"`,
  and `"No E-Mail"` (control)
- Key covariates: `recency`, `history`, `mens`, `womens`, `zip_code`, `newbie`, `channel`
- Outcome variables: `visit`, `conversion`, `spend`
- This is a cross-sectional randomised experiment — there is no time dimension in the raw data
- PyMC sampling and bootstrap procedures must use a fixed random seed for reproducibility

## Data Cleaning & Preprocessing
- No missing values or duplicates — no imputation required
- Encode `channel` and `zip_code` as one-hot encoded columns
- Standardise `recency`, `history` to z-scores for PSM and model inputs
- `mens`, `womens`, `newbie` are already binary — no transformation needed
- Engineer `high_value` flag (history > median) for HTE subgroup analysis
- Keep raw `history` and `spend` untransformed for descriptive stats and KPI cards

## App Structure — Pages / Tabs

### Tab 1: Overview & Data Explorer
- Summary cards: n treated (Men's), n treated (Women's), n control, overall conversion rate
  by segment
- Spend distribution chart by segment (box plot or violin plot)
- Covariate balance table (before matching): mean comparison of all covariates across all
  three groups (`recency`, `history`, `mens`, `womens`, `zip_code`, `newbie`, `channel`)
- Brief plain-English explanation of the dataset, the causal question being answered, and
  why a randomised experiment still benefits from causal analysis

### Tab 2: Propensity Score Matching (PSM)
- Propensity score distribution plot: overlapping histograms for treatment vs control (before
  and after matching)
- Covariate balance plot: standardised mean differences before vs after matching — "love plot"
- Matched sample ATT estimate with 95% CI (bootstrap)
- Table showing: n matched pairs, common support range, % of treated matched
- Run separately for Men's campaign vs control and Women's campaign vs control
- Plain-English interpretation paragraph

### Tab 3: Bayesian A/B Test
- Prior/posterior distribution plot for the treatment effect using PyMC
- Credible interval KPI card (e.g. "95% HDI: $9.20 — $14.80 uplift per customer")
- Probability of positive effect displayed prominently
- ROPE (Region of Practical Equivalence) toggle — user sets minimum meaningful effect size
- Trace plots for MCMC diagnostics (collapsible)
- Support multi-arm comparison: Men's vs Control, Women's vs Control, and Men's vs Women's
- Plain-English interpretation paragraph

### Tab 4: Uplift Modelling / Heterogeneous Treatment Effects (HTE)
- scikit-uplift TwoModels (T-Learner) and SoloModel (S-Learner) for CATE estimation
- DoWhy CausalModel for explicit causal graph definition and refutation tests
- Refutation test results shown as a collapsible section (similar to PyMC trace plots in Tab 3)
- CATE (Conditional Average Treatment Effect) distribution plot — histogram of individual
  uplift scores
- Feature importance chart: which covariates drive heterogeneity in treatment response
- Uplift by decile chart: rank customers by predicted uplift, show actual spend lift per decile
  (Qini-style)
- Segment comparison: average predicted CATE for Men's campaign vs Women's campaign
- Plain-English interpretation: "which customers respond most, and why?"

### Tab 5: Multi-Arm Campaign Analysis
- OLS regression comparing all three arms simultaneously using statsmodels, with segment
  dummies and key covariates
- Coefficient plot: treatment effect estimates with 95% CIs for each arm
- Interaction analysis: does treatment effect vary by `channel`, `newbie` status, or
  `zip_code` (urban/rural proxy)?
- Marginal effects table by subgroup
- Plain-English interpretation paragraph

### Tab 6: Method Comparison
- Summary table: all methods side-by-side (ATT estimate, CI lower, CI upper, method, arm)
- Forest plot: visual comparison of all estimates with confidence/credible intervals — one
  panel per campaign arm (Men's, Women's)
- Commentary section: when to use each method, assumptions, and limitations
- "Key Takeaway" card summarising where methods agree and what the dashboard concludes

## UX & Design Requirements
- Sticky top navbar with app title and tab navigation
- Each tab has a collapsible "Methodology" section explaining assumptions and limitations in
  plain English — aimed at non-technical stakeholders
- All charts use a consistent dark Plotly theme (`plotly_dark` template)
- KPI cards use a consistent card component with metric, label, and delta indicator
- Responsive layout using dbc.Row / dbc.Col
- Loading spinners on all computationally heavy outputs (PSM bootstrap, PyMC sampling, model fitting)
- Currency displayed as $ throughout (Hillstrom is a US retailer)

## Code Quality Requirements
- Single-file app (app.py) is acceptable but callbacks must be clearly sectioned with comments
- All causal estimation logic extracted into a separate `causal_utils.py` module
- `requirements.txt` included
- `README.md` included with: project description, methods covered, how to run locally,
  screenshot placeholder, and a "What this demonstrates" section for portfolio context

## What This Project Must Demonstrate (Portfolio Signal)
- Ability to select and apply the right causal method for a given business question
- Understanding of assumptions behind each method (common support, SUTVA, ignorability, etc.)
- Communication of uncertainty — not just point estimates
- Individual-level causal thinking via HTE/uplift, not just average effects
- Translation of technical results into plain-English business language
