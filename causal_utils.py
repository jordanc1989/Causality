"""
causal_utils.py
---------------
All causal estimation logic for the Hillstrom dashboard.
Results are cached to .cache/results.pkl on first run and loaded on subsequent runs.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd

# Narrow suppression: PyMC/ArviZ emit deprecation chatter on import; scikit-uplift
# emits a FutureWarning about pandas groupby. Keep everything else visible.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklift")

CACHE_DIR = ".cache"
CACHE_FILE = os.path.join(CACHE_DIR, "results.pkl")
RANDOM_SEED = 42

# Set to False to force a full recompute (and overwrite the pickle) the next
# time the app starts — use after any change to this file's estimation logic.
# Leave True for production/Plotly Cloud so the pre-computed pickle is loaded
# instantly instead of re-running ~3-5 min of PSM/Bayesian/uplift fits.
USE_CACHE = True

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_data():
    """Load and preprocess the Hillstrom dataset."""
    from sklift.datasets import fetch_hillstrom

    bunch = fetch_hillstrom(target_col="all")
    # bunch["data"] = features, bunch["target"] = visit/conversion/spend, bunch["treatment"] = segment
    df = bunch["data"].copy()
    df["segment"] = bunch["treatment"].values
    target_df = bunch["target"]
    for col in target_df.columns:
        df[col] = target_df[col].values

    # One-hot encode categoricals (drop_first=False so all levels are explicit;
    # we drop one reference level manually to avoid perfect multicollinearity)
    # zip_code: reference = Urban
    df["zip_suburban"] = (df["zip_code"] == "Surburban").astype(int)
    df["zip_rural"] = (df["zip_code"] == "Rural").astype(int)
    # channel: reference = Phone
    df["channel_web"] = (df["channel"] == "Web").astype(int)
    df["channel_multichannel"] = (df["channel"] == "Multichannel").astype(int)

    # Keep ordinal encodings too — used only for OLS interaction display
    df["zip_code_enc"] = df["zip_code"].map({"Urban": 0, "Surburban": 1, "Rural": 2})
    df["channel_enc"] = df["channel"].map({"Phone": 0, "Web": 1, "Multichannel": 2})

    # Binary treatment indicators
    df["is_mens"] = (df["segment"] == "Mens E-Mail").astype(int)
    df["is_womens"] = (df["segment"] == "Womens E-Mail").astype(int)
    df["is_control"] = (df["segment"] == "No E-Mail").astype(int)

    return df


# ---------------------------------------------------------------------------
# Propensity Score Matching (PSM)
# ---------------------------------------------------------------------------

COVARIATES = [
    "recency",
    "history",
    "mens",
    "womens",
    "zip_suburban",
    "zip_rural",
    "channel_web",
    "channel_multichannel",
    "newbie"
]


def _fit_propensity_and_match(sub, arm_col, seed):
    """
    Fit a logistic propensity model on the given sub-sample and run
    1:1 nearest-neighbour matching with replacement. Returns matched
    treated/control spend arrays and PS distances.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors

    X = sub[COVARIATES].values
    y = sub[arm_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(X_scaled, y)
    propensity = lr.predict_proba(X_scaled)[:, 1]

    sub = sub.copy()
    sub["propensity"] = propensity
    treated = sub[sub[arm_col] == 1]
    control = sub[sub[arm_col] == 0]

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control[["propensity"]].values)
    distances, indices = nn.kneighbors(treated[["propensity"]].values)

    matched_control = control.iloc[indices.flatten()]
    return treated, control, matched_control, distances


def _compute_psm_for_arm(df, arm):
    """
    Run PSM for one arm vs control.

    ATT is estimated via a *causal* bootstrap: each replicate resamples the
    combined treated+control pool, re-fits the propensity score, re-matches,
    and recomputes the ATT. This correctly propagates uncertainty from the
    propensity-score estimation and the matching step, and avoids the naive
    pair-level bootstrap which is invalid for NN matching with replacement
    (Abadie & Imbens, 2006, 2008).

    arm: "mens" | "womens"
    Returns dict with all PSM artefacts.
    """
    arm_col = f"is_{arm}"
    mask = (df[arm_col] == 1) | (df["is_control"] == 1)
    sub = df[mask].copy().reset_index(drop=True)

    # Point estimate
    treated, control, matched_control, distances = _fit_propensity_and_match(
        sub, arm_col, RANDOM_SEED
    )
    att_point = float(
        np.mean(treated["spend"].values) - np.mean(matched_control["spend"].values)
    )
    avg_ps_distance = float(np.mean(distances.flatten()))

    # Common support on point-estimate propensities
    cs_lower = float(
        max(treated["propensity"].min(), control["propensity"].min())
    )
    cs_upper = float(
        min(treated["propensity"].max(), control["propensity"].max())
    )
    n_outside_support = int(
        ((treated["propensity"] < cs_lower) | (treated["propensity"] > cs_upper)).sum()
    )

    # SMDs before and after matching (point estimate matching)
    def smd(a, b):
        pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
        if pooled_std == 0:
            return 0.0
        return (np.mean(a) - np.mean(b)) / pooled_std

    smd_before = {}
    smd_after = {}
    for cov in COVARIATES:
        smd_before[cov] = smd(treated[cov].values, control[cov].values)
        smd_after[cov] = smd(treated[cov].values, matched_control[cov].values)

    # Causal bootstrap: resample the pooled sub-sample with replacement,
    # re-fit propensity and re-match on each replicate. 200 reps gives
    # stable 95% percentile CIs at reasonable compute cost (~2-4 min/arm).
    rng = np.random.default_rng(RANDOM_SEED)
    n_boot = 200
    n_total = len(sub)
    att_boot = []
    for b in range(n_boot):
        idx = rng.integers(0, n_total, size=n_total)
        boot_sub = sub.iloc[idx].reset_index(drop=True)
        # Need both treated and control in the bootstrap sample
        if boot_sub[arm_col].sum() < 10 or (boot_sub[arm_col] == 0).sum() < 10:
            continue
        try:
            t_b, _, mc_b, _ = _fit_propensity_and_match(
                boot_sub, arm_col, RANDOM_SEED + b
            )
            att_boot.append(
                float(np.mean(t_b["spend"].values) - np.mean(mc_b["spend"].values))
            )
        except Exception:
            # e.g. singular matrix on degenerate sample — skip
            continue

    att_boot = np.array(att_boot)
    att_ci_lo = float(np.percentile(att_boot, 2.5))
    att_ci_hi = float(np.percentile(att_boot, 97.5))
    pct_matched = len(treated) / len(treated) * 100  # always 100% (no caliper)

    return {
        "arm": arm,
        "propensity_treated": treated["propensity"].values,
        "propensity_control": control["propensity"].values,
        "smd_before": smd_before,
        "smd_after": smd_after,
        "att_point": att_point,
        "att_ci_lo": att_ci_lo,
        "att_ci_hi": att_ci_hi,
        "n_matched": len(treated),
        "n_treated_total": len(treated),
        "pct_matched": pct_matched,
        "avg_ps_distance": avg_ps_distance,
        "cs_lower": cs_lower,
        "cs_upper": cs_upper,
        "n_outside_support": n_outside_support,
        "n_boot_successful": len(att_boot),
    }


def run_psm(df):
    """Run PSM for both Men's and Women's arms."""
    return {
        "mens": _compute_psm_for_arm(df, "mens"),
        "womens": _compute_psm_for_arm(df, "womens")
    }


# ---------------------------------------------------------------------------
# Bayesian A/B Test
# ---------------------------------------------------------------------------

ARM_PAIRS = {
    "mens_vs_control": ("Mens E-Mail", "No E-Mail"),
    "womens_vs_control": ("Womens E-Mail", "No E-Mail"),
    "mens_vs_womens": ("Mens E-Mail", "Womens E-Mail")
}


def _run_bayesian_pair(df, pair_key):
    """
    Fit a two-part (hurdle) PyMC model comparing spend for one arm pair.

    Spend is ~99% zeros with a right-skewed positive tail. A plain Normal
    likelihood is a severe misspecification (it assigns mass to negative
    spend and the sigma term is uninterpretable). We instead decompose:

        spend = conversion * amount_given_conversion

    with:
        conversion  ~ Bernoulli(p)           # models whether the customer spends
        amount      ~ LogNormal(mu, sigma)   # models how much, given spend > 0

    The per-customer expected spend is E[spend] = p * exp(mu + sigma**2/2),
    and `delta` is the difference between the two arms' expected spend.
    This is the target of interest for the A/B comparison and is on the
    same dollar scale as the raw arm means, so it is directly comparable
    to PSM's ATT and OLS's main effect.

    Runs on the full arm data (no subsampling) via the nutpie NUTS
    sampler — fits in a few seconds for ~20-40k observations per arm.
    """
    import pymc as pm
    import arviz as az

    arm_a_label, arm_b_label = ARM_PAIRS[pair_key]

    a_spend = df[df["segment"] == arm_a_label]["spend"].values.astype(float)
    b_spend = df[df["segment"] == arm_b_label]["spend"].values.astype(float)

    a_converted = (a_spend > 0).astype(int)
    b_converted = (b_spend > 0).astype(int)

    a_pos = a_spend[a_spend > 0]
    b_pos = b_spend[b_spend > 0]

    # Weakly informative prior on log-spend, centred on pooled log mean
    log_pooled = np.log(np.concatenate([a_pos, b_pos]))
    log_mu_prior = float(np.mean(log_pooled))
    log_sd_prior = float(np.std(log_pooled))

    with pm.Model() as model:
        # Conversion probability (Beta(1,1) = uniform)
        p_a = pm.Beta("p_a", alpha=1.0, beta=1.0)
        p_b = pm.Beta("p_b", alpha=1.0, beta=1.0)
        pm.Bernoulli("obs_conv_a", p=p_a, observed=a_converted)
        pm.Bernoulli("obs_conv_b", p=p_b, observed=b_converted)

        # Log-amount among converters
        mu_log_a = pm.Normal("mu_log_a", mu=log_mu_prior, sigma=log_sd_prior * 2)
        mu_log_b = pm.Normal("mu_log_b", mu=log_mu_prior, sigma=log_sd_prior * 2)
        sigma_log_a = pm.HalfNormal("sigma_log_a", sigma=log_sd_prior)
        sigma_log_b = pm.HalfNormal("sigma_log_b", sigma=log_sd_prior)

        pm.LogNormal(
            "obs_amount_a", mu=mu_log_a, sigma=sigma_log_a, observed=a_pos
        )
        pm.LogNormal(
            "obs_amount_b", mu=mu_log_b, sigma=sigma_log_b, observed=b_pos
        )

        # Expected per-customer spend: E[spend] = P(convert) * E[amount | convert]
        exp_spend_a = pm.Deterministic(
            "exp_spend_a",
            p_a * pm.math.exp(mu_log_a + 0.5 * sigma_log_a**2),
        )
        exp_spend_b = pm.Deterministic(
            "exp_spend_b",
            p_b * pm.math.exp(mu_log_b + 0.5 * sigma_log_b**2),
        )
        pm.Deterministic("delta", exp_spend_a - exp_spend_b)

        idata = pm.sample(
            draws=2000,
            tune=1000,
            chains=2,
            nuts_sampler="nutpie",
            random_seed=RANDOM_SEED,
            progressbar=True,
            return_inferencedata=True,
        )

    delta_samples = idata.posterior["delta"].values.flatten()
    hdi = az.hdi(idata, var_names=["delta"], hdi_prob=0.95)["delta"].values

    report_vars = ["delta", "exp_spend_a", "exp_spend_b", "p_a", "p_b",
                   "mu_log_a", "mu_log_b", "sigma_log_a", "sigma_log_b"]
    diagnostics = az.summary(idata, var_names=report_vars, round_to=3)
    rhat_delta = float(diagnostics.loc["delta", "r_hat"])
    bulk_ess_delta = float(diagnostics.loc["delta", "ess_bulk"])
    tail_ess_delta = float(diagnostics.loc["delta", "ess_tail"])

    diag_rows = []
    for var in report_vars:
        if var in diagnostics.index:
            diag_rows.append(
                {
                    "parameter": var,
                    "r_hat": diagnostics.loc[var, "r_hat"],
                    "ess_bulk": diagnostics.loc[var, "ess_bulk"],
                    "ess_tail": diagnostics.loc[var, "ess_tail"],
                    "mean": diagnostics.loc[var, "mean"],
                    "sd": diagnostics.loc[var, "sd"],
                }
            )

    delta_chains = idata.posterior["delta"].values  # (chains, draws)

    # Posterior predictive replicates for a PPC plot in the Bayesian tab
    # (one-shot generation; kept small to fit in the cache).
    ppc_a = None
    ppc_b = None
    try:
        with model:
            post_pred = pm.sample_posterior_predictive(
                idata,
                var_names=["obs_amount_a", "obs_amount_b"],
                random_seed=RANDOM_SEED,
                progressbar=False,
            )
        # Take the first 500 predictive draws for plotting
        ppc_a = post_pred.posterior_predictive["obs_amount_a"].values[0, :500].flatten()
        ppc_b = post_pred.posterior_predictive["obs_amount_b"].values[0, :500].flatten()
    except Exception:
        pass

    return {
        "pair_key": pair_key,
        "arm_a_label": arm_a_label,
        "arm_b_label": arm_b_label,
        "delta_samples": delta_samples,
        "delta_chains": delta_chains,
        "exp_spend_a_samples": idata.posterior["exp_spend_a"].values.flatten(),
        "exp_spend_b_samples": idata.posterior["exp_spend_b"].values.flatten(),
        "hdi_lo": float(hdi[0]),
        "hdi_hi": float(hdi[1]),
        "p_positive": float(np.mean(delta_samples > 0)),
        "mean_a": float(np.mean(a_spend)),
        "mean_b": float(np.mean(b_spend)),
        "rhat_delta": rhat_delta,
        "bulk_ess_delta": bulk_ess_delta,
        "tail_ess_delta": tail_ess_delta,
        "diagnostics_table": diag_rows,
        "ppc_amount_a": ppc_a,
        "ppc_amount_b": ppc_b,
        "observed_amount_a": a_pos,
        "observed_amount_b": b_pos,
    }


def run_bayesian_ab(df):
    """Run Bayesian A/B for all three arm pairs."""
    results = {}
    for key in ARM_PAIRS:
        print(f"  [Bayesian] Fitting {key}...")
        results[key] = _run_bayesian_pair(df, key)
    return results


# ---------------------------------------------------------------------------
# Uplift Modelling / HTE
# ---------------------------------------------------------------------------


def _qini_curve_continuous(sorted_df, arm_col):
    """
    Radcliffe (2007) Qini curve for continuous outcomes.

    For a population ranked by predicted uplift (descending), the canonical
    Qini value at rank k is:

        Q(k) = R_T(k) - R_C(k) * (N_T(k) / N_C(k))

    where R_T(k) and R_C(k) are cumulative outcomes in treated/control and
    N_T(k), N_C(k) are cumulative treated/control counts. At k=N this equals
    R_T_total - R_C_total * (N_T/N_C), i.e. the total incremental revenue
    captured over random targeting. The x-axis is the fraction of population
    targeted. scikit-uplift's `qini_curve` enforces binary outcomes and can't
    be used on `spend` directly — this is the same formula, generalised.
    """
    n_rows = len(sorted_df)
    is_t = sorted_df[arm_col].values.astype(float)
    y_vals = sorted_df["spend"].values.astype(float)

    cum_t = np.cumsum(y_vals * is_t)
    cum_c = np.cumsum(y_vals * (1 - is_t))
    n_t_cum = np.cumsum(is_t)
    n_c_cum = np.cumsum(1 - is_t)

    valid = (n_t_cum > 0) & (n_c_cum > 0)
    xs = (np.arange(1, n_rows + 1) / n_rows)[valid].tolist()
    ys = (cum_t[valid] - cum_c[valid] * (n_t_cum[valid] / n_c_cum[valid])).tolist()

    # Prepend origin for a clean plot
    if xs and xs[0] > 0:
        xs = [0.0] + xs
        ys = [0.0] + ys
    return xs, ys


def _run_uplift_arm(df, arm):
    """
    Run T-Learner and S-Learner for one arm vs control using scikit-uplift.

    Uses 5-fold *stratified* cross-fitting on the treatment indicator so every
    fold has both arms represented, which matters for small-sample fold fits.
    CATE estimates are out-of-sample for every observation.

    Feature importance is reported as |treated-model importance - control-model
    importance| from the T-Learner: features the two outcome models use
    differently are the ones driving heterogeneous treatment effects. Plain
    `estimator_trmnt.feature_importances_` would instead tell you what predicts
    spend in the treated group — a different question.
    """
    from sklift.models import TwoModels, SoloModel
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import StratifiedKFold

    arm_col = f"is_{arm}"
    mask = (df[arm_col] == 1) | (df["is_control"] == 1)
    sub = df[mask].copy().reset_index(drop=True)

    X = sub[COVARIATES].values
    y = sub["spend"].values
    treatment = sub[arm_col].values

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cate_t = np.zeros(len(sub))
    cate_s = np.zeros(len(sub))
    feat_imp_diff_accum = np.zeros(len(COVARIATES))

    for _, (train_idx, test_idx) in enumerate(kf.split(X, treatment)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        t_train = treatment[train_idx]

        X_train_df = pd.DataFrame(X_train, columns=COVARIATES)
        X_test_df = pd.DataFrame(X_test, columns=COVARIATES)

        # T-Learner
        t_model = TwoModels(
            estimator_trmnt=RandomForestRegressor(
                n_estimators=100, random_state=RANDOM_SEED
            ),
            estimator_ctrl=RandomForestRegressor(
                n_estimators=100, random_state=RANDOM_SEED
            ),
            method="vanilla",
        )
        t_model.fit(X_train_df, y_train, t_train)
        cate_t[test_idx] = t_model.predict(X_test_df)
        feat_imp_diff_accum += np.abs(
            t_model.estimator_trmnt.feature_importances_
            - t_model.estimator_ctrl.feature_importances_
        )

        # S-Learner
        s_model = SoloModel(
            estimator=RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED),
            method="treatment_interaction",
        )
        s_model.fit(X_train_df, y_train, t_train)
        cate_s[test_idx] = s_model.predict(X_test_df)

    feat_imp_diff = feat_imp_diff_accum / kf.get_n_splits()
    # Normalise to sum to 1 for display consistency with the old chart
    feat_imp_norm = (
        feat_imp_diff / feat_imp_diff.sum() if feat_imp_diff.sum() > 0 else feat_imp_diff
    )

    sub["cate_t"] = cate_t
    sub["cate_s"] = cate_s

    def _decile_lift(sub_sorted):
        """Compute actual spend lift per decile for a population sorted by predicted CATE."""
        sub_sorted = sub_sorted.reset_index(drop=True)
        sub_sorted["decile"] = pd.qcut(sub_sorted.index, q=10, labels=False)
        rows = []
        for d in range(10):
            dec = sub_sorted[sub_sorted["decile"] == d]
            t_mean = dec[dec[arm_col] == 1]["spend"].mean()
            c_mean = dec[dec[arm_col] == 0]["spend"].mean()
            lift = (
                t_mean - c_mean
                if not (np.isnan(t_mean) or np.isnan(c_mean))
                else 0.0
            )
            rows.append({"decile": d + 1, "lift": lift})
        return rows

    sub_sorted_t = sub.sort_values("cate_t", ascending=False)
    sub_sorted_s = sub.sort_values("cate_s", ascending=False)

    decile_lift = _decile_lift(sub_sorted_t)
    decile_lift_s = _decile_lift(sub_sorted_s)

    qini_x, qini_y = _qini_curve_continuous(sub_sorted_t, arm_col)
    qini_x_s, qini_y_s = _qini_curve_continuous(sub_sorted_s, arm_col)

    # Qini coefficient (normalised AUC) — trapezoidal area vs. the zero line.
    # Positive = model ranks uplift better than random targeting.
    def _qini_auc(xs, ys):
        if len(xs) < 2:
            return 0.0
        # `trapezoid` replaced `trapz` in numpy 2.0; fall back for older envs
        trapz = getattr(np, "trapezoid", None) or np.trapz
        return float(trapz(ys, xs))

    return {
        "arm": arm,
        "cate_t": cate_t,
        "cate_s": cate_s,
        "feat_imp": dict(zip(COVARIATES, feat_imp_norm)),
        "feat_imp_label": "Heterogeneity importance (|treat - ctrl| model diff)",
        "decile_lift": decile_lift,
        "decile_lift_s": decile_lift_s,
        "qini_x": qini_x,
        "qini_y": qini_y,
        "qini_x_s": qini_x_s,
        "qini_y_s": qini_y_s,
        "qini_auc_t": _qini_auc(qini_x, qini_y),
        "qini_auc_s": _qini_auc(qini_x_s, qini_y_s),
        "avg_cate_t": float(np.mean(cate_t)),
        "avg_cate_s": float(np.mean(cate_s))
    }


def run_uplift(df):
    """Run uplift modelling for both arms."""
    return {
        "mens": _run_uplift_arm(df, "mens"),
        "womens": _run_uplift_arm(df, "womens")
    }


# ---------------------------------------------------------------------------
# Multi-Arm OLS
# ---------------------------------------------------------------------------


def run_ols(df):
    """
    OLS regression with treatment dummies, covariates, and interaction terms.
    Outcome: spend. Categoricals (zip_code, channel) are one-hot encoded;
    reference levels are Urban and Phone respectively.
    """
    import statsmodels.formula.api as smf

    model_df = df.copy()
    model_df["mens_email"] = (model_df["segment"] == "Mens E-Mail").astype(int)
    model_df["womens_email"] = (model_df["segment"] == "Womens E-Mail").astype(int)

    # Main effects + interactions with OHE categoricals
    formula = (
        "spend ~ mens_email + womens_email "
        "+ recency + history + newbie "
        "+ zip_suburban + zip_rural "
        "+ channel_web + channel_multichannel "
        "+ mens_email:newbie + womens_email:newbie "
        "+ mens_email:channel_web + womens_email:channel_web "
        "+ mens_email:channel_multichannel + womens_email:channel_multichannel "
        "+ mens_email:zip_suburban + womens_email:zip_suburban "
        "+ mens_email:zip_rural + womens_email:zip_rural"
    )

    # HC3 heteroscedasticity-robust standard errors: spend is right-skewed with
    # variance that scales with the treatment means, so default OLS SEs would
    # be biased. HC3 is the recommended small-sample-corrected White estimator.
    result = smf.ols(formula, data=model_df).fit(cov_type="HC3")

    # Coefficient table
    _ci = result.conf_int()
    coef_df = (
        pd.DataFrame(
            {
                "coef": result.params,
                "ci_lo": _ci[0],
                "ci_hi": _ci[1],
                "pvalue": result.pvalues
            }
        )
        .reset_index()
        .rename(columns={"index": "term"})
    )

    # Marginal effects by subgroup
    subgroups = []
    for newbie_val, newbie_label in [(0, "Existing"), (1, "New")]:
        for channel_web, channel_mc, channel_label in [
            (0, 0, "Phone"),
            (1, 0, "Web"),
            (0, 1, "Multichannel")
        ]:
            for zip_sub, zip_rural_val, zip_label in [
                (0, 0, "Urban"),
                (1, 0, "Suburban"),
                (0, 1, "Rural")
            ]:
                me_mens = (
                    result.params.get("mens_email", 0)
                    + result.params.get("mens_email:newbie", 0) * newbie_val
                    + result.params.get("mens_email:channel_web", 0) * channel_web
                    + result.params.get("mens_email:channel_multichannel", 0)
                    * channel_mc
                    + result.params.get("mens_email:zip_suburban", 0) * zip_sub
                    + result.params.get("mens_email:zip_rural", 0) * zip_rural_val
                )
                me_womens = (
                    result.params.get("womens_email", 0)
                    + result.params.get("womens_email:newbie", 0) * newbie_val
                    + result.params.get("womens_email:channel_web", 0) * channel_web
                    + result.params.get("womens_email:channel_multichannel", 0)
                    * channel_mc
                    + result.params.get("womens_email:zip_suburban", 0) * zip_sub
                    + result.params.get("womens_email:zip_rural", 0) * zip_rural_val
                )
                subgroups.append(
                    {
                        "newbie": newbie_label,
                        "channel": channel_label,
                        "zip_code": zip_label,
                        "me_mens": me_mens,
                        "me_womens": me_womens
                    }
                )

    subgroup_df = pd.DataFrame(subgroups)

    # Population-weighted ATE and its HC3 CI via the delta method.
    # The `mens_email` / `womens_email` coefficients on their own are the
    # effect for the reference subgroup (Existing + Phone + Urban); they are
    # not directly comparable to PSM's ATT or the Bayesian delta. The ATE
    # below is the average of the linear-prediction marginal effects over
    # the actual sample distribution of the covariates, which IS on the same
    # scale as the other methods.
    def _ate_with_ci(arm_prefix):
        term_main = arm_prefix
        inter_terms = {
            f"{arm_prefix}:newbie": float(model_df["newbie"].mean()),
            f"{arm_prefix}:channel_web": float(model_df["channel_web"].mean()),
            f"{arm_prefix}:channel_multichannel": float(
                model_df["channel_multichannel"].mean()
            ),
            f"{arm_prefix}:zip_suburban": float(model_df["zip_suburban"].mean()),
            f"{arm_prefix}:zip_rural": float(model_df["zip_rural"].mean()),
        }
        params = result.params
        cov = result.cov_params()
        contrast = np.zeros(len(params))
        if term_main in params.index:
            contrast[params.index.get_loc(term_main)] = 1.0
        for t, w in inter_terms.items():
            if t in params.index:
                contrast[params.index.get_loc(t)] = w
        ate = float(contrast @ params.values)
        se = float(np.sqrt(contrast @ cov.values @ contrast))
        return ate, ate - 1.96 * se, ate + 1.96 * se

    ate_mens, ate_mens_lo, ate_mens_hi = _ate_with_ci("mens_email")
    ate_womens, ate_womens_lo, ate_womens_hi = _ate_with_ci("womens_email")

    return {
        "coef_df": coef_df,
        "subgroup_df": subgroup_df,
        "r_squared": result.rsquared,
        "n_obs": int(result.nobs),
        "summary_text": result.summary().as_text(),
        "ate_mens": ate_mens,
        "ate_mens_lo": ate_mens_lo,
        "ate_mens_hi": ate_mens_hi,
        "ate_womens": ate_womens,
        "ate_womens_lo": ate_womens_lo,
        "ate_womens_hi": ate_womens_hi,
    }


# ---------------------------------------------------------------------------
# Cache Management
# ---------------------------------------------------------------------------


def build_cache():
    """Compute all results and save to disk. Returns the results dict."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("[Cache] Loading data...")
    df = load_data()

    print("[Cache] Running PSM (causal bootstrap 200 reps x 2 arms, ~4-8 min)...")
    psm = run_psm(df)

    print("[Cache] Running Bayesian A/B (PyMC, 3 arm pairs)...")
    bayesian = run_bayesian_ab(df)

    print("[Cache] Running Uplift models (T-Learner + S-Learner w/ 2 arms)...")
    uplift = run_uplift(df)

    print("[Cache] Running Multi-Arm OLS...")
    ols = run_ols(df)

    results = {
        "df": df,
        "psm": psm,
        "bayesian": bayesian,
        "uplift": uplift,
        "ols": ols,
    }

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(results, f)

    print(f"[Cache] Saved to {CACHE_FILE}")
    return results


def load_or_build_cache():
    """Load cached results if USE_CACHE and a pickle exists, otherwise recompute."""
    if USE_CACHE and os.path.exists(CACHE_FILE):
        print(f"[Cache] USE_CACHE=True — loading from {CACHE_FILE}...")
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    if not USE_CACHE:
        print("[Cache] USE_CACHE=False — forcing rebuild (this will take several minutes)...")
    else:
        print("[Cache] No cache found — computing (this will take several minutes)...")
    return build_cache()
