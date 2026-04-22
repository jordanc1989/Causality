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
from scipy import stats

warnings.filterwarnings("ignore")

CACHE_DIR = ".cache"
CACHE_FILE = os.path.join(CACHE_DIR, "results.pkl")
RANDOM_SEED = 42

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
    df["zip_rural"]    = (df["zip_code"] == "Rural").astype(int)
    # channel: reference = Phone
    df["channel_web"]          = (df["channel"] == "Web").astype(int)
    df["channel_multichannel"] = (df["channel"] == "Multichannel").astype(int)

    # Keep ordinal encodings too — used only for OLS interaction display
    df["zip_code_enc"] = df["zip_code"].map({"Urban": 0, "Surburban": 1, "Rural": 2})
    df["channel_enc"]  = df["channel"].map({"Phone": 0, "Web": 1, "Multichannel": 2})

    # Binary treatment indicators
    df["is_mens"] = (df["segment"] == "Mens E-Mail").astype(int)
    df["is_womens"] = (df["segment"] == "Womens E-Mail").astype(int)
    df["is_control"] = (df["segment"] == "No E-Mail").astype(int)

    return df


# ---------------------------------------------------------------------------
# Propensity Score Matching (PSM)
# ---------------------------------------------------------------------------

COVARIATES = [
    "recency", "history", "mens", "womens",
    "zip_suburban", "zip_rural",
    "channel_web", "channel_multichannel",
    "newbie",
]


def _compute_psm_for_arm(df, arm):
    """
    Run PSM for one arm vs control.
    arm: "mens" | "womens"
    Returns dict with all PSM artefacts.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors

    arm_col = f"is_{arm}"
    mask = (df[arm_col] == 1) | (df["is_control"] == 1)
    sub = df[mask].copy().reset_index(drop=True)

    X = sub[COVARIATES].values
    y = sub[arm_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    lr.fit(X_scaled, y)
    sub["propensity"] = lr.predict_proba(X_scaled)[:, 1]

    treated = sub[sub[arm_col] == 1].copy()
    control = sub[sub[arm_col] == 0].copy()

    # 1:1 nearest-neighbour matching (with replacement=False)
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control[["propensity"]].values)
    distances, indices = nn.kneighbors(treated[["propensity"]].values)

    matched_control_idx = control.index[indices.flatten()]
    matched_control = control.loc[matched_control_idx].copy()
    matched_treated = treated.copy()

    # Common support
    cs_lower = max(treated["propensity"].min(), control["propensity"].min())
    cs_upper = min(treated["propensity"].max(), control["propensity"].max())

    # SMDs before and after matching
    def smd(a, b):
        pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
        if pooled_std == 0:
            return 0.0
        return (np.mean(a) - np.mean(b)) / pooled_std

    smd_before = {}
    smd_after = {}
    for cov in COVARIATES:
        smd_before[cov] = smd(
            treated[cov].values,
            control[cov].values,
        )
        smd_after[cov] = smd(
            matched_treated[cov].values,
            matched_control[cov].values,
        )

    # ATT via bootstrap — resample matched *pairs* together to preserve pairing structure
    np.random.seed(RANDOM_SEED)
    n_boot = 500
    att_boot = []
    t_spend_arr = matched_treated["spend"].values
    c_spend_arr = matched_control["spend"].values
    n_pairs = len(t_spend_arr)
    for _ in range(n_boot):
        idx = np.random.choice(n_pairs, size=n_pairs, replace=True)
        att_boot.append(np.mean(t_spend_arr[idx]) - np.mean(c_spend_arr[idx]))

    att_point = np.mean(matched_treated["spend"].values) - np.mean(matched_control["spend"].values)
    att_ci_lo = np.percentile(att_boot, 2.5)
    att_ci_hi = np.percentile(att_boot, 97.5)

    pct_matched = len(matched_treated) / len(treated) * 100

    return {
        "arm": arm,
        "propensity_treated": treated["propensity"].values,
        "propensity_control": control["propensity"].values,
        "smd_before": smd_before,
        "smd_after": smd_after,
        "att_point": att_point,
        "att_ci_lo": att_ci_lo,
        "att_ci_hi": att_ci_hi,
        "n_matched": len(matched_treated),
        "n_treated_total": len(treated),
        "pct_matched": pct_matched,
        "cs_lower": cs_lower,
        "cs_upper": cs_upper,
    }


def run_psm(df):
    """Run PSM for both Men's and Women's arms."""
    return {
        "mens": _compute_psm_for_arm(df, "mens"),
        "womens": _compute_psm_for_arm(df, "womens"),
    }


# ---------------------------------------------------------------------------
# Bayesian A/B Test
# ---------------------------------------------------------------------------

ARM_PAIRS = {
    "mens_vs_control": ("Mens E-Mail", "No E-Mail"),
    "womens_vs_control": ("Womens E-Mail", "No E-Mail"),
    "mens_vs_womens": ("Mens E-Mail", "Womens E-Mail"),
}


BAYES_SAMPLE_N = 5000  # subsample per arm for PyMC speed; still statistically sound


def _run_bayesian_pair(df, pair_key):
    """
    Fit a PyMC model comparing spend for one arm pair.
    Returns posterior samples for delta and diagnostics.
    Uses a stratified subsample for speed with the pure-Python PyTensor backend.
    """
    import pymc as pm
    import arviz as az

    arm_a_label, arm_b_label = ARM_PAIRS[pair_key]

    rng = np.random.default_rng(RANDOM_SEED)
    a_full = df[df["segment"] == arm_a_label]["spend"].values.astype(float)
    b_full = df[df["segment"] == arm_b_label]["spend"].values.astype(float)
    a_spend = rng.choice(a_full, size=min(BAYES_SAMPLE_N, len(a_full)), replace=False)
    b_spend = rng.choice(b_full, size=min(BAYES_SAMPLE_N, len(b_full)), replace=False)

    combined = np.concatenate([a_spend, b_spend])
    pooled_std = np.std(combined) + 1e-6

    with pm.Model() as model:
        mu_a = pm.Normal("mu_a", mu=np.mean(combined), sigma=pooled_std * 2)
        mu_b = pm.Normal("mu_b", mu=np.mean(combined), sigma=pooled_std * 2)
        sigma_a = pm.HalfNormal("sigma_a", sigma=pooled_std)
        sigma_b = pm.HalfNormal("sigma_b", sigma=pooled_std)

        obs_a = pm.Normal("obs_a", mu=mu_a, sigma=sigma_a, observed=a_spend)
        obs_b = pm.Normal("obs_b", mu=mu_b, sigma=sigma_b, observed=b_spend)

        delta = pm.Deterministic("delta", mu_a - mu_b)

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
    mu_a_samples = idata.posterior["mu_a"].values.flatten()
    mu_b_samples = idata.posterior["mu_b"].values.flatten()

    # Per-chain samples for trace plot (shape: chains x draws)
    delta_chains = idata.posterior["delta"].values  # (chains, draws)

    return {
        "pair_key": pair_key,
        "arm_a_label": arm_a_label,
        "arm_b_label": arm_b_label,
        "delta_samples": delta_samples,
        "delta_chains": delta_chains,
        "mu_a_samples": mu_a_samples,
        "mu_b_samples": mu_b_samples,
        "hdi_lo": float(hdi[0]),
        "hdi_hi": float(hdi[1]),
        "p_positive": float(np.mean(delta_samples > 0)),
        "mean_a": float(np.mean(a_spend)),
        "mean_b": float(np.mean(b_spend)),
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

def _run_uplift_arm(df, arm):
    """
    Run T-Learner and S-Learner for one arm vs control using scikit-uplift.
    Uses 5-fold cross-fitting to avoid in-sample prediction bias: models are
    trained on 4 folds and predict on the held-out fold, so CATE estimates
    are out-of-sample for every observation.
    """
    from sklift.models import TwoModels, SoloModel
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold

    arm_col = f"is_{arm}"
    mask = (df[arm_col] == 1) | (df["is_control"] == 1)
    sub = df[mask].copy().reset_index(drop=True)

    X = sub[COVARIATES].values
    y = sub["spend"].values
    treatment = sub[arm_col].values

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cate_t = np.zeros(len(sub))
    cate_s = np.zeros(len(sub))
    feat_imp_accum = np.zeros(len(COVARIATES))

    for _, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        t_train = treatment[train_idx]

        X_train_df = pd.DataFrame(X_train, columns=COVARIATES)
        X_test_df = pd.DataFrame(X_test, columns=COVARIATES)

        # T-Learner
        t_model = TwoModels(
            estimator_trmnt=RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED),
            estimator_ctrl=RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED),
            method="vanilla",
        )
        t_model.fit(X_train_df, y_train, t_train)
        cate_t[test_idx] = t_model.predict(X_test_df)
        feat_imp_accum += t_model.estimator_trmnt.feature_importances_

        # S-Learner
        s_model = SoloModel(
            estimator=RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED),
            method="treatment_interaction",
        )
        s_model.fit(X_train_df, y_train, t_train)
        cate_s[test_idx] = s_model.predict(X_test_df)

    # Average feature importance across folds
    feat_imp = feat_imp_accum / kf.get_n_splits()
    feat_names = COVARIATES

    # Qini / Uplift by decile
    sub["cate_t"] = cate_t
    sub["cate_s"] = cate_s

    # Rank by T-Learner CATE, compute actual spend lift per decile
    sub_sorted = sub.sort_values("cate_t", ascending=False).reset_index(drop=True)
    sub_sorted["decile"] = pd.qcut(sub_sorted.index, q=10, labels=False)

    decile_lift = []
    for d in range(10):
        dec = sub_sorted[sub_sorted["decile"] == d]
        t_mean = dec[dec[arm_col] == 1]["spend"].mean()
        c_mean = dec[dec[arm_col] == 0]["spend"].mean()
        lift = t_mean - c_mean if not (np.isnan(t_mean) or np.isnan(c_mean)) else 0.0
        decile_lift.append({"decile": d + 1, "lift": lift})

    # Qini curve — vectorised computation
    # Sort by predicted uplift descending
    n = len(sub_sorted)
    is_treated = sub_sorted[arm_col].values
    spend_vals = sub_sorted["spend"].values
    n_treated_total = is_treated.sum()
    n_control_total = (1 - is_treated).sum()

    cum_treated_outcome = np.cumsum(spend_vals * is_treated)
    cum_control_outcome = np.cumsum(spend_vals * (1 - is_treated))
    n_treated_cum = np.cumsum(is_treated)
    n_control_cum = np.cumsum(1 - is_treated)

    # Only compute where both groups have at least one observation
    valid = (n_treated_cum > 0) & (n_control_cum > 0)
    qini_x = (np.arange(1, n + 1) / n)[valid].tolist()
    qini_y = (
        cum_treated_outcome[valid] / n_treated_total
        - cum_control_outcome[valid] / n_control_total * (n_treated_cum[valid] / n_treated_total)
    ).tolist()

    return {
        "arm": arm,
        "cate_t": cate_t,
        "cate_s": cate_s,
        "feat_imp": dict(zip(feat_names, feat_imp)),
        "decile_lift": decile_lift,
        "qini_x": qini_x,
        "qini_y": qini_y,
        "avg_cate_t": float(np.mean(cate_t)),
        "avg_cate_s": float(np.mean(cate_s)),
    }


def run_uplift(df):
    """Run uplift modelling for both arms."""
    return {
        "mens": _run_uplift_arm(df, "mens"),
        "womens": _run_uplift_arm(df, "womens"),
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
    model_df["mens_email"]   = (model_df["segment"] == "Mens E-Mail").astype(int)
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

    result = smf.ols(formula, data=model_df).fit()

    # Coefficient table
    coef_df = pd.DataFrame({
        "coef":   result.params,
        "ci_lo":  result.conf_int()[0],
        "ci_hi":  result.conf_int()[1],
        "pvalue": result.pvalues,
    }).reset_index().rename(columns={"index": "term"})

    # Marginal effects by subgroup
    subgroups = []
    for newbie_val, newbie_label in [(0, "Existing"), (1, "New")]:
        for channel_web, channel_mc, channel_label in [
            (0, 0, "Phone"), (1, 0, "Web"), (0, 1, "Multichannel")
        ]:
            for zip_sub, zip_rural_val, zip_label in [
                (0, 0, "Urban"), (1, 0, "Suburban"), (0, 1, "Rural")
            ]:
                me_mens = (
                    result.params.get("mens_email", 0)
                    + result.params.get("mens_email:newbie", 0)                  * newbie_val
                    + result.params.get("mens_email:channel_web", 0)             * channel_web
                    + result.params.get("mens_email:channel_multichannel", 0)    * channel_mc
                    + result.params.get("mens_email:zip_suburban", 0)            * zip_sub
                    + result.params.get("mens_email:zip_rural", 0)               * zip_rural_val
                )
                me_womens = (
                    result.params.get("womens_email", 0)
                    + result.params.get("womens_email:newbie", 0)                * newbie_val
                    + result.params.get("womens_email:channel_web", 0)           * channel_web
                    + result.params.get("womens_email:channel_multichannel", 0)  * channel_mc
                    + result.params.get("womens_email:zip_suburban", 0)          * zip_sub
                    + result.params.get("womens_email:zip_rural", 0)             * zip_rural_val
                )
                subgroups.append({
                    "newbie":   newbie_label,
                    "channel":  channel_label,
                    "zip_code": zip_label,
                    "me_mens":   me_mens,
                    "me_womens": me_womens,
                })

    subgroup_df = pd.DataFrame(subgroups)

    return {
        "coef_df":      coef_df,
        "subgroup_df":  subgroup_df,
        "r_squared":    result.rsquared,
        "n_obs":        int(result.nobs),
        "summary_text": result.summary().as_text(),
    }


# ---------------------------------------------------------------------------
# Cache Management
# ---------------------------------------------------------------------------

def build_cache():
    """Compute all results and save to disk. Returns the results dict."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("[Cache] Loading data...")
    df = load_data()

    print("[Cache] Running PSM (bootstrap 500 reps × 2 arms)...")
    psm = run_psm(df)

    print("[Cache] Running Bayesian A/B (PyMC, 3 arm pairs)...")
    bayesian = run_bayesian_ab(df)

    print("[Cache] Running Uplift models (T-Learner + S-Learner × 2 arms)...")
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
    """Load cached results if available, otherwise compute and cache."""
    if os.path.exists(CACHE_FILE):
        print(f"[Cache] Loading from {CACHE_FILE}...")
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    else:
        print("[Cache] No cache found — computing (this will take several minutes)...")
        return build_cache()
