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

warnings.filterwarnings("ignore")

CACHE_DIR = ".cache"
CACHE_FILE = os.path.join(CACHE_DIR, "results.pkl")
RANDOM_SEED = 10

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

    # 1:1 nearest-neighbour matching WITH replacement. Each treated unit is
    # paired with its nearest control by propensity score independently, so a
    # given control may serve as the match for more than one treated unit.
    # Trade-off: minimises matching bias at the cost of slightly inflated
    # variance (quantified by the bootstrap CI below).
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

    att_point = np.mean(matched_treated["spend"].values) - np.mean(
        matched_control["spend"].values
    )
    att_ci_lo = np.percentile(att_boot, 2.5)
    att_ci_hi = np.percentile(att_boot, 97.5)
    avg_ps_distance = float(np.mean(distances.flatten()))
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
        "avg_ps_distance": avg_ps_distance,
        "cs_lower": cs_lower,
        "cs_upper": cs_upper
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


BAYES_SAMPLE_N = 5000  # subsample per arm for PyMC speed, still statistically sound


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

    # MCMC diagnostics for all parameters
    diagnostics = az.summary(
        idata, var_names=["delta", "mu_a", "mu_b", "sigma_a", "sigma_b"], round_to=3
    )
    rhat_delta = float(diagnostics.loc["delta", "r_hat"])
    bulk_ess_delta = float(diagnostics.loc["delta", "ess_bulk"])
    tail_ess_delta = float(diagnostics.loc["delta", "ess_tail"])

    # Build diagnostics table
    diag_rows = []
    for var in ["delta", "mu_a", "mu_b", "sigma_a", "sigma_b"]:
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
        "rhat_delta": rhat_delta,
        "bulk_ess_delta": bulk_ess_delta,
        "tail_ess_delta": tail_ess_delta,
        "diagnostics_table": diag_rows,
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

    def _qini_curve(sorted_df):
        """Compute modified Qini curve for a DataFrame already sorted by predicted uplift."""
        n_rows = len(sorted_df)
        is_t = sorted_df[arm_col].values
        y_vals = sorted_df["spend"].values
        n_t_total = is_t.sum()
        n_c_total = (1 - is_t).sum()
        cum_t = np.cumsum(y_vals * is_t)
        cum_c = np.cumsum(y_vals * (1 - is_t))
        n_t_cum = np.cumsum(is_t)
        n_c_cum = np.cumsum(1 - is_t)
        valid = (n_t_cum > 0) & (n_c_cum > 0)
        xs = (np.arange(1, n_rows + 1) / n_rows)[valid].tolist()
        ys = (
            cum_t[valid] / n_t_total
            - cum_c[valid] / n_c_total * (n_t_cum[valid] / n_t_total)
        ).tolist()
        return xs, ys

    # T-Learner ranking — decile lift and Qini
    qini_x, qini_y = _qini_curve(sub_sorted)

    # S-Learner ranking — decile lift and Qini
    sub_sorted_s = sub.sort_values("cate_s", ascending=False).reset_index(drop=True)
    sub_sorted_s["decile"] = pd.qcut(sub_sorted_s.index, q=10, labels=False)

    decile_lift_s = []
    for d in range(10):
        dec = sub_sorted_s[sub_sorted_s["decile"] == d]
        t_mean = dec[dec[arm_col] == 1]["spend"].mean()
        c_mean = dec[dec[arm_col] == 0]["spend"].mean()
        lift = t_mean - c_mean if not (np.isnan(t_mean) or np.isnan(c_mean)) else 0.0
        decile_lift_s.append({"decile": d + 1, "lift": lift})

    qini_x_s, qini_y_s = _qini_curve(sub_sorted_s)

    return {
        "arm": arm,
        "cate_t": cate_t,
        "cate_s": cate_s,
        "feat_imp": dict(zip(feat_names, feat_imp)),
        "decile_lift": decile_lift,
        "decile_lift_s": decile_lift_s,
        "qini_x": qini_x,
        "qini_y": qini_y,
        "qini_x_s": qini_x_s,
        "qini_y_s": qini_y_s,
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

    result = smf.ols(formula, data=model_df).fit()

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

    return {
        "coef_df": coef_df,
        "subgroup_df": subgroup_df,
        "r_squared": result.rsquared,
        "n_obs": int(result.nobs),
        "summary_text": result.summary().as_text()
    }


# ---------------------------------------------------------------------------
# Cache Management
# ---------------------------------------------------------------------------


def build_cache():
    """Compute all results and save to disk. Returns the results dict."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("[Cache] Loading data...")
    df = load_data()

    print("[Cache] Running PSM (bootstrap 500 reps x 2 arms)...")
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
