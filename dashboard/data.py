import causal_utils as cu

print("=" * 60)
print("Causal Inference Dashboard")
print("=" * 60)
RESULTS = cu.load_or_build_cache()
DF = RESULTS["df"]
PSM = RESULTS["psm"]
BAYESIAN = RESULTS["bayesian"]
UPLIFT = RESULTS["uplift"]
OLS = RESULTS["ols"]
print("Dashboard ready!")
