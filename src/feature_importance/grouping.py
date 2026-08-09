"""Maps this project's real engineered feature names to feature groups,
for group-level permutation importance aggregation.

Groups are derived from the actual feature-engineering naming convention
(src/analysis/feature_engineering.py) and the real columns present across
all four feature configurations -- not invented. Five groups cover every
real predictor in this project:

  Price & Return       -- Bitcoin/Ethereum_Price_log_return and its lags,
                           momentum, and rolling MEAN (smoothed return level)
  Volatility            -- rolling STD of log returns (dispersion/risk,
                           conceptually distinct from return level)
  Trading Volume         -- all Volume-derived features (raw lags, pct
                           change, log, rolling mean/std)
  Google Trends           -- Search_Interest lags and first differences
  Fear & Greed / Sentiment -- Fear_Greed_Index (lags, diffs, rolling mean)
                              and Market_Sentiment_Encoded (the ordinal
                              encoding of the same underlying Fear & Greed
                              sentiment categories -- grouped together
                              since both come from the same data source,
                              not split into two groups for one source)

No "Market Capitalisation" group exists here because this project
deliberately excludes market cap from its predictive features (documented
elsewhere in the project: near-perfect correlation with price, r ~ 0.9997)
-- there is no real feature to assign to that group, so it is not
fabricated.
"""

import re

FEATURE_GROUPS = ["Price & Return", "Volatility", "Trading Volume", "Google Trends", "Fear & Greed / Sentiment", "Other"]


def assign_feature_group(feature_name: str) -> str:
    """Deterministic, real-naming-convention-based group assignment.
    Order matters: more specific patterns (rolling_std, momentum) are
    checked before the general Price/Volume catch-alls."""
    name = feature_name

    if "Search_Interest" in name:
        return "Google Trends"
    if "Fear_Greed_Index" in name or "Market_Sentiment_Encoded" in name:
        return "Fear & Greed / Sentiment"

    if "Volume" in name:
        return "Trading Volume"

    if "Price" in name:
        if "rolling_std" in name:
            return "Volatility"
        # log_return itself, its lags, rolling_mean, momentum, trailing-return
        return "Price & Return"

    return "Other"


def build_feature_dictionary(feature_names: list[str]) -> "pd.DataFrame":
    """TABLE 1: Feature | Feature Group | Source | Engineering Method |
    -- built from the real feature names actually passed in (e.g. one
    config's real X_test.columns), not invented."""
    import pandas as pd

    rows = []
    for feat in feature_names:
        group = assign_feature_group(feat)

        # Source: the asset/variable the feature derives from -- the
        # leading token(s) before the transform suffix, per this
        # project's real naming convention.
        if feat.startswith("Bitcoin_"):
            source = "Bitcoin"
        elif feat.startswith("Ethereum_"):
            source = "Ethereum"
        elif feat.startswith("Fear_Greed_Index") or feat.startswith("Market_Sentiment_Encoded"):
            source = "Fear & Greed Index (external)"
        else:
            source = "Unknown"

        # Engineering method: the transform suffix, read off the real name.
        method_parts = []
        if "log_return" in feat:
            method_parts.append("log return")
        elif feat.endswith("_log"):
            method_parts.append("log transform")
        if re.search(r"lag_\d+", feat):
            lag = re.search(r"lag_(\d+)", feat).group(1)
            method_parts.append(f"lag {lag}")
        if "diff" in feat:
            method_parts.append("first difference")
        if "rolling_mean" in feat:
            window = re.search(r"rolling_mean_(\d+)", feat)
            method_parts.append(f"rolling mean ({window.group(1)}d)" if window else "rolling mean")
        if "rolling_std" in feat:
            window = re.search(r"rolling_std_(\d+)", feat)
            method_parts.append(f"rolling std ({window.group(1)}d)" if window else "rolling std")
        if "momentum" in feat:
            window = re.search(r"momentum_(\d+)", feat)
            method_parts.append(f"momentum ({window.group(1)}d)" if window else "momentum")
        if "pct_change" in feat:
            method_parts.append("pct change")
        if "trailing" in feat and "return" in feat:
            method_parts.append("trailing return")
        if feat.startswith("Market_Sentiment_Encoded"):
            method_parts.append("ordinal sentiment encoding (0-4)")
        if not method_parts:
            method_parts.append("raw/level")

        rows.append({
            "feature": feat, "feature_group": group, "source": source,
            "engineering_method": " + ".join(method_parts),
        })

    return pd.DataFrame(rows)
