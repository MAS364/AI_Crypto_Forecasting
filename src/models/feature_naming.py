"""Feature-set naming conventions: internal keys, on-disk feature CSV
filename templates, and human-readable display names.

Deliberately dependency-free (only pandas would even be plausible here,
and even that isn't needed) -- this module exists specifically so code
that only needs these naming constants doesn't have to import
src.models.pipeline, which pulls in every model implementation
including src.models.lstm_model, which hard-imports TensorFlow at
module level.

src/explainability/utils.py imports from here, not from
src.models.pipeline, for exactly that reason: computing SHAP for a
Random Forest or XGBoost model has no legitimate reason to require
TensorFlow just to resolve a feature CSV's filename.

src.models.pipeline re-exports these same names (see the import there)
so every existing caller of e.g. `from src.models.pipeline import
FEATURE_FILE_TEMPLATES` continues to work identically -- this is a
relocation of where the values are defined, not a change to what any
existing caller receives.
"""

# Internal keys (unchanged, matching the feature CSV filenames produced by
# src/analysis/feature_engineering.py and 06_feature_engineering.ipynb) mapped
# to the descriptive names used in every saved output -- matching exactly
# the terminology used throughout the notebooks.
FEATURE_FILE_TEMPLATES = {
    "primary": "{a}_market_only_features{s}.csv",
    "candidate": "{a}_market_google_trends_features{s}.csv",
    "ablation": "{a}_market_fear_greed_features{s}.csv",
    "combined": "{a}_market_all_external_features{s}.csv",
}
DISPLAY_NAMES = {
    "primary": "Market Only",
    "candidate": "Market + Google Trends",
    "ablation": "Market + Fear & Greed Index",
    "combined": "Market + Google Trends + Fear & Greed Index",
}


def _horizon_suffix(horizon: int) -> str:
    """1-day uses the original, unsuffixed feature files; 7 and 14 use their
    own suffix so all three horizons' feature files coexist without any one
    overwriting another -- matching the convention already established in
    06_feature_engineering.ipynb."""
    return f"_{horizon}day" if horizon != 1 else ""
