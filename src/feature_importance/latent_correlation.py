"""Descriptive correlation between the 8 autoencoder latent dimensions
(encoded_0..encoded_7) and the real original engineered features they
were compressed from.

This is a DESCRIPTIVE analysis only. A latent dimension correlating
strongly with one or two original features does not mean the autoencoder
"learned" that variable specifically, and does not establish causality --
the autoencoder is trained to reconstruct the whole feature vector at
once, so any single latent dimension is free to encode a mixture of
several original signals. Every function and output column in this
module exists to support that reading, not to claim a one-to-one mapping.

Row alignment
-------------
The encoded test CSV (results/autoencoder/encoded_features/{prefix}_
encoded_test.csv) and the matching original engineered-feature CSV
(data/processed/features/{...}.csv) are two different files that must be
aligned correctly before any correlation is computed. This module never
assumes matching row order or matching row counts -- it merges the two
frames on the real "Date" column (an inner join), which is the only
column both files guarantee, and reports the number of rows that
actually matched (`number_of_aligned_observations`). Both files ultimately
descend from the same source feature file for a given (asset, horizon,
feature_set_key) -- see src/pipeline.py's _train_autoencoders_if_missing(),
which builds the encoded CSVs directly from split_data() on that same
file -- so in practice every row is expected to align, but the merge is
still done explicitly on Date rather than assumed by position, exactly
because "same number of rows" is not proof of correct alignment.

Split used
----------
This module correlates the TEST partition only, the same partition
Random Forest and XGBoost were evaluated on and the same partition
01_autoencoder_permutation_importance.py / run_feature_importance_grid.py
compute permutation importance against -- so a reader looking at PFI and
these correlations side by side is looking at the same rows, not two
different subsets of history.
"""

import sys
import logging
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src import config
from src.exception import CustomException
from src.models.feature_naming import FEATURE_FILE_TEMPLATES, _horizon_suffix

logger = logging.getLogger(__name__)

AE_DATA_DIR = config.RESULTS_DIR / "autoencoder" / "encoded_features"

ASSETS = ["Bitcoin", "Ethereum"]
HORIZONS = [1, 7, 14]
FEATURE_SET_KEYS = ["primary", "candidate", "ablation", "combined"]
LATENT_FEATURES = [f"encoded_{i}" for i in range(8)]

MIN_OBSERVATIONS_FOR_CORRELATION = 10  # below this, a correlation coefficient is not meaningful


def _load_aligned_test_frame(asset: str, horizon: int, feature_set_key: str) -> pd.DataFrame:
    """Loads and Date-aligns one (asset, horizon, feature_set_key)
    configuration's encoded test rows with the matching original
    engineered-feature test rows. Returns one merged dataframe: Date,
    encoded_0..encoded_7, and every original predictor column (target
    and price-level columns excluded). Raises FileNotFoundError with the
    exact missing path if either source is absent -- never silently
    skipped.
    """
    price_col = config.ASSET_VARIABLES[asset]["price"]
    suffix = _horizon_suffix(horizon)
    prefix = f"{asset.lower()}_{feature_set_key}{suffix}"

    encoded_path = AE_DATA_DIR / f"{prefix}_encoded_test.csv"
    if not encoded_path.exists():
        raise FileNotFoundError(f"{encoded_path} not found.")
    encoded_df = pd.read_csv(encoded_path, parse_dates=["Date"])

    original_filename = FEATURE_FILE_TEMPLATES[feature_set_key].format(a=asset.lower(), s=suffix)
    original_path = config.FEATURES_DIR / original_filename
    if not original_path.exists():
        raise FileNotFoundError(f"{original_path} not found.")
    original_full = pd.read_csv(original_path, parse_dates=["Date"])

    # The original file holds train+val+test together; restrict to the
    # same test rows via the real Date column shared with the encoded
    # file, not by re-deriving a split independently (two independent
    # 70/15/15 computations could silently drift apart; an explicit
    # Date-based inner join cannot).
    target_col = f"{price_col}_log_return_target_t_plus_{horizon}"
    level_col = f"{price_col}_level"
    drop_cols = {target_col, level_col}
    original_predictor_cols = [c for c in original_full.columns if c not in drop_cols and c != "Date"]

    merged = encoded_df[["Date"] + LATENT_FEATURES].merge(
        original_full[["Date"] + original_predictor_cols], on="Date", how="inner",
    )
    if merged.empty:
        raise ValueError(
            f"No overlapping Date values between {encoded_path} and {original_path} -- "
            "cannot align latent and original features for this configuration."
        )
    return merged, original_predictor_cols


def compute_latent_original_correlations(
    assets: list = ASSETS, horizons: list = HORIZONS, feature_set_keys: list = FEATURE_SET_KEYS,
) -> pd.DataFrame:
    """TABLE: long-format Pearson + Spearman correlation between every
    latent dimension and every original engineered feature, for every
    (asset, horizon, feature_set_key) combination, computed on the
    Date-aligned test partition only. Both Pearson (linear) and Spearman
    (monotonic, rank-based) are reported side by side since neither
    alone fully characterises a relationship an autoencoder may have
    learned; direction (sign) is retained in both, values are not
    clipped or rounded.
    """
    try:
        rows = []
        for asset, horizon, feature_set_key in product(assets, horizons, feature_set_keys):
            try:
                merged, original_cols = _load_aligned_test_frame(asset, horizon, feature_set_key)
            except FileNotFoundError as exc:
                logger.warning("Skipping %s/%s/%s -- %s", asset, horizon, feature_set_key, exc)
                continue

            n_obs = len(merged)
            for latent_feature in LATENT_FEATURES:
                latent_series = merged[latent_feature]
                for original_feature in original_cols:
                    original_series = merged[original_feature]

                    pearson_r, spearman_rho = np.nan, np.nan
                    if n_obs >= MIN_OBSERVATIONS_FOR_CORRELATION:
                        # A zero-variance column (constant over the test window)
                        # makes both correlation coefficients undefined, not
                        # zero -- left as NaN rather than silently coerced.
                        if latent_series.std() > 0 and original_series.std() > 0:
                            pearson_r, _ = pearsonr(latent_series, original_series)
                            spearman_rho, _ = spearmanr(latent_series, original_series)

                    rows.append({
                        "asset": asset, "horizon": horizon, "feature_set_key": feature_set_key,
                        "latent_feature": latent_feature, "original_feature": original_feature,
                        "pearson_correlation": pearson_r, "spearman_correlation": spearman_rho,
                        "absolute_pearson_correlation": abs(pearson_r) if pd.notna(pearson_r) else np.nan,
                        "absolute_spearman_correlation": abs(spearman_rho) if pd.notna(spearman_rho) else np.nan,
                        "number_of_aligned_observations": n_obs,
                        "split": "test",
                        "representation_pair": "Autoencoder-Encoded latent vs Original engineered feature",
                    })

        if not rows:
            raise RuntimeError(
                "No latent-original correlations were computed -- check that "
                "results/autoencoder/encoded_features/*_encoded_test.csv and "
                "data/processed/features/*.csv both exist for at least one configuration."
            )
        return pd.DataFrame(rows)
    except CustomException:
        raise
    except Exception as e:
        raise CustomException(e, sys)


def summarize_top_associations(long_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """TABLE: for every latent dimension within every (asset, horizon,
    feature_set_key), the top-N original features by absolute Pearson
    correlation (ties broken by absolute Spearman). This reports the
    strongest descriptive associations only -- it is not a claim that a
    latent dimension "represents" any single one of them; an autoencoder
    bottleneck is a compressed mixture, and a latent dimension routinely
    shows moderate correlation with several original features at once,
    not a clean one-to-one match. Rows with an undefined correlation
    (constant column or too few aligned observations) are excluded from
    ranking, not treated as zero.
    """
    try:
        ranked = long_df.dropna(subset=["absolute_pearson_correlation"]).copy()
        ranked["rank"] = (
            ranked.groupby(["asset", "horizon", "feature_set_key", "latent_feature"])[
                "absolute_pearson_correlation"
            ].rank(ascending=False, method="first").astype(int)
        )
        top = ranked[ranked["rank"] <= top_n].sort_values(
            ["asset", "horizon", "feature_set_key", "latent_feature", "rank"]
        ).reset_index(drop=True)
        return top
    except Exception as e:
        raise CustomException(e, sys)
