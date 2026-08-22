"""Combines the Original-branch permutation importance results (in
results/feature_importance/processed/, produced by
run_feature_importance_grid.py) with the Autoencoder-Encoded-branch
results (in results/feature_importance_autoencoder/processed/, produced
by run_feature_importance_grid_autoencoder.py) into:

  1. One full feature dictionary -- every engineered feature across all
     4 configurations, PLUS the 8 autoencoder latent dimensions, clearly
     labelled by representation.
  2. A plain long-format union of both branches' raw PFI results, kept
     for anyone who wants to filter/pivot the raw numbers themselves --
     explicitly labelled as a union, NOT a cross-representation-
     comparable ranking (see point 3).
  3. A DEFENSIBLE Original-vs-Autoencoder-Encoded comparison (test RMSE,
     within-configuration normalised importance, importance
     concentration) via src.feature_importance.representation_comparison
     -- see that module's docstring for why raw PFI magnitudes across
     representations are not directly comparable and what this uses
     instead.
  4. A written diagnosis of why the "Feature importance comparison"
     Streamlit tab was reported broken, and what actually fixes it,
     independently verified (not just asserted) against this
     repository's real code and a live local run in Phase 1 -- see
     FEATURE_IMPORTANCE_TAB_DIAGNOSIS.md.

This supersedes an earlier version of this script (Phase 1 verification
finding): the earlier version's "combined ranking" was a plain
concatenation of the two branches with no reconciliation -- item 3 above
is the actual fix. That earlier file's name is still produced here for
continuity, but is now clearly documented as a union rather than
implied to be a ranking.

HOW TO RUN
----------
1. Run run_feature_importance_grid.py and run_feature_importance_grid_autoencoder.py first.
2. From the project root:
       python run_combined_feature_importance_report.py

OUTPUT
------
    results/feature_importance_autoencoder/processed/full_feature_dictionary.csv
    results/feature_importance_autoencoder/processed/all_representations_long.csv
    results/feature_importance_autoencoder/processed/representation_comparison_summary.csv
    FEATURE_IMPORTANCE_TAB_DIAGNOSIS.md
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.feature_importance.grouping import build_feature_dictionary
from src.feature_importance.representation_comparison import build_representation_comparison_summary

ORIGINAL_PROCESSED = config.RESULTS_DIR / "feature_importance" / "processed"
AE_PROCESSED = config.RESULTS_DIR / "feature_importance_autoencoder" / "processed"
AE_RAW = config.RESULTS_DIR / "feature_importance_autoencoder" / "raw"

OUT_PROCESSED = AE_PROCESSED
DIAGNOSIS_PATH = PROJECT_ROOT / "FEATURE_IMPORTANCE_TAB_DIAGNOSIS.md"

DIAGNOSIS_TEXT = """# Feature Importance Comparison Tab -- Diagnosis

Independently verified against the real project code, real files on disk,
and a live local run (Phase 1 verification of this analysis) -- not
guessed and not taken on faith from an earlier draft of this diagnosis:

- `deployment/app.py`'s `/feature-importance-comparison`,
  `/feature-importance-agreement-summary`, and `/feature-group-importance`
  endpoints do `pd.read_csv(...)` on `results/feature_importance/processed/*.csv`
  and return the records -- no bug in that code.
- Hitting all three endpoints locally via FastAPI's TestClient returned
  HTTP 200 with the expected row counts and real data (not just imports
  cleanly -- actually exercised).
- Every column the Streamlit page
  (`deployment/frontend/pages/6_Feature_Importance_Comparison.py`) expects
  is present in the CSVs on disk, with correct dtypes.
- `src/feature_importance/aggregate.py`'s column-producing logic matches
  what's actually saved in those CSVs -- no drift between code and data.

**There is no code or data bug in the local project for the Original
branch of this tab.** As of this update, the tab ALSO has real
endpoints, API-client functions, and Streamlit sections for the
Autoencoder-Encoded results and the latent-original correlation summary
(see the "What changed in this update" section below) -- so the same
"no local code/data bug" conclusion now extends to those too, verified
the same way (TestClient, HTTP 200, real data).

## Most likely real cause of a failure on the DEPLOYED (Cloud Run) tab

1. **Wrong/missing `CRYPTO_API_URL` on the deployed frontend service.**
   `deployment/frontend/api_client.py` reads the backend URL from the
   `CRYPTO_API_URL` environment variable, defaulting to
   `http://127.0.0.1:8000` if unset. `docker-compose.yml` only sets this
   correctly for local `docker compose up` (`http://api:8080`, a
   compose-internal hostname that does **not** exist on Cloud Run).
   **Fix:** in the GCP Cloud Run console (or your deploy script), set the
   frontend service's `CRYPTO_API_URL` env var to the backend service's
   actual public Cloud Run URL, then redeploy/restart the frontend service.

2. **Stale backend container image.** `deployment/Dockerfile` explicitly
   copies `results/feature_importance/` (and, as of this update,
   `results/feature_importance_autoencoder/`) into the backend image at
   build time. If the currently-running Cloud Run backend revision was
   built before these CSVs existed, that revision's image genuinely does
   not contain them.
   **Fix:** rebuild the backend Docker image from the current repo state
   and deploy the new image/revision to Cloud Run.

## How to tell which one it is

Open the tab and read the actual error text:
- "Can't reach the API at ..." -> issue #1 (env var / networking)
- "... not found -- run ... first" -> issue #2 (stale image)

## What changed in this update

- New endpoints: `/feature-importance-comparison-autoencoder`,
  `/feature-importance-agreement-summary-autoencoder`,
  `/feature-group-importance-autoencoder`, `/latent-original-correlations`,
  `/latent-original-top-associations`, `/representation-comparison-summary`.
- New Streamlit sections in
  `deployment/frontend/pages/6_Feature_Importance_Comparison.py`: an
  Original / Autoencoder-Encoded representation selector, a latent-original
  correlation explorer, and a representation-comparison summary table.
- `deployment/Dockerfile` now also copies
  `results/feature_importance_autoencoder/` into the backend image.

None of this required changing the Original-branch code that was already
confirmed working -- it was purely additive.
"""


def require(path: Path, hint: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. {hint}")
    return pd.read_csv(path)


def main() -> None:
    print(f"Project root: {PROJECT_ROOT}\n")

    # --------------------------------------------------------------
    # 1. FULL FEATURE DICTIONARY
    # --------------------------------------------------------------
    original_long = require(
        ORIGINAL_PROCESSED / "feature_dictionary.csv",
        "Run run_feature_importance_grid.py first (produces the Original-branch results).",
    )
    original_features = sorted(original_long["feature"].dropna().unique())
    print(f"Original engineered features found: {len(original_features)}")

    encoded_dims = [f"encoded_{i}" for i in range(8)]
    encoded_dict = pd.DataFrame([
        {
            "feature": f, "feature_group": "Other (Autoencoder Latent Dimension)",
            "source": "Autoencoder (learned, compressed representation)",
            "engineering_method": "8-dimensional autoencoder bottleneck -- no single real-world "
                                   "source variable; see results/feature_importance_autoencoder/"
                                   "processed/latent_original_correlations_long.csv for its "
                                   "descriptive correlation with the real original features.",
        }
        for f in encoded_dims
    ])

    original_dict = build_feature_dictionary(original_features)
    original_dict.insert(0, "representation", "Original")
    encoded_dict.insert(0, "representation", "Autoencoder-Encoded")

    full_dict = pd.concat([original_dict, encoded_dict], ignore_index=True)
    full_dict_path = OUT_PROCESSED / "full_feature_dictionary.csv"
    OUT_PROCESSED.mkdir(parents=True, exist_ok=True)
    full_dict.to_csv(full_dict_path, index=False)
    print(f"Saved full feature dictionary ({len(full_dict)} rows: "
          f"{len(original_dict)} Original + {len(encoded_dict)} Autoencoder-Encoded) -> {full_dict_path}")

    # --------------------------------------------------------------
    # 2. PLAIN LONG-FORMAT UNION (explicitly NOT a cross-representation
    #    ranking -- see module docstring and point 3 below for the
    #    actual defensible comparison)
    # --------------------------------------------------------------
    original_long_pfi = require(
        config.RESULTS_DIR / "feature_importance" / "raw" / "permutation_importance_long.csv",
        "Run run_feature_importance_grid.py first.",
    )
    ae_long_pfi = require(
        AE_RAW / "permutation_importance_autoencoder_long.csv",
        "Run run_feature_importance_grid_autoencoder.py first.",
    )

    all_representations_long = pd.concat([original_long_pfi, ae_long_pfi], ignore_index=True)
    union_path = OUT_PROCESSED / "all_representations_long.csv"
    all_representations_long.sort_values(
        ["representation", "asset", "horizon", "feature_set_key", "model_name", "rank"]
    ).to_csv(union_path, index=False)
    print(f"Saved raw long-format union ({len(all_representations_long)} rows, "
          f"representations={sorted(all_representations_long['representation'].unique())}) -> {union_path}\n"
          "  NOTE: this is a plain union for filtering/pivoting the raw numbers yourself. "
          "'rank' is only meaningful WITHIN its own representation/model/config -- see "
          "representation_comparison_summary.csv for the actual cross-representation comparison.")

    # --------------------------------------------------------------
    # 3. DEFENSIBLE CROSS-REPRESENTATION COMPARISON
    # --------------------------------------------------------------
    comparison = build_representation_comparison_summary(original_long_pfi, ae_long_pfi)
    comparison_path = OUT_PROCESSED / "representation_comparison_summary.csv"
    comparison.to_csv(comparison_path, index=False)
    n_encoded_better = (comparison["better_test_rmse_representation"] == "Autoencoder-Encoded").sum()
    n_original_better = (comparison["better_test_rmse_representation"] == "Original").sum()
    print(f"\nSaved representation comparison summary ({len(comparison)} configurations) -> {comparison_path}")
    print(f"  Original had the lower test RMSE in {n_original_better}/{len(comparison)} configurations")
    print(f"  Autoencoder-Encoded had the lower test RMSE in {n_encoded_better}/{len(comparison)} configurations")

    # --------------------------------------------------------------
    # 4. TAB DIAGNOSIS
    # --------------------------------------------------------------
    DIAGNOSIS_PATH.write_text(DIAGNOSIS_TEXT)
    print(f"\nSaved tab diagnosis -> {DIAGNOSIS_PATH}")

    print("\nDone.")


if __name__ == "__main__":
    main()
