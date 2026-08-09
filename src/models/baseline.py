"""The Naive (Persistence) baseline model.

Extracted from 07_modeling.ipynb. Predicts tomorrow's return as equal
to today's already-observed return -- no training, no parameters. Every
other model in this project must beat this to justify its complexity.

This corrects an earlier version of this notebook that used yesterday's
return (`{price_col}_log_return_lag_1`) instead of today's -- that
version silently discarded the most recent real observation, using a
two-day-old value to forecast tomorrow. `{price_col}_log_return`
(unlagged) is today's own, already-realised return, which is the
correct persistence definition: prediction(t+1) = observed return(t).
"""

import pandas as pd

from src.evaluation.metrics import evaluate


def run_naive(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, price_col: str, horizon: int = 1) -> dict:
    """At horizon=1, predicts using today's own already-realised return
    (see module docstring for why not yesterday's). At horizon > 1, the
    horizon-analogue of "yesterday predicts tomorrow" is "the last
    `horizon` days predict the next `horizon` days" -- uses the trailing
    cumulative return feature built alongside the target at that horizon
    (`{price_col}_trailing_{horizon}day_return`), which is already
    leakage-checked in feature engineering, not recomputed here.
    """
    level = f"{price_col}_level"
    target = f"{price_col}_log_return_target_t_plus_{horizon}"

    if horizon == 1:
        reference_col = f"{price_col}_log_return"
    else:
        reference_col = f"{price_col}_trailing_{horizon}day_return"

    preds, metrics = {}, {}
    for name, split in [("validation", val), ("test", test)]:
        p = split[reference_col]
        metrics[name] = evaluate(split[level], split[target], p)
        preds[name] = pd.DataFrame({"event_date": split["Date"], "actual": split[target], "predicted": p})

    return {"predictions": preds, "metrics": metrics}
