"""Shared HTTP client for every Streamlit page -- all pages call the
FastAPI backend over HTTP (deployment/app.py), never import src/ directly.
This keeps the frontend genuinely decoupled from the backend, matching
what "HTTP-based" was asked for.
"""

import os
from typing import Optional

import requests
import streamlit as st

API_BASE_URL = os.environ.get("CRYPTO_API_URL", "http://127.0.0.1:8000")


def _get(path: str, params: dict = None) -> dict:
    try:
        r = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(
            f"Can't reach the API at {API_BASE_URL}. Start it with "
            "`uvicorn deployment.app:app --reload` before using this page."
        )
        st.stop()
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e.response.json().get('detail', str(e))}")
        st.stop()
    except requests.exceptions.RequestException as e:
        # Catches everything else (Timeout, ReadTimeout, etc.) that isn't
        # a ConnectionError or HTTPError. Without this, those exception
        # types fell straight through _get() uncaught, and got silently
        # swallowed by a bare `except Exception:` in calling pages --
        # producing a blank fallback value with no error shown anywhere,
        # rather than the intended error message.
        st.error(f"API request to {path} failed: {e}")
        st.stop()


def _get_optional(path: str, params: dict = None):
    """Like _get, but never shows a UI error or calls st.stop() -- for
    genuinely non-critical calls (e.g. a single KPI card on the Home
    page) where the caller already has a sensible fallback and the rest
    of the page should render fine either way. Returns None on any
    failure instead of relying on a caller's try/except to catch _get's
    st.stop() call -- Streamlit's internal StopException is a subclass
    of Exception, so a bare `except Exception:` around a call to _get()
    can silently catch it, which is fragile and also eats whatever
    st.error() message was meant to explain the failure."""
    try:
        r = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return None


def health_check() -> bool:
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


@st.cache_data(ttl=300)
def get_root_info() -> dict:
    return _get("/")


@st.cache_data(ttl=300)
def get_csv_requirements(asset: str) -> dict:
    return _get("/csv-requirements", params={"asset": asset})


@st.cache_data(ttl=60)
def get_decision_summary() -> list[dict]:
    return _get("/decision-summary")


@st.cache_data(ttl=60)
def get_decision_summary_optional() -> list[dict]:
    """Same data as get_decision_summary(), but degrades to an empty list
    instead of halting the page -- for Home.py's KPI cards, which have a
    sensible empty-state fallback and shouldn't stop the whole landing
    page over one summary table. Pages that genuinely can't function
    without this data (Model comparison, ARIMA) should keep using
    get_decision_summary() so they correctly halt with a clear error."""
    return _get_optional("/decision-summary") or []


@st.cache_data(ttl=60)
def get_research_evidence() -> dict:
    return _get("/research-evidence")


@st.cache_resource
def get_model_registry() -> dict:
    return _get("/models")


def predict(horizon: int, asset: str, feature_set: str, representation: str, model: str, rows: list[dict]) -> dict:
    """No caching -- every prediction request is a fresh call, matching
    the "never retrain, but always predict fresh on this input" contract."""
    r = requests.post(f"{API_BASE_URL}/predict", json={
        "horizon": horizon, "asset": asset, "feature_set": feature_set,
        "representation": representation, "model": model, "rows": rows,
    }, timeout=30)
    if r.status_code >= 400:
        raise RuntimeError(r.json().get("detail", r.text))
    return r.json()


def predict_csv(file_bytes: bytes, filename: str, asset: str, model: str, horizon: int, representation: str = "Original", feature_set: Optional[str] = None) -> dict:
    data = {"asset": asset, "model": model, "horizon": horizon, "representation": representation}
    if feature_set:
        data["feature_set"] = feature_set
    r = requests.post(
        f"{API_BASE_URL}/predict/csv",
        files={"file": (filename, file_bytes, "text/csv")},
        data=data,
        # 180s, not 60s -- the Autoencoder-Encoded branch's first call in
        # a freshly started API process pays a real one-time cost
        # importing TensorFlow and loading the .keras encoder, on top of
        # engineering features across the whole upload. Subsequent calls
        # in the same running process should be much faster regardless.
        timeout=180,
    )
    if r.status_code >= 400:
        detail = r.json().get("detail", r.text)
        if isinstance(detail, dict) and "validation_errors" in detail:
            raise ValidationErrors(detail["validation_errors"])
        raise RuntimeError(str(detail))
    return r.json()


class ValidationErrors(Exception):
    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


@st.cache_data(ttl=300)
def get_feature_importance_comparison() -> list[dict]:
    return _get("/feature-importance-comparison")


@st.cache_data(ttl=300)
def get_feature_importance_agreement_summary() -> list[dict]:
    return _get("/feature-importance-agreement-summary")


@st.cache_data(ttl=300)
def get_feature_group_importance() -> list[dict]:
    return _get("/feature-group-importance")


@st.cache_data(ttl=300)
def get_feature_importance_comparison_autoencoder() -> list[dict]:
    return _get("/feature-importance-comparison-autoencoder")


@st.cache_data(ttl=300)
def get_feature_importance_agreement_summary_autoencoder() -> list[dict]:
    return _get("/feature-importance-agreement-summary-autoencoder")


@st.cache_data(ttl=300)
def get_feature_group_importance_autoencoder() -> list[dict]:
    return _get("/feature-group-importance-autoencoder")


@st.cache_data(ttl=300)
def get_latent_original_correlations(asset: str = None, horizon: int = None, feature_set: str = None) -> list[dict]:
    params = {k: v for k, v in {"asset": asset, "horizon": horizon, "feature_set": feature_set}.items() if v}
    return _get("/latent-original-correlations", params=params)


@st.cache_data(ttl=300)
def get_latent_original_top_associations(asset: str = None, horizon: int = None, feature_set: str = None) -> list[dict]:
    params = {k: v for k, v in {"asset": asset, "horizon": horizon, "feature_set": feature_set}.items() if v}
    return _get("/latent-original-top-associations", params=params)


@st.cache_data(ttl=300)
def get_representation_comparison_summary() -> list[dict]:
    return _get("/representation-comparison-summary")


@st.cache_data(ttl=300)
def get_arima_info(asset: str, horizon: int, feature_set: str = "primary") -> dict:
    return _get("/arima-info", params={"asset": asset, "horizon": horizon, "feature_set": feature_set})


@st.cache_data(ttl=300)
def get_naive_info(asset: str, horizon: int, feature_set: str = "primary") -> dict:
    return _get("/naive-info", params={"asset": asset, "horizon": horizon, "feature_set": feature_set})


@st.cache_data(ttl=300)
def get_last_training_date() -> Optional[str]:
    data = _get_optional("/last-training-date")
    return data["last_training_date"] if data else None


@st.cache_data(ttl=300)
def get_feature_importance_for_model(asset: str, horizon: int, model: str, feature_set: str = "primary") -> list[dict]:
    return _get("/feature-importance-for-model", params={
        "asset": asset, "horizon": horizon, "model": model, "feature_set": feature_set,
    })


@st.cache_data(ttl=300)
def get_feature_importance_for_model_optional(asset: str, horizon: int, model: str, feature_set: str = "primary"):
    """Same data as get_feature_importance_for_model(), but returns None
    instead of halting the page on failure -- for Forecast's best-effort
    report-building. Also returns None (not an empty list) if the
    combination genuinely has no permutation importance data (e.g. the
    deployed model uses the Autoencoder-Encoded representation, which
    the PFI grid deliberately does not cover), so callers can distinguish
    "not available" from "available but empty" the same way."""
    data = _get_optional("/feature-importance-for-model", params={
        "asset": asset, "horizon": horizon, "model": model, "feature_set": feature_set,
    })
    return data if data else None


@st.cache_data(ttl=60)
def get_full_metrics() -> list[dict]:
    return _get("/full-metrics")


@st.cache_data(ttl=60)
def get_full_metrics_optional() -> list[dict]:
    """Same data as get_full_metrics(), but degrades to an empty list
    instead of halting the page -- for Forecast's best-effort report
    context. See get_feature_importance_for_model_optional()'s docstring for
    why this is necessary rather than relying on try/except alone."""
    return _get_optional("/full-metrics") or []


@st.cache_data(ttl=60)
def get_predictions(asset: str, horizon: int, model: str, feature_set: str = "primary", representation: str = "Original", split: str = "test") -> list[dict]:
    return _get("/predictions", params={
        "asset": asset, "horizon": horizon, "model": model,
        "feature_set": feature_set, "representation": representation, "split": split,
    })
