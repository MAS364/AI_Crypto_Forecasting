"""Report and download generation -- builds every download artefact from
data already returned by existing API endpoints. Never calls anything
that retrains or recomputes a model; this module only formats and
packages results that already exist.
"""

import io
import json
import subprocess
import zipfile
from datetime import datetime, timezone

import pandas as pd

APP_VERSION = "1.0.0"

# Mirrors src/models/pipeline.py's DISPLAY_NAMES exactly -- kept as a
# local copy rather than importing from src/ (the frontend never imports
# src/ directly, only talks to the API over HTTP). Any place that shows
# result["feature_set"] to a human reader must go through this, not show
# the raw internal key ("combined") -- that was a real bug found while
# auditing UI/implementation alignment: every download was showing
# "Combined" instead of "Market + Google Trends + Fear & Greed Index".
FEATURE_SET_DISPLAY_NAMES = {
    "primary": "Market Only",
    "candidate": "Market + Google Trends",
    "ablation": "Market + Fear & Greed Index",
    "combined": "Market + Google Trends + Fear & Greed Index",
}


def display_feature_set(feature_set_key: str) -> str:
    """Human-readable feature configuration name -- use this everywhere
    a person reads it (reports, UI). The raw key stays as-is in
    metadata.json, where it's read by code, not people."""
    return FEATURE_SET_DISPLAY_NAMES.get(feature_set_key, feature_set_key.title())


def get_git_commit() -> str:
    """Best-effort -- returns None if not in a git repo (e.g. inside a
    Docker image without .git/ copied in), never raises."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=3,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def build_forecast_csv(result: dict, preview_df: pd.DataFrame = None) -> str:
    """forecast_results.csv -- Date, Actual Price (if available),
    Predicted Price, Predicted Return %, Predicted Direction, Forecast
    Horizon, Model, Asset, Representation, Feature Configuration."""
    direction = "Up" if result["predicted_return"] > 0 else "Down"
    actual_price = None
    if preview_df is not None:
        price_col = f"{result['asset']}_Price"
        if price_col in preview_df.columns:
            actual_price = preview_df[price_col].iloc[-1]

    row = {
        "Date": result.get("date_range", ["", ""])[1],
        "Actual Price": actual_price,
        "Predicted Price": result["predicted_price"],
        "Predicted Return %": result["predicted_return"] * 100,
        "Predicted Direction": direction,
        "Forecast Horizon (days)": result["horizon"],
        "Model Used": result["model"],
        "Asset": result["asset"],
        "Representation": result["representation"],
        "Feature Configuration": display_feature_set(result["feature_set"]),
    }
    return pd.DataFrame([row]).to_csv(index=False)


def _report_css() -> str:
    return """
    <style>
    body { font-family: -apple-system, "Segoe UI", sans-serif; max-width: 760px; margin: 40px auto; color: #1a1a1a; line-height: 1.6; padding: 0 20px; }
    h1 { font-size: 22px; font-weight: 600; border-bottom: 2px solid #1D9E75; padding-bottom: 12px; }
    h2 { font-size: 16px; font-weight: 600; margin-top: 32px; color: #1D9E75; }
    .kpi-grid { display: flex; gap: 16px; margin: 16px 0; flex-wrap: wrap; }
    .kpi { background: #f7f8f9; border-radius: 8px; padding: 12px 16px; min-width: 140px; }
    .kpi-label { font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.03em; }
    .kpi-value { font-size: 20px; font-weight: 600; margin-top: 4px; }
    table { width: 100%; border-collapse: collapse; margin: 12px 0; }
    th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #eee; font-size: 13px; }
    th { color: #666; font-weight: 500; }
    .limitations { background: #fff8e6; border-left: 3px solid #eab308; padding: 12px 16px; font-size: 13px; }
    .footer { margin-top: 40px; font-size: 11px; color: #999; border-top: 1px solid #eee; padding-top: 12px; }
    </style>
    """


def build_executive_report_html(result: dict, preview_df: pd.DataFrame = None, decision_row: dict = None) -> str:
    """executive_forecast_report.html -- for a supervisor, examiner, or
    non-technical reader. Real, styled HTML, not plain text."""
    direction = "increase" if result["predicted_return"] > 0 else "decrease"
    pct = abs(result["predicted_return"]) * 100
    actual_price = None
    if preview_df is not None:
        price_col = f"{result['asset']}_Price"
        if price_col in preview_df.columns:
            actual_price = preview_df[price_col].iloc[-1]

    hit_rate_html = ""
    if decision_row:
        hit_rate_html = f"""
        <div class="kpi"><div class="kpi-label">Historical hit rate</div><div class="kpi-value">{decision_row['Hit Rate (%)']:.1f}%</div></div>
        """

    return f"""<html><head><meta charset="utf-8">{_report_css()}</head><body>
<h1>AI for Cryptocurrency Time Series Analysis -- Executive Forecast Report</h1>
<p>{result['asset']} is forecast to <strong>{direction}</strong> by approximately <strong>{pct:.2f}%</strong> over the next {result['horizon']} day(s).</p>

<h2>Forecast summary</h2>
<div class="kpi-grid">
  {f'<div class="kpi"><div class="kpi-label">Current price</div><div class="kpi-value">${actual_price:,.2f}</div></div>' if actual_price else ''}
  <div class="kpi"><div class="kpi-label">Predicted price</div><div class="kpi-value">${result['predicted_price']:,.2f}</div></div>
  <div class="kpi"><div class="kpi-label">Direction</div><div class="kpi-value">{direction.title()}</div></div>
  {hit_rate_html}
</div>

<h2>Model used</h2>
<table>
<tr><th>Model</th><td>{result['model']}</td></tr>
<tr><th>Representation</th><td>{result['representation']}</td></tr>
<tr><th>Feature configuration</th><td>{display_feature_set(result['feature_set'])}</td></tr>
<tr><th>Forecast horizon</th><td>{result['horizon']} day(s)</td></tr>
</table>

<h2>Plain English interpretation</h2>
<p>The {result['model']} model, trained on the {display_feature_set(result['feature_set'])} feature configuration, predicts {result['asset']} will {direction} by approximately {pct:.2f}% over the next {result['horizon']} day(s). This is based on real historical data ending {result.get('date_range', ['',''])[1]}.</p>

<h2>Research evidence summary</h2>
<p>Correlation and Granger causality analysis found only weak statistical relationships between price and external variables (Google Trends, Fear &amp; Greed Index) -- consistent with this project's broader finding that external signals provide limited, inconsistent forecasting improvement.</p>

<h2>Limitations</h2>
<div class="limitations">
This forecast reflects patterns in historical data and is not investment advice. No calibrated confidence interval is available for this specific prediction. Cryptocurrency markets are highly volatile; past patterns may not hold during regime changes.
</div>

<div class="footer">Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} -- AI Crypto Forecasting Platform v{APP_VERSION}</div>
</body></html>"""


def build_technical_report_html(result: dict, metrics_row: dict = None, feature_importance: list = None) -> str:
    """technical_forecast_report.html -- for recruiters, ML engineers,
    data scientists. Inference configuration, metrics, feature
    importance -- real numbers, no filler."""
    metrics_html = ""
    if metrics_row:
        metrics_html = f"""
        <h2>Performance metrics (test set)</h2>
        <table>
        <tr><th>RMSE (returns)</th><td>{metrics_row.get('RMSE_returns', 'N/A')}</td></tr>
        <tr><th>MAE (returns)</th><td>{metrics_row.get('MAE_returns', 'N/A')}</td></tr>
        <tr><th>R²</th><td>{metrics_row.get('R2_returns', 'N/A')}</td></tr>
        <tr><th>Hit rate</th><td>{metrics_row.get('Hit_Rate_%', 'N/A')}%</td></tr>
        </table>
        """

    fi_html = ""
    if feature_importance:
        rows = "".join(f"<tr><td>{f['feature']}</td><td>{f.get('mean_importance', 'N/A'):.6f}</td></tr>" for f in feature_importance[:10])
        fi_html = f"""
        <h2>Feature importance (permutation, top 10)</h2>
        <table><tr><th>Feature</th><th>Mean Importance (Permuted RMSE - Baseline RMSE)</th></tr>{rows}</table>
        """

    return f"""<html><head><meta charset="utf-8">{_report_css()}</head><body>
<h1>Technical Forecast Report</h1>

<h2>Inference configuration</h2>
<table>
<tr><th>Asset</th><td>{result['asset']}</td></tr>
<tr><th>Model</th><td>{result['model']}</td></tr>
<tr><th>Feature set</th><td>{display_feature_set(result['feature_set'])}</td></tr>
<tr><th>Representation</th><td>{result['representation']}</td></tr>
<tr><th>Horizon</th><td>{result['horizon']} day(s)</td></tr>
<tr><th>Predicted return</th><td>{result['predicted_return']:.6f}</td></tr>
<tr><th>Predicted price</th><td>${result['predicted_price']:,.4f}</td></tr>
</table>

{metrics_html}
{fi_html}

<div class="footer">Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} -- AI Crypto Forecasting Platform v{APP_VERSION}</div>
</body></html>"""


def build_metadata_json(result: dict, inference_time_seconds: float = None) -> str:
    """metadata.json -- timestamp, config, app version, best-effort git
    commit. inference_time_seconds should be measured client-side around
    the actual API call, since that's what a user experiences -- not
    invented here."""
    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "asset": result["asset"],
        "forecast_horizon_days": result["horizon"],
        "model": result["model"],
        "representation": result["representation"],
        "feature_configuration": result["feature_set"],
        "application_version": APP_VERSION,
        "inference_time_seconds": inference_time_seconds,
        "git_commit": get_git_commit(),
    }
    return json.dumps(metadata, indent=2)


def build_forecast_package_zip(
    result: dict, preview_df: pd.DataFrame = None, decision_row: dict = None,
    metrics_row: dict = None, feature_importance: list = None,
    model_comparison_csv: str = None, research_evidence_csv: str = None,
    inference_time_seconds: float = None,
) -> bytes:
    """forecast_package.zip -- bundles every artefact in one download.
    Any component that isn't available (e.g. feature importance not yet generated) is
    simply omitted, not faked."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("forecast_results.csv", build_forecast_csv(result, preview_df))
        zf.writestr("executive_forecast_report.html", build_executive_report_html(result, preview_df, decision_row))
        zf.writestr("technical_forecast_report.html", build_technical_report_html(result, metrics_row, feature_importance))
        zf.writestr("metadata.json", build_metadata_json(result, inference_time_seconds))
        if feature_importance:
            fi_df = pd.DataFrame(feature_importance)
            zf.writestr("feature_importance.csv", fi_df.to_csv(index=False))
        if model_comparison_csv:
            zf.writestr("model_comparison.csv", model_comparison_csv)
        if research_evidence_csv:
            zf.writestr("research_evidence.csv", research_evidence_csv)
    buf.seek(0)
    return buf.read()
