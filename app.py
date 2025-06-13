import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os

st.set_page_config(page_title="Groundwater Fast Forecast", layout="wide")
st.title("Groundwater Forecasting — Fast Table View")

DATA_PATH, HORIZON_M = "GW data (missing filled).csv", 60
SUMMARY_CSV = "yearly_summaries.csv"

# ---------- helpers ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_raw(path: str):
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" +
                                df["Months"].astype(str).str.zfill(2) + "-01")
    return df.sort_values("Date").reset_index(drop=True)

def clean_series(df, well):
    s = df[well].copy()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    s = s.where(s.between(q1 - 3*iqr, q3 + 3*iqr)).interpolate(limit_direction="both")
    return pd.Series(s.values, index=df["Date"])

def seasonal_naive_forecast(series, horizon):
    """Predict y[t+h] = y[t+h-12]. No fitting, so metrics are on a simple 1-step hold-out."""
    y_pred = series.shift(12).dropna()
    test = series.loc[y_pred.index]
    rmse = round(np.sqrt(mean_squared_error(test, y_pred)), 4)
    future_idx = pd.date_range(series.index[-1] + pd.DateOffset(months=1),
                               periods=horizon, freq="MS")
    future_vals = series.iloc[-12:]  # last seasonal cycle
    future = pd.Series(np.tile(future_vals.values, int(np.ceil(horizon/12)))[:horizon],
                       index=future_idx)
    return {"RMSE test": rmse}, future.round(2)

def holt_winters_forecast(series, horizon):
    mdl = ExponentialSmoothing(series,
                               trend="add",
                               seasonal="add",
                               seasonal_periods=12,
                               initialization_method="estimated").fit()
    rmse = round(np.sqrt(mean_squared_error(series, mdl.fittedvalues)), 4)
    future_idx = pd.date_range(series.index[-1] + pd.DateOffset(months=1),
                               periods=horizon, freq="MS")
    future = mdl.forecast(horizon)
    future.index = future_idx
    return {"AIC": round(mdl.aic,1), "RMSE train": rmse}, future.round(2)

# ---------- UI --------------------------------------------------------------------
raw = load_raw(DATA_PATH)
if raw is None:
    st.error("CSV not found. Upload file below.")
    if up := st.sidebar.file_uploader("Upload CSV", type="csv"):
        Path(DATA_PATH).write_bytes(up.read())
        st.experimental_rerun()
    st.stop()

wells = [c for c in raw.columns if c.startswith("W")]
well  = st.sidebar.selectbox("Well", wells)
model_choice = st.sidebar.radio("Model",
                                ["Seasonal Naïve (super-fast)", "Holt-Winters (fast)"])

series = clean_series(raw, well)
if len(series) < 24:
    st.warning("Need ≥24 points for seasonal forecasting.")
    st.stop()

with st.spinner("Calculating forecast…"):
    if model_choice.startswith("Seasonal"):
        metrics, future = seasonal_naive_forecast(series, HORIZON_M)
    else:
        metrics, future = holt_winters_forecast(series, HORIZON_M)

# ---------- Display ----------------------------------------------------------------
st.subheader("Model & accuracy")
st.table(pd.DataFrame(metrics, index=["Value"]))

st.subheader("5-year monthly forecast")
st.dataframe(future.to_frame("Depth"), use_container_width=True)

# ---------- Save summary row -------------------------------------------------------
if st.button("💾 Save yearly summary"):
    row = {"Well": well}
    # Calculate yearly averages from the forecast
    yearly_avg = future.resample("A").mean()
    for yr in range(2025, 2030):
        val = yearly_avg.get(str(yr))
        row[str(yr)] = val.values[0] if val is not None and len(val) > 0 else np.nan
    row.update(metrics)

    new_df = pd.DataFrame([row])

    # Save to CSV, append if exists
    if os.path.exists(SUMMARY_CSV):
        existing_df = pd.read_csv(SUMMARY_CSV)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(SUMMARY_CSV, index=False)
    else:
        new_df.to_csv(SUMMARY_CSV, index=False)

    # Also store in session state for immediate download option
    st.session_state.setdefault("summary_rows", []).append(new_df)
    st.success(f"Saved yearly summary to '{SUMMARY_CSV}' — total saved rows: {len(st.session_state['summary_rows'])}")

# ---------- Download summaries -----------------------------------------------------
n_rows = len(st.session_state.get("summary_rows", []))
st.sidebar.markdown(f"**Saved summaries in session:** {n_rows}")
if n_rows:
    combined = pd.concat(st.session_state["summary_rows"]).reset_index(drop=True)
    st.sidebar.download_button("⬇️ Download CSV from session",
                               combined.to_csv(index=False).encode(),
                               f"well_summaries_{datetime.today().date()}.csv",
                               "text/csv")

if os.path.exists(SUMMARY_CSV):
    with open(SUMMARY_CSV, "rb") as f:
        st.sidebar.download_button("⬇️ Download saved CSV file",
                                   f.read(),
                                   SUMMARY_CSV,
                                   "text/csv")
