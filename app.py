# app_table_only.py  —  Groundwater forecasts (ARIMA, tables only)

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="Groundwater Forecasts (Table Only)", layout="wide")
st.title("Groundwater Forecasting — Table View")

DATA_PATH = "GW data (missing filled).csv"
HORIZON_M = 60      # 5-year horizon (months)

# ---------- helpers ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_raw(path: str) -> pd.DataFrame | None:
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" +
                                df["Months"].astype(str).str.zfill(2) + "-01")
    return df.sort_values("Date").reset_index(drop=True)

def clean_series(df: pd.DataFrame, well: str) -> pd.DataFrame:
    s = df[well].copy()
    q1, q3 = s.quantile([0.25, 0.75]); iqr = q3 - q1
    s = s.where(s.between(q1 - 3*iqr, q3 + 3*iqr)).interpolate(limit_direction="both")
    return pd.DataFrame({"Date": df["Date"], well: s}).dropna().reset_index(drop=True)

@st.cache_data(show_spinner=True)
def train_arima(series: pd.Series, seasonal: bool, horizon: int) -> tuple[dict, pd.DataFrame]:
    split = int(len(series)*0.8)
    train, test = series.iloc[:split], series.iloc[split:]
    order, s_order = (1,1,1), (1,1,1,12) if seasonal else (0,0,0,0)

    model = ARIMA(train, order=order, seasonal_order=s_order).fit()
    rmse = round(np.sqrt(mean_squared_error(test, model.forecast(len(test)))), 4)

    model_full = ARIMA(series, order=order, seasonal_order=s_order).fit()
    forecast = model_full.get_forecast(horizon)
    future = pd.DataFrame({
        "Date": pd.date_range(series.index[-1] + pd.DateOffset(months=1),
                              periods=horizon, freq="MS"),
        "Depth": forecast.predicted_mean.values.round(2)
    })
    metrics = {"AIC": round(model_full.aic,1),
               "BIC": round(model_full.bic,1),
               "RMSE test": rmse}
    return metrics, future

# ---------- session storage -------------------------------------------------------
if "summary_rows" not in st.session_state:
    st.session_state["summary_rows"] = []

# ---------- UI --------------------------------------------------------------------
raw = load_raw(DATA_PATH)
if raw is None:
    st.error("CSV not found. Upload a groundwater file.")
    if up := st.sidebar.file_uploader("Upload CSV", type="csv"):
        Path(DATA_PATH).write_bytes(up.read()); st.experimental_rerun()
    st.stop()

wells = [c for c in raw.columns if c.startswith("W")]
well = st.sidebar.selectbox("Well", wells)
seasonal = st.sidebar.checkbox("Include 12-month seasonality", True)

clean = clean_series(raw, well)
if len(clean) < 30:
    st.warning("Not enough data points (<30) to fit ARIMA reliably."); st.stop()

series = pd.Series(clean[well].values, index=clean["Date"])

with st.spinner("Fitting ARIMA…"):
    metrics, future = train_arima(series, seasonal, HORIZON_M)

# ---------- Show results ----------------------------------------------------------
st.subheader("ARIMA metrics")
st.table(pd.DataFrame(metrics, index=["Value"]))

st.subheader("5-year monthly forecast")
st.dataframe(future, use_container_width=True)

# ---------- Save summary ----------------------------------------------------------
if st.button("💾 Save this forecast row"):
    row = {"Well": well}
    yearly = future.assign(Y=future["Date"].dt.year).groupby("Y").first()["Depth"]
    for yr in range(2025, 2030):
        row[str(yr)] = yearly.get(yr, np.nan)
    row.update(metrics)
    st.session_state["summary_rows"].append(pd.DataFrame([row]))
    st.success(f"Saved – total rows: {len(st.session_state['summary_rows'])}")

# ---------- Sidebar CSV download --------------------------------------------------
n_rows = len(st.session_state["summary_rows"])
st.sidebar.markdown(f"**Saved summaries:** {n_rows}")
if n_rows:
    combined = pd.concat(st.session_state["summary_rows"]).reset_index(drop=True)
    st.sidebar.download_button("⬇️ Download summaries CSV",
                               combined.to_csv(index=False).encode(),
                               file_name=f"well_summaries_{datetime.today().date()}.csv",
                               mime="text/csv")
    if st.sidebar.checkbox("Show combined table"):
        st.subheader("Combined saved summaries")
        st.dataframe(combined, use_container_width=True)
