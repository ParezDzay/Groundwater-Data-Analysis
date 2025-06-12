# app.py — Groundwater forecasts (ARIMA only)

import streamlit as st, pandas as pd, numpy as np, plotly.express as px
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Groundwater Forecasts (ARIMA)", layout="wide")
st.title("Groundwater Forecasting — Depth View (ARIMA Only)")

DATA_PATH  = "GW data (missing filled).csv"
HORIZON_M  = 60  # 5-year horizon

@st.cache_data(show_spinner=False)
def load_raw(path):
    if not Path(path).exists(): return None
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Year"].astype(str)+"-"+df["Months"].astype(str)+"-01")
    return df.sort_values("Date").reset_index(drop=True)

def clean_series(df, well):
    s = df[well].copy()
    q1,q3 = s.quantile(0.25), s.quantile(0.75); iqr = q3 - q1
    s = s.where(s.between(q1-3*iqr, q3+3*iqr)).interpolate(limit_direction="both")
    out = pd.DataFrame({"Date": df["Date"], well: s})
    return out.dropna().reset_index(drop=True)

def clip_bounds(series):
    lo, hi = series.min(), series.max()
    rng = hi - lo if hi > lo else max(hi, 1)
    return max(0, lo - 0.2*rng), hi + 0.2*rng

def train_arima(series, seasonal, lo, hi):
    split = int(len(series)*0.8)
    train, test = series.iloc[:split], series.iloc[split:]
    res = ARIMA(train, order=(1,1,1),
                seasonal_order=(1,1,1,12) if seasonal else (0,0,0,0)).fit()
    rmse = round(np.sqrt(mean_squared_error(test, res.forecast(len(test)))), 4)
    res_full = ARIMA(series, order=(1,1,1),
                     seasonal_order=(1,1,1,12) if seasonal else (0,0,0,0)).fit()
    fc = res_full.get_forecast(HORIZON_M)
    future = pd.DataFrame({
        "Date": pd.date_range(series.index[-1]+pd.DateOffset(months=1), periods=HORIZON_M, freq="MS"),
        "Depth": np.clip(fc.predicted_mean.values, lo, hi)
    })
    metrics = {"AIC": round(res_full.aic, 1), "BIC": round(res_full.bic, 1), "RMSE test": rmse}
    return metrics, res_full, future

if "summary_rows" not in st.session_state:
    st.session_state["summary_rows"] = []

raw = load_raw(DATA_PATH)
if raw is None:
    st.error("CSV not found. Upload to continue.")
    if up := st.file_uploader("Upload CSV", type="csv"):
        Path(DATA_PATH).write_bytes(up.getvalue()); st.experimental_rerun()
    st.stop()

wells = [c for c in raw.columns if c.startswith("W")]
well = st.sidebar.selectbox("Well", wells)
seasonal = st.sidebar.checkbox("Include 12-month seasonality", True)

clean = clean_series(raw, well)
lo, hi = clip_bounds(clean[well])
series = pd.Series(clean[well].values, index=clean["Date"])

metrics, res, future = train_arima(series, seasonal, lo, hi)
st.subheader("📈 ARIMA Metrics")
st.json(metrics)

# ---- plot ----
df_act = pd.DataFrame({"Date": clean["Date"], "Depth": clean[well], "Type": "Actual"})
df_fit = pd.DataFrame({"Date": series.index, "Depth": res.fittedvalues.clip(lo, hi), "Type": "Predicted"})
df_for = future.assign(Type="Forecast")
plot_df = pd.concat([df_act, df_fit, df_for])
fig = px.line(plot_df, x="Date", y="Depth", color="Type",
              labels={"Depth": "Water-table depth (m)"},
              title=f"{well} — ARIMA Fit & 5-Year Forecast")
fig.update_yaxes(autorange="reversed")
for t in fig.data:
    if t.name == "Forecast": t.update(line=dict(dash="dash"))
fig.add_vline(x=df_act["Date"].max(), line_dash="dot", line_color="gray")
st.plotly_chart(fig, use_container_width=True)

# ---- table ----
st.subheader("🗒️ 5-Year Forecast Table")
st.dataframe(df_for.style.format({"Depth": "{:.2f}"}), use_container_width=True)

# ---- save summary row ----
if st.button("💾 Save this forecast"):
    row = {"Well": well}
    yr_depth = (df_for.assign(Y=df_for["Date"].dt.year)
                        .groupby("Y").first()["Depth"])
    for yr in range(2025, 2030):
        row[str(yr)] = round(yr_depth.get(yr, np.nan), 2)
    for col in ["AIC", "BIC", "RMSE test"]:
        row[col] = metrics.get(col, np.nan)
    row["lags"] = ""
    row["layers"] = ""
    st.session_state["summary_rows"].append(pd.DataFrame([row]))
    st.success(f"Saved! Total rows: {len(st.session_state['summary_rows'])}")

# ---- sidebar combined download ----
n_rows = len(st.session_state["summary_rows"])
st.sidebar.markdown(f"**Saved summaries:** {n_rows}")
if n_rows:
    combined = pd.concat(st.session_state["summary_rows"]).reset_index(drop=True)
    st.sidebar.download_button("⬇️ Download summary CSV",
                               combined.to_csv(index=False).encode(),
                               file_name=f"well_summaries_{datetime.today().date()}.csv",
                               mime="text/csv")
    if st.sidebar.checkbox("Show summary table"):
        st.subheader("📋 Combined Saved Summaries")
        st.dataframe(combined, use_container_width=True)
