# app.py — Forecast dashboard with single combined download
# ---------------------------------------------------------

import streamlit as st, pandas as pd, numpy as np, plotly.express as px
from pathlib import Path; from datetime import datetime
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Groundwater Forecasts (ANN & ARIMA)", layout="wide")
st.title("Groundwater Forecasting — Depth View (clipped, savable)")

DATA_PATH, FORE_HORIZON = "GW data (missing filled).csv", 60

# --- helpers & model functions  (unchanged from last version) ---
# [Load, clean_series, add_lags, clip_bounds, train_ann, train_arima here]
# ... (for brevity, keep the exact same helper functions from the previous script) ...

# ---------- session store ----------
if "saved_forecasts" not in st.session_state:
    st.session_state["saved_forecasts"] = []

# ---------- UI ----------
raw = pd.read_csv(DATA_PATH) if Path(DATA_PATH).exists() else None
if raw is None:
    st.error("CSV not found. Upload it to continue.")
    if up := st.file_uploader("Upload CSV", type="csv"):
        Path(DATA_PATH).write_bytes(up.getvalue()); st.experimental_rerun()
    st.stop()

raw["Date"] = pd.to_datetime(raw["Year"].astype(str) + "-" + raw["Months"].astype(str) + "-01")
wells = [c for c in raw.columns if c.startswith("W")]
well = st.sidebar.selectbox("Well", wells)
model = st.sidebar.radio("Model", ["🔮 ANN", "📈 ARIMA"])

clean = clean_series(raw, well)
lo, hi = clip_bounds(clean[well])

if model == "🔮 ANN":
    lag_steps = st.sidebar.slider("Lag steps", 1, 24, 12)
    if len(clean) < lag_steps * 10:
        lag_steps = max(1, len(clean) // 10)
        st.info(f"Lags auto-reduced to {lag_steps}.")
    layers = tuple(int(x) for x in st.sidebar.text_input(
        "Hidden layers", "64,32").split(",") if x.strip())
    scaler_choice = st.sidebar.selectbox("Scaler", ["Standard", "Robust"])
    feat = add_lags(clean, well, lag_steps)
    metrics, hist, future = train_ann(feat, well, layers, lag_steps,
                                      scaler_choice, lo, hi)
    st.subheader("ANN metrics"); st.json(metrics)
    df_actual = hist[["Date", well]].rename(columns={well: "Depth"}).assign(Type="Actual")
    df_fit = hist[["Date", "pred"]].rename(columns={"pred": "Depth"}).assign(Type="Predicted")
    meta = {"well": well, "model": "ANN", "lags": lag_steps,
            "layers": layers, "scaler": scaler_choice}
else:
    seasonal = st.sidebar.checkbox("Include 12-month seasonality", True)
    series = pd.Series(clean[well].values, index=clean["Date"])
    metrics, res, future = train_arima(series, seasonal, lo, hi)
    st.subheader("ARIMA metrics"); st.json(metrics)
    df_actual = pd.DataFrame({"Date": series.index,
                              "Depth": series.values,
                              "Type": "Actual"})
    df_fit = pd.DataFrame({"Date": series.index,
                           "Depth": res.fittedvalues.clip(lo, hi),
                           "Type": "Predicted"})
    meta = {"well": well, "model": "ARIMA", "seasonal": seasonal}

df_fore = future.assign(Type="Forecast")
plot_df = pd.concat([df_actual, df_fit, df_fore])

fig = px.line(plot_df, x="Date", y="Depth", color="Type",
              title=f"{well} — {model.strip()} fit & 5-year forecast (clipped)",
              labels={"Depth": "Water-table depth (m)"})
fig.update_yaxes(autorange="reversed")
for t in fig.data:
    if t.name == "Forecast":
        t.update(line=dict(dash="dash"))
fig.add_vline(x=df_actual["Date"].max(), line_dash="dot", line_color="gray")
st.plotly_chart(fig, use_container_width=True)

st.subheader("🗒️ 5-Year Forecast Table")
st.dataframe(df_fore.style.format({"Depth": "{:.2f}"}), use_container_width=True)

# ---------- Save this forecast ----------
if st.button("💾 Save this forecast"):
    df_save = df_fore.copy()
    for k, v in meta.items(): df_save[k] = v
    st.session_state["saved_forecasts"].append(df_save)
    st.success(f"Saved! Total saved forecasts: {len(st.session_state['saved_forecasts'])}")

# ---------- Combined download + preview ----------
saved_count = len(st.session_state["saved_forecasts"])
st.sidebar.markdown(f"**Saved forecasts:** {saved_count}")

if saved_count:
    show = st.sidebar.checkbox("Show combined table")
    combined = pd.concat(st.session_state["saved_forecasts"]).reset_index(drop=True)
    st.sidebar.download_button("⬇️ Download combined CSV",
                               combined.to_csv(index=False).encode(),
                               file_name=f"all_saved_forecasts_{datetime.today().date()}.csv",
                               mime="text/csv")
    if show:
        st.subheader("📚 Combined Saved Forecasts")
        st.dataframe(combined.style.format({"Depth": "{:.2f}"}), use_container_width=True)
