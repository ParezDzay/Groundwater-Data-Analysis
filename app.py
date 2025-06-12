# app.py  – Streamlit dashboard
# ───────────────────────────────────────────────────────────────────
# Pages
#   • Home
#   • 🔮 ANN Forecast
#   • 📈 ARIMA Forecast
#   • 📋 Saved Summaries  (download one combined CSV)
# -------------------------------------------------------------------
import streamlit as st, pandas as pd, numpy as np, plotly.express as px
from pathlib import Path
from datetime import datetime
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA

# ───────── CONFIG ─────────
st.set_page_config(page_title="Groundwater Forecasts", layout="wide")
DATA_PATH = "GW data (missing filled).csv"
HORIZON_M = 60  # 5-year forecast

# ───────── UTILS ─────────
@st.cache_data
def load_raw(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" +
                                df["Months"].astype(str) + "-01")
    return df.sort_values("Date").reset_index(drop=True)

def clean_series(df, well):
    s = df[well]
    q1, q3 = s.quantile(0.25), s.quantile(0.75); iqr = q3-q1
    s = s.where(s.between(q1-3*iqr, q3+3*iqr)).interpolate(limit_direction="both")
    out = pd.DataFrame({"Date": df["Date"], well: s, "Months": df["Date"].dt.month})
    out["month_sin"] = np.sin(2*np.pi*out["Months"]/12)
    out["month_cos"] = np.cos(2*np.pi*out["Months"]/12)
    return out.dropna().reset_index(drop=True)

def add_lags(df, well, n):
    o = df.copy()
    for k in range(1, n+1):
        o[f"{well}_lag{k}"] = o[well].shift(k)
    return o.dropna().reset_index(drop=True)

def clip_bounds(series):
    lo, hi = series.min(), series.max()
    rng = hi - lo if hi > lo else max(hi, 1)
    return max(0, lo-0.2*rng), hi + 0.2*rng

# ───────── MODEL FUNCTIONS ─────────
def fit_ann(df_feat, well, layers, lags, scaler_type, lo, hi):
    X = df_feat.drop(columns=[well,"Date"]); y = df_feat[well]
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.2,shuffle=False)
    scaler = RobustScaler() if scaler_type=="Robust" else StandardScaler()
    mdl = MLPRegressor(hidden_layer_sizes=layers, max_iter=500,
                       random_state=42, early_stopping=True)
    mdl.fit(scaler.fit_transform(Xtr), ytr)
    ytr_pred = np.clip(mdl.predict(scaler.transform(Xtr)), lo, hi)
    yte_pred = np.clip(mdl.predict(scaler.transform(Xte)), lo, hi)
    df_feat.loc[Xtr.index,"pred"] = ytr_pred
    df_feat.loc[Xte.index,"pred"] = yte_pred
    metrics = {"R² train":round(r2_score(ytr,ytr_pred),4),
               "RMSE train":round(np.sqrt(mean_squared_error(ytr,ytr_pred)),4),
               "R² test":round(r2_score(yte,yte_pred),4),
               "RMSE test":round(np.sqrt(mean_squared_error(yte,yte_pred)),4)}
    feats = scaler.feature_names_in_
    r = df_feat.tail(1).iloc[0].copy(); fut=[]
    for _ in range(HORIZON_M):
        for k in range(lags,1,-1):
            r[f"{well}_lag{k}"] = r[f"{well}_lag{k-1}"]
        r[f"{well}_lag1"] = r["pred"]
        nxt = r["Date"] + pd.DateOffset(months=1)
        r.update({"Date":nxt,"Months":nxt.month,
                  "month_sin":np.sin(2*np.pi*nxt.month/12),
                  "month_cos":np.cos(2*np.pi*nxt.month/12)})
        val = np.clip(mdl.predict(scaler.transform(r[feats].to_frame().T))[0], lo, hi)
        r[well] = r["pred"] = val
        fut.append({"Date":nxt,"Depth":val})
    return metrics, df_feat, pd.DataFrame(fut)

def fit_arima(series, seasonal, lo, hi):
    order, sorder = (1,1,1), (1,1,1,12) if seasonal else (0,0,0,0)
    if len(series) < 20:
        order, sorder = (0,1,1), (0,0,0,0)
    try:
        res = ARIMA(series, order=order, seasonal_order=sorder).fit()
    except Exception:
        res = ARIMA(series, order=(0,1,1)).fit()
    fc = res.get_forecast(HORIZON_M)
    future = pd.DataFrame({"Date": pd.date_range(series.index[-1]+pd.DateOffset(months=1),
                                                 periods=HORIZON_M, freq="MS"),
                           "Depth": np.clip(fc.predicted_mean.values, lo, hi)})
    metrics = {"AIC":round(res.aic,1),
               "BIC":round(res.bic,1),
               "RMSE test":round(np.sqrt(mean_squared_error(
                              series, res.fittedvalues)),4)}
    return metrics, res.fittedvalues.clip(lo,hi), future

# ───────── SESSION STORE ─────────
if "summaries" not in st.session_state:
    st.session_state["summaries"] = []   # list of DataFrames

# ───────── PAGE ROUTER ─────────
page = st.sidebar.radio("Page", ["Home", "🔮 ANN Forecast", "📈 ARIMA Forecast",
                                 "📋 Saved Summaries"])

raw = load_raw(DATA_PATH)
if raw is None:
    st.error("CSV not found."); st.stop()

wells = [c for c in raw.columns if c.startswith("W")]

# ------------------------------------------------------------------ HOME
if page == "Home":
    st.markdown("Choose **ANN Forecast** or **ARIMA Forecast** in the sidebar.")

# ------------------------------------------------------------------ ANN
elif page == "🔮 ANN Forecast":
    well  = st.selectbox("Well", wells, key="ann_well")
    lags  = st.slider("Lag steps", 1, 24, 12)
    layers_txt = st.text_input("Hidden layers", "64,32")
    layers = tuple(int(x) for x in layers_txt.split(",") if x.strip())
    scaler_choice = st.selectbox("Scaler", ["Standard","Robust"])
    if st.button("Run forecast"):
        clean = clean_series(raw, well)
        if len(clean) < lags*10:
            st.warning("Not enough data for that many lags."); st.stop()
        lo, hi = clip_bounds(clean[well])
        feat = add_lags(clean, well, lags)
        m, hist, fut = fit_ann(feat, well, layers, lags, scaler_choice, lo, hi)

        # ---------- PLOT ----------
        df_act = pd.DataFrame({"Date":clean["Date"], "Depth":clean[well], "Type":"Actual"})
        df_fit = hist[["Date","pred"]].rename(columns={"pred":"Depth"}).assign(Type="Predicted")
        df_for = fut.assign(Type="Forecast")
        fig = px.line(pd.concat([df_act, df_fit, df_for]), x="Date", y="Depth", color="Type",
                      title=f"{well} — ANN forecast")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
        st.json(m)

        st.dataframe(df_for, use_container_width=True)

        if st.button("💾 Save summary"):
            row = {"Well": well, "lags": lags,
                   "layers": layers_txt, **{str(y): round(df_for[df_for.Date.dt.year==y]["Depth"].mean(),2)
                                            for y in range(2025,2030)},
                   **m}
            st.session_state["summaries"].append(pd.DataFrame([row]))
            st.success("Summary saved.")

# ------------------------------------------------------------------ ARIMA
elif page == "📈 ARIMA Forecast":
    well = st.selectbox("Well", wells, key="arima_well")
    seasonal = st.checkbox("Include 12-month seasonality", True)
    if st.button("Run forecast", key="arima_run"):
        clean = clean_series(raw, well)
        lo, hi = clip_bounds(clean[well])
        series = pd.Series(clean[well].values, index=clean["Date"])
        m, fit_series, fut = fit_arima(series, seasonal, lo, hi)

        df_act = pd.DataFrame({"Date":series.index, "Depth":series.values, "Type":"Actual"})
        df_fit = pd.DataFrame({"Date":series.index, "Depth":fit_series, "Type":"Predicted"})
        df_for = fut.assign(Type="Forecast")
        fig = px.line(pd.concat([df_act, df_fit, df_for]), x="Date", y="Depth", color="Type",
                      title=f"{well} — ARIMA forecast")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
        st.json(m)
        st.dataframe(df_for, use_container_width=True)

        if st.button("💾 Save summary", key="arima_save"):
            row = {"Well": well, "lags": "", "layers": "",
                   **{str(y): round(df_for[df_for.Date.dt.year==y]["Depth"].mean(),2)
                      for y in range(2025,2030)},
                   **m}
            st.session_state["summaries"].append(pd.DataFrame([row]))
            st.success("Summary saved.")

# ------------------------------------------------------------------ SAVED
elif page == "📋 Saved Summaries":
    cnt = len(st.session_state["summaries"])
    st.write(f"### Total summaries saved: {cnt}")
    if cnt:
        combined = pd.concat(st.session_state["summaries"]).reset_index(drop=True)
        st.dataframe(combined, use_container_width=True)
        st.download_button("⬇️ Download combined CSV",
                           combined.to_csv(index=False).encode(),
                           file_name=f"well_summaries_{datetime.today().date()}.csv",
                           mime="text/csv")
    else:
        st.info("No summaries saved yet. Run a forecast first.")
