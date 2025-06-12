# app.py — Groundwater forecasts (ANN + ARIMA)
# --------------------------------------------
# • “Save this forecast” adds ONE summary row per well:
#   Well , 2025 , 2026 , 2027 , 2028 , 2029 , R² train , RMSE train ,
#   R² test , RMSE test , lags , layers
# • Sidebar shows running count + combined-CSV download

import streamlit as st, pandas as pd, numpy as np, plotly.express as px
from pathlib import Path
from datetime import datetime
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Groundwater Forecasts", layout="wide")
st.title("Groundwater Forecasting — Depth View")

DATA_PATH  = "GW data (missing filled).csv"
HORIZON_M  = 60         # 5-year horizon

# ---------- helpers ----------
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
    out = pd.DataFrame({"Date": df["Date"], well: s, "Months": df["Date"].dt.month})
    out["month_sin"] = np.sin(2*np.pi*out["Months"]/12)
    out["month_cos"] = np.cos(2*np.pi*out["Months"]/12)
    return out.dropna().reset_index(drop=True)

def add_lags(df, well, n):
    out = df.copy()
    for k in range(1, n+1):
        out[f"{well}_lag{k}"] = out[well].shift(k)
    return out.dropna().reset_index(drop=True)

def clip_bounds(series):
    lo, hi = series.min(), series.max()
    rng = hi - lo if hi > lo else max(hi, 1)
    return max(0, lo - 0.2*rng), hi + 0.2*rng

# ---------- ANN ----------
def train_ann(df_feat, well, layers, lags, scaler_type, lo, hi):
    X = df_feat.drop(columns=[well,"Date"]); y=df_feat[well]
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,shuffle=False)
    scaler = RobustScaler() if scaler_type=="Robust" else StandardScaler()
    mdl = MLPRegressor(hidden_layer_sizes=layers,max_iter=2000,
                       random_state=42,early_stopping=True)
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
        for k in range(lags,1,-1): r[f"{well}_lag{k}"] = r[f"{well}_lag{k-1}"]
        r[f"{well}_lag1"] = r["pred"]
        nxt = r["Date"] + pd.DateOffset(months=1)
        r.update({"Date":nxt,"Months":nxt.month,
                  "month_sin":np.sin(2*np.pi*nxt.month/12),
                  "month_cos":np.cos(2*np.pi*nxt.month/12)})
        val = np.clip(mdl.predict(scaler.transform(r[feats].to_frame().T))[0], lo, hi)
        r[well] = r["pred"] = val
        fut.append({"Date":nxt,"Depth":val})
    return metrics, df_feat, pd.DataFrame(fut)

# ---------- ARIMA ----------
def train_arima(series, seasonal, lo, hi):
    split = int(len(series)*0.8)
    train, test = series.iloc[:split], series.iloc[split:]
    res = ARIMA(train,order=(1,1,1),
                seasonal_order=(1,1,1,12) if seasonal else (0,0,0,0)).fit()
    rmse = round(np.sqrt(mean_squared_error(test, res.forecast(len(test)))),4)
    res_full = ARIMA(series,order=(1,1,1),
                     seasonal_order=(1,1,1,12) if seasonal else (0,0,0,0)).fit()
    fc = res_full.get_forecast(HORIZON_M)
    future = pd.DataFrame({"Date":pd.date_range(series.index[-1]+pd.DateOffset(months=1),
                                               periods=HORIZON_M,freq="MS"),
                           "Depth":np.clip(fc.predicted_mean.values, lo, hi)})
    metrics = {"AIC":round(res_full.aic,1),"BIC":round(res_full.bic,1),"RMSE test":rmse}
    return metrics, res_full, future

# ---------- session storage ----------
if "summary_rows" not in st.session_state:
    st.session_state["summary_rows"] = []

# ---------- UI ---------- 
raw = load_raw(DATA_PATH)
if raw is None:
    st.error("CSV not found. Upload to continue.")
    if up:=st.file_uploader("Upload CSV",type="csv"):
        Path(DATA_PATH).write_bytes(up.getvalue()); st.experimental_rerun()
    st.stop()

wells=[c for c in raw.columns if c.startswith("W")]
well = st.sidebar.selectbox("Well",wells)
model = st.sidebar.radio("Model",["🔮 ANN","📈 ARIMA"])

clean = clean_series(raw, well)
lo, hi = clip_bounds(clean[well])

if model=="🔮 ANN":
    lags = st.sidebar.slider("Lag steps",1,24,12)
    if len(clean)<lags*10: lags=max(1,len(clean)//10); st.info(f"Lags auto-reduced to {lags}")
    layers = tuple(int(x) for x in st.sidebar.text_input("Hidden layers","64,32").split(",") if x.strip())
    scaler_choice = st.sidebar.selectbox("Scaler",["Standard","Robust"])
    feat = add_lags(clean,well,lags)
    metrics,hist,future = train_ann(feat,well,layers,lags,scaler_choice,lo,hi)
    meta = {"lags":lags,"layers":",".join(map(str,layers))}
else:
    seasonal = st.sidebar.checkbox("Include 12-month seasonality",True)
    series = pd.Series(clean[well].values,index=clean["Date"])
    metrics,res,future = train_arima(series,seasonal,lo,hi)
    meta = {"lags":"","layers":""}

st.subheader(f"{model.strip()} metrics"); st.json(metrics)

# ---- plot ----
df_act = pd.DataFrame({"Date":clean["Date"], "Depth":clean[well], "Type":"Actual"})
df_fit = (hist[["Date","pred"]].rename(columns={"pred":"Depth"})
          if model=="🔮 ANN" else
          pd.DataFrame({"Date":series.index,"Depth":res.fittedvalues.clip(lo,hi),"Type":"Predicted"}))
df_for = future.assign(Type="Forecast")
plot_df = pd.concat([df_act.assign(Type="Actual"), df_fit.assign(Type="Predicted"), df_for])
fig = px.line(plot_df,x="Date",y="Depth",color="Type",
              labels={"Depth":"Water-table depth (m)"},
              title=f"{well} — {model.strip()} fit & 5-year forecast")
fig.update_yaxes(autorange="reversed")
for t in fig.data:
    if t.name=="Forecast": t.update(line=dict(dash="dash"))
fig.add_vline(x=df_act["Date"].max(),line_dash="dot",line_color="gray")
st.plotly_chart(fig,use_container_width=True)

# ---- table  ----
st.subheader("🗒️ 5-Year Forecast Table")
st.dataframe(df_for.style.format({"Depth":"{:.2f}"}),use_container_width=True)

# ---- SAVE summary row ----
if st.button("💾 Save this forecast"):
    row = {"Well":well}
    # yearly values
    yr_depth = (df_for.assign(Y=df_for["Date"].dt.year)
                        .groupby("Y").first()["Depth"])
    for yr in range(2025,2030):
        row[str(yr)] = round(yr_depth.get(yr, np.nan),2)
    # metrics columns
    for col in ["R² train","RMSE train","R² test","RMSE test"]:
        row[col] = metrics.get(col,np.nan)
    row["lags"]   = meta["lags"]
    row["layers"] = meta["layers"]
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
