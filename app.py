# app.py — Groundwater Forecasting (with caching & auto-save)
# Models: SARIMA · Random-Forest (lags) · LSTM · CNN-LSTM   (TF optional)
# -----------------------------------------------------------------------
# runtime.txt    python-3.11.8
# -----------------------------------------------------------------------

import streamlit as st, pandas as pd, numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os, warnings, hashlib, json
warnings.filterwarnings("ignore", category=UserWarning)

# ───────── optional TF import ─────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten
    from tensorflow.keras.callbacks import EarlyStopping
    tf.get_logger().setLevel("ERROR")
    _TF = True
except ModuleNotFoundError:
    _TF = False

# ───────── page config ─────────
st.set_page_config(page_title="Groundwater Forecasts", layout="wide")
st.title("Groundwater Forecasting — Classic, ML & Deep (cached)")

DATA_PATH, HORIZON_M = "GW data (missing filled).csv", 60
SUMMARY_CSV = "yearly_summaries.csv"

# ───────── init session dicts ─────────
st.session_state.setdefault("model_cache",  {})   # key → (metrics, future)
st.session_state.setdefault("summary_rows", [])   # list of DataFrames

# ───────── util helpers ─────────
def cache_key(**kwargs) -> str:
    """Stable hash of parameters for cache & CSV de-dup."""
    return hashlib.md5(json.dumps(kwargs, sort_keys=True).encode()).hexdigest()

@st.cache_data(show_spinner=False)
def load_raw(path: str):
    if not Path(path).exists(): return None
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Year"].astype(str)+"-"+df["Months"].astype(str).str.zfill(2)+"-01")
    return df.sort_values("Date").reset_index(drop=True)

def clean_series(df, well):
    s = df[well].copy()
    q1,q3 = s.quantile([0.25,0.75]); iqr=q3-q1
    s = s.where(s.between(q1-3*iqr,q3+3*iqr)).interpolate(limit_direction="both")
    return pd.Series(s.values,index=df["Date"])

# ───────── model back-ends ─────────
def sarima(series):
    order,s_order=(1,1,1),(1,1,1,12)
    split=int(len(series)*0.8)
    mdl= SARIMAX(series.iloc[:split],order=order,seasonal_order=s_order,
                 enforce_stationarity=False,enforce_invertibility=False).fit(disp=False)
    rmse=round(np.sqrt(mean_squared_error(series.iloc[split:],mdl.forecast(len(series)-split))),4)
    full= SARIMAX(series,order=order,seasonal_order=s_order,
                  enforce_stationarity=False,enforce_invertibility=False).fit(disp=False)
    fc=full.forecast(HORIZON_M).round(2)
    idx=pd.date_range(series.index[-1]+pd.DateOffset(months=1),periods=HORIZON_M,freq="MS")
    return {"RMSE":rmse,"AIC":round(full.aic,1),"BIC":round(full.bic,1)}, pd.Series(fc.values,index=idx)

def rf(series,n_lags):
    df_lag=pd.concat({f"lag{k}":series.shift(k) for k in range(1,n_lags+1)},axis=1).dropna()
    X,y=df_lag.values,series.loc[df_lag.index].values
    split=int(len(X)*0.8)
    rf=RandomForestRegressor(n_estimators=300,random_state=42).fit(X[:split],y[:split])
    rmse=round(np.sqrt(mean_squared_error(y[split:],rf.predict(X[split:]))),4)
    hist=list(series.values[-n_lags:]); fc=[]
    for _ in range(HORIZON_M):
        nxt=rf.predict(np.array(hist[-n_lags:][::-1]).reshape(1,-1))[0]
        fc.append(round(nxt,2)); hist.append(nxt)
    idx=pd.date_range(series.index[-1]+pd.DateOffset(months=1),periods=HORIZON_M,freq="MS")
    return {"RMSE":rmse,"Lags":n_lags,"Trees":300}, pd.Series(fc,index=idx)

if _TF:
    def _build(shape,kind="lstm"):
        if kind=="lstm":
            net=Sequential([LSTM(64,activation="tanh",input_shape=shape),Dense(1)])
        else:
            net=Sequential([Conv1D(32,3,activation="relu",input_shape=shape),
                            MaxPooling1D(2),Flatten(),
                            Dense(32,activation="relu"),Dense(1)])
        net.compile(optimizer="adam",loss="mse"); return net
    def deep(series,n_lags,epochs,kind):
        sc=MinMaxScaler(); scaled=sc.fit_transform(series.to_numpy().reshape(-1,1)).flatten()
        X,y=[],[]
        for i in range(n_lags,len(scaled)):
            X.append(scaled[i-n_lags:i]); y.append(scaled[i])
        X,y=np.array(X),np.array(y)
        split=int(len(X)*0.8)
        Xtr,Xte,ytr,yte=X[:split],X[split:],y[:split],y[split:]
        Xtr=Xtr.reshape((Xtr.shape[0],Xtr.shape[1],1))
        Xte=Xte.reshape((Xte.shape[0],Xte.shape[1],1))
        net=_build((n_lags,1),kind)
        net.fit(Xtr,ytr,validation_data=(Xte,yte),
                epochs=epochs,batch_size=16,verbose=0,
                callbacks=[EarlyStopping(patience=5,restore_best_weights=True)])
        rmse=round(np.sqrt(mean_squared_error(yte,net.predict(Xte,verbose=0).flatten())),4)
        hist=list(scaled[-n_lags:]); fc=[]
        for _ in range(HORIZON_M):
            yhat=net.predict(np.array(hist[-n_lags:]).reshape(1,n_lags,1),verbose=0)[0][0]
            fc.append(yhat); hist.append(yhat)
        fc=sc.inverse_transform(np.array(fc).reshape(-1,1)).flatten().round(2)
        idx=pd.date_range(series.index[-1]+pd.DateOffset(months=1),periods=HORIZON_M,freq="MS")
        return {"RMSE":rmse,"Lags":n_lags,"Epochs":epochs}, pd.Series(fc,index=idx)

# ───────── UI controls ─────────
raw=load_raw(DATA_PATH)
if raw is None:
    st.error("CSV not found. Upload below.")
    if f:=st.sidebar.file_uploader("Upload CSV",type="csv"): Path(DATA_PATH).write_bytes(f.read()); st.experimental_rerun()
    st.stop()

wells=[c for c in raw.columns if c.startswith("W")]
scope=st.sidebar.radio("Scope",["Single well","All wells"])
if scope=="Single well": well=st.sidebar.selectbox("Well",wells)

models=["SARIMA",
        "Random-Forest",
        "LSTM"+("" if _TF else " ❌"),
        "CNN-LSTM"+("" if _TF else " ❌")]
model_choice=st.sidebar.radio("Model",models)

if model_choice=="Random-Forest":
    n_lags=st.sidebar.slider("RF lags",6,24,12,2)
elif model_choice.startswith("LSTM") or model_choice.startswith("CNN"):
    if not _TF: st.error("TensorFlow not installed."); st.stop()
    n_lags=st.sidebar.slider("DL lags",6,24,12,2)
    epochs=st.sidebar.slider("Epochs",10,100,30,10)

# ───────── core run with caching ─────────
def get_forecast(this_well):
    key_params={"well":this_well,"model":model_choice,"lags":n_lags if 'n_lags' in globals() else None,
                "epochs":epochs if 'epochs' in globals() else None}
    k=cache_key(**key_params)
    if k in st.session_state["model_cache"]:
        return st.session_state["model_cache"][k]

    series=clean_series(raw,this_well)
    if model_choice=="SARIMA":                metrics,future=sarima(series)
    elif model_choice=="Random-Forest":       metrics,future=rf(series,n_lags)
    elif model_choice.startswith("LSTM"):     metrics,future=deep(series,n_lags,epochs,"lstm")
    else:                                     metrics,future=deep(series,n_lags,epochs,"cnn")

    st.session_state["model_cache"][k]=(metrics,future)
    return metrics,future

# ───────── run for target wells ─────────
targets=wells if scope=="All wells" else [well]
summary=[]
with st.spinner("Training / forecasting…"):
    for w in targets:
        metrics,future=get_forecast(w)
        annual=future.resample("A").mean()
        row={"Well":w}
        for yr in range(2025,2030):
            sel=annual[annual.index.year==yr]
            row[str(yr)]=round(sel.iloc[0],2) if not sel.empty else np.nan
        row.update(metrics); summary.append(row)

summary_df=pd.DataFrame(summary)
st.subheader("Yearly average depth (m)")
st.dataframe(summary_df,use_container_width=True)

if scope=="Single well":
    metrics,future=get_forecast(well)
    st.subheader("Model metrics")
    st.table(pd.DataFrame(metrics,index=["Value"]))
    st.subheader("5-year monthly forecast")
    st.dataframe(future.to_frame("Depth"),use_container_width=True)

# ───────── auto-save summaries ─────────
def append_unique(csv_path,df,key_cols):
    if os.path.exists(csv_path):
        current=pd.read_csv(csv_path)
        merged=pd.concat([current,df]).drop_duplicates(subset=key_cols,keep="last")
        merged.to_csv(csv_path,index=False)
    else:
        df.to_csv(csv_path,index=False)

append_unique(SUMMARY_CSV,summary_df,["Well"]+ [str(y) for y in range(2025,2030)] + list(summary_df.columns.drop(["Well"])))

st.session_state["summary_rows"].append(summary_df)

# ───────── download buttons ─────────
n_rows=len(st.session_state["summary_rows"])
st.sidebar.markdown(f"**Summary tables this session:** {n_rows}")
if n_rows:
    combo=pd.concat(st.session_state["summary_rows"]).reset_index(drop=True)
    st.sidebar.download_button("⬇ Session CSV",combo.to_csv(index=False).encode(),
                               f"session_summaries_{datetime.today().date()}.csv","text/csv")
if os.path.exists(SUMMARY_CSV):
    with open(SUMMARY_CSV,"rb") as f:
        st.sidebar.download_button("⬇ Consolidated CSV",f.read(),
                                   SUMMARY_CSV,"text/csv")
