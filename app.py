# app.py — Groundwater Forecasting  (FAST-mode default)
# ---------------------------------------------------------------------
# Models          : SARIMA · Random-Forest (lags) · LSTM · CNN-LSTM   (TF optional)
# Fast-mode tweaks:             ──────────────────────────────────────
#   • SARIMA   fixed (0,1,1)(0,1,1,12)  +  maxiter=25
#   • RF       80 trees (vs 300)        +  lags slider
#   • DL       default epochs = 8 (slider visible when fast-mode off)
#   • All-well horizon = 24 months in fast-mode (60 when off)
#   • Parallel loop across wells (joblib) when >1 well
#   • Cache by (well, model, lags, epochs, fast) to avoid re-fit
#
# runtime.txt   python-3.11.8
# requirements  streamlit==1.45.1  pandas==2.3.0  numpy==1.26.4
#               scikit-learn==1.4.2  statsmodels==0.14.4
#               joblib==1.4.0  tensorflow-cpu==2.15.0  (optional)
# ---------------------------------------------------------------------

import streamlit as st, pandas as pd, numpy as np, os, json, hashlib, warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from joblib import Parallel, delayed
warnings.filterwarnings("ignore", category=UserWarning)

# ───────── optional TensorFlow ─────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten
    from tensorflow.keras.callbacks import EarlyStopping
    tf.get_logger().setLevel("ERROR")
    _TF = True
except ModuleNotFoundError:
    _TF = False

# ───────── Streamlit page ─────────
st.set_page_config(page_title="Groundwater Forecasts", layout="wide")
st.title("Groundwater Forecasting — fast by default")

DATA_PATH   = "GW data (missing filled).csv"
SUMMARY_CSV = "yearly_summaries.csv"

# ───────── session state ─────────
st.session_state.setdefault("cache",       {})   # key → (metrics,future)
st.session_state.setdefault("summ_tables", [])   # list of dfs this session

# ───────── helpers ─────────
def md5(obj): return hashlib.md5(json.dumps(obj,sort_keys=True).encode()).hexdigest()

@st.cache_data(show_spinner=False)
def load_raw(path):
    if not Path(path).exists(): return None
    df=pd.read_csv(path)
    df["Date"]=pd.to_datetime(df["Year"].astype(str)+"-"+df["Months"].astype(str).str.zfill(2)+"-01")
    return df.sort_values("Date").reset_index(drop=True)

def clean_series(df,well):
    s=df[well].copy()
    q1,q3=s.quantile([0.25,0.75]); iqr=q3-q1
    s=s.where(s.between(q1-3*iqr,q3+3*iqr)).interpolate(limit_direction="both")
    return pd.Series(s.values,index=df["Date"])

# ───────── model back-ends ─────────
def sarima(series,H,fast):
    mdl = SARIMAX(series, order=(0,1,1), seasonal_order=(0,1,1,12),
                  enforce_stationarity=False, enforce_invertibility=False)
    res = mdl.fit(disp=False, maxiter=25 if fast else 50)
    fc  = res.get_forecast(H).predicted_mean.round(2)
    idx = pd.date_range(series.index[-1]+pd.DateOffset(months=1), periods=H, freq="MS")
    return {"RMSE":round(res.mse**0.5,4),"AIC":round(res.aic,1),"BIC":round(res.bic,1)}, pd.Series(fc.values,idx)

def rf(series,H,lags,fast):
    trees = 80 if fast else 300
    df_lag = pd.concat({f"lag{k}":series.shift(k) for k in range(1,lags+1)},axis=1).dropna()
    X,y   = df_lag.values, series.loc[df_lag.index].values
    split = int(len(X)*0.8)
    rf    = RandomForestRegressor(n_estimators=trees, random_state=1).fit(X[:split],y[:split])
    rmse  = round(np.sqrt(mean_squared_error(y[split:], rf.predict(X[split:]))),4)
    hist  = list(series.values[-lags:]); fc=[]
    for _ in range(H):
        nxt=rf.predict(np.array(hist[-lags:][::-1]).reshape(1,-1))[0]
        fc.append(round(nxt,2)); hist.append(nxt)
    idx = pd.date_range(series.index[-1]+pd.DateOffset(months=1),periods=H,freq="MS")
    return {"RMSE":rmse,"Lags":lags,"Trees":trees}, pd.Series(fc,idx)

if _TF:
    def _build(shape,kind):
        if kind=="lstm":
            net=Sequential([LSTM(64,activation="tanh",input_shape=shape),Dense(1)])
        else:
            net=Sequential([Conv1D(32,3,activation="relu",input_shape=shape),
                            MaxPooling1D(2),Flatten(),Dense(32,activation="relu"),Dense(1)])
        net.compile(optimizer="adam",loss="mse"); return net
    def deep(series,H,lags,epochs,kind):
        sc=MinMaxScaler()
        scaled=sc.fit_transform(series.values.reshape(-1,1)).flatten()
        X,y=[],[]
        for i in range(lags,len(scaled)):
            X.append(scaled[i-lags:i]); y.append(scaled[i])
        X,y=np.array(X),np.array(y)
        split=int(len(X)*0.8)
        Xtr,Xte,ytr,yte=X[:split],X[split:],y[:split],y[split:]
        Xtr=Xtr.reshape((-1,lags,1)); Xte=Xte.reshape((-1,lags,1))
        net=_build((lags,1),kind)
        net.fit(Xtr,ytr,validation_data=(Xte,yte),epochs=epochs,batch_size=16,
                verbose=0,callbacks=[EarlyStopping(patience=3,restore_best_weights=True)])
        rmse=round(np.sqrt(mean_squared_error(yte,net.predict(Xte,verbose=0).flatten())),4)
        hist=list(scaled[-lags:]); fc=[]
        for _ in range(H):
            yhat=net.predict(np.array(hist[-lags:]).reshape(1,lags,1),verbose=0)[0][0]
            fc.append(yhat); hist.append(yhat)
        fc=sc.inverse_transform(np.array(fc).reshape(-1,1)).flatten().round(2)
        idx=pd.date_range(series.index[-1]+pd.DateOffset(months=1),periods=H,freq="MS")
        return {"RMSE":rmse,"Lags":lags,"Epochs":epochs}, pd.Series(fc,idx)

# ───────── UI controls ─────────
raw=load_raw(DATA_PATH)
if raw is None:
    st.error("CSV not found. Upload below.")
    if up:=st.sidebar.file_uploader("Upload CSV",type="csv"):
        Path(DATA_PATH).write_bytes(up.read()); st.experimental_rerun()
    st.stop()

wells=[c for c in raw.columns if c.startswith("W")]
scope=st.sidebar.radio("Scope",["Single well","All wells"])
if scope=="Single well": well=st.sidebar.selectbox("Well",wells)

fast_mode=st.sidebar.checkbox("Fast mode (quick & rough)",value=True)

models=["SARIMA","Random-Forest"]
if _TF:
    models+=["LSTM","CNN-LSTM"] if fast_mode else ["LSTM","CNN-LSTM"]  # deep still allowed
else:
    models+=["LSTM ❌","CNN-LSTM ❌"]
model_choice=st.sidebar.radio("Model",models)

if model_choice=="Random-Forest":
    n_lags=st.sidebar.slider("RF lags",6,24,12,2)
elif model_choice.startswith(("LSTM","CNN")) and _TF:
    n_lags=st.sidebar.slider("DL lags",6,24,12,2)
    epochs=8 if fast_mode else st.sidebar.slider("Epochs",10,100,30,10)

H = 24 if (scope=="All wells" and fast_mode) else 60

# ───────── cache fetch/run ─────────
def forecast_for(well_id):
    key=md5({"well":well_id,"model":model_choice,"lags":locals().get("n_lags"),
             "epochs":locals().get("epochs"),"fast":fast_mode,"H":H})
    if key in st.session_state["cache"]:
        return st.session_state["cache"][key]
    s=clean_series(raw,well_id)
    if model_choice=="SARIMA":       m,f=sarima(s,H,fast_mode)
    elif model_choice=="Random-Forest": m,f=rf(s,H,n_lags,fast_mode)
    elif model_choice=="LSTM":       m,f=deep(s,H,n_lags,epochs,"lstm")
    else:                            m,f=deep(s,H,n_lags,epochs,"cnn")
    st.session_state["cache"][key]=(m,f); return m,f

# ───────── run (parallel when many) ─────────
targets=wells if scope=="All wells" else [well]

def run_one(w): return w,*forecast_for(w)

if len(targets)>1:
    results=Parallel(n_jobs=-1)(delayed(run_one)(w) for w in targets)
else:
    results=[run_one(targets[0])]

rows=[]
for w,metrics,future in results:
    annual=future.resample("A").mean()
    row={"Well":w}
    for yr in range(2025,2030):
        sel=annual[annual.index.year==yr]
        row[str(yr)]=round(sel.iloc[0],2) if not sel.empty else np.nan
    row.update(metrics); rows.append(row)

summary=pd.DataFrame(rows)
st.subheader("Yearly average depth (m)")
st.dataframe(summary,use_container_width=True)

# detailed view for single well
if scope=="Single well":
    _,metrics,future=results[0]
    st.subheader("Model metrics"); st.table(pd.DataFrame(metrics,index=["Value"]))
    st.subheader(f"{H//12}-year monthly forecast"); st.dataframe(future.to_frame("Depth"),use_container_width=True)

# ───────── auto-append & dedup CSV ─────────
if not summary.empty:
    if os.path.exists(SUMMARY_CSV):
        cur=pd.read_csv(SUMMARY_CSV)
        all_=pd.concat([cur,summary]).drop_duplicates(subset=["Well","RMSE","Lags","Epochs","Trees"],keep="last")
        all_.to_csv(SUMMARY_CSV,index=False)
    else:
        summary.to_csv(SUMMARY_CSV,index=False)
    st.session_state["summ_tables"].append(summary)

# ───────── downloads ─────────
n=len(st.session_state["summ_tables"])
st.sidebar.markdown(f"**Session tables:** {n}")
if n:
    combo=pd.concat(st.session_state["summ_tables"]).reset_index(drop=True)
    st.sidebar.download_button("⬇ Session CSV",combo.to_csv(index=False).encode(),
                               f"session_{datetime.today().date()}.csv","text/csv")
if os.path.exists(SUMMARY_CSV):
    with open(SUMMARY_CSV,"rb") as f:
        st.sidebar.download_button("⬇ All-runs CSV",f.read(),SUMMARY_CSV,"text/csv")
