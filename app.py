# app.py — Groundwater forecasting (ANN + ARIMA)  ✨ fixed fitted-values error ✨
# ---------------------------------------------------------------------------
# Key change:  train_arima now returns the *ARIMAResults* object so we can
#              access `.fittedvalues`. Forecast table still produced.

import streamlit as st, pandas as pd, numpy as np, plotly.express as px
from pathlib import Path; from datetime import datetime
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Groundwater Forecasting (ANN & ARIMA)", layout="wide")
st.title("Groundwater Forecasting – ANN vs ARIMA (Depth View)")

DATA_PATH, FORE_HORIZON = "GW data (missing filled).csv", 60  # 5 years

# ───────── helpers ─────────
@st.cache_data(show_spinner=False)
def load_data(path):
    if not Path(path).exists(): return None
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Year"].astype(str)+"-"+df["Months"].astype(str)+"-01")
    df["month_sin"] = np.sin(2*np.pi*df["Months"]/12); df["month_cos"] = np.cos(2*np.pi*df["Months"]/12)
    return df.sort_values("Date").reset_index(drop=True)

def add_lags(df, well, n):
    out=df.copy()
    for k in range(1,n+1): out[f"{well}_lag{k}"]=out[well].shift(k)
    return out.dropna().reset_index(drop=True)

# ─── ANN ───
def train_ann(df_feat, well, layers, lags):
    X=df_feat.drop(columns=[well,"Date"]); y=df_feat[well]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,shuffle=False)
    sc=StandardScaler(); Xtr_s=sc.fit_transform(Xtr); Xte_s=sc.transform(Xte)
    mdl=MLPRegressor(hidden_layer_sizes=layers,max_iter=2000,random_state=42,early_stopping=True).fit(Xtr_s,ytr)
    df_feat.loc[Xtr.index,"pred"]=mdl.predict(Xtr_s); df_feat.loc[Xte.index,"pred"]=mdl.predict(Xte_s)
    met={"R² train":round(r2_score(ytr,df_feat.loc[Xtr.index,"pred"]),4),
         "RMSE train":round(np.sqrt(mean_squared_error(ytr,df_feat.loc[Xtr.index,"pred"])),4),
         "R² test":round(r2_score(yte,df_feat.loc[Xte.index,"pred"]),4),
         "RMSE test":round(np.sqrt(mean_squared_error(yte,df_feat.loc[Xte.index,"pred"])),4)}
    feats=sc.feature_names_in_; r=df_feat.tail(1).iloc[0].copy(); rows=[]
    for _ in range(FORE_HORIZON):
        for k in range(lags,1,-1): r[f"{well}_lag{k}"]=r[f"{well}_lag{k-1}"]
        r[f"{well}_lag1"]=r["pred"]; nxt=r["Date"]+pd.DateOffset(months=1)
        r.update({"Date":nxt,"Months":nxt.month,
                  "month_sin":np.sin(2*np.pi*nxt.month/12),
                  "month_cos":np.cos(2*np.pi*nxt.month/12)})
        val=mdl.predict(sc.transform(r[feats].to_frame().T))[0]
        r[well]=r["pred"]=val; rows.append({"Date":nxt,"Depth":val})
    return met, df_feat, pd.DataFrame(rows)

# ─── ARIMA ───
def train_arima(series, seasonal):
    order, sorder=(1,1,1),(1,1,1,12) if seasonal else (0,0,0,0)
    res=ARIMA(series,order=order,seasonal_order=sorder).fit()
    fc=res.get_forecast(FORE_HORIZON)
    future=pd.DataFrame({"Date":pd.date_range(series.index[-1]+pd.DateOffset(months=1),
                                             periods=FORE_HORIZON,freq="MS"),
                         "Depth":fc.predicted_mean.values})
    met={"AIC":round(res.aic,1),"BIC":round(res.bic,1)}
    return met, res, future

# ───────── UI ─────────
df=load_data(DATA_PATH)
if df is None:
    st.error("CSV not found. Upload it to continue."); 
    if up:=st.file_uploader("Upload CSV",type="csv"):
        Path(DATA_PATH).write_bytes(up.getvalue()); st.experimental_rerun()
    st.stop()

wells=[c for c in df.columns if c.startswith("W")]; well=st.sidebar.selectbox("Well",wells)
model=st.sidebar.radio("Model",["🔮 ANN","📈 ARIMA"])

if model=="🔮 ANN":
    lags=st.sidebar.slider("Lag steps",1,24,12)
    layers=tuple(int(x) for x in st.sidebar.text_input("Hidden layers","64,32").split(",") if x.strip())
    feat=add_lags(df[["Date","Months","month_sin","month_cos",well]],well,lags)
    met,hist,future=train_ann(feat,well,layers,lags)
    st.subheader("ANN metrics"); st.json(met)

    df_act=hist[["Date",well]].rename(columns={well:"Depth"}).assign(Type="Actual")
    df_fit=hist[["Date","pred"]].rename(columns={"pred":"Depth"}).assign(Type="Predicted")
    df_for=future.assign(Type="Forecast")
else:
    seasonal=st.sidebar.checkbox("Include 12-month seasonality",True)
    series=pd.Series(df[well].values,index=df["Date"])
    met,res,future=train_arima(series,seasonal)
    st.subheader("ARIMA metrics"); st.json(met)

    df_act=pd.DataFrame({"Date":series.index,"Depth":series.values,"Type":"Actual"})
    df_fit=pd.DataFrame({"Date":series.index,"Depth":res.fittedvalues,"Type":"Predicted"})
    df_for=future.assign(Type="Forecast")

# ---- plot ----
plot_df=pd.concat([df_act,df_fit,df_for])
fig=px.line(plot_df,x="Date",y="Depth",color="Type",
            labels={"Depth":"Water-table depth (m)"},
            title=f"{well} — {model.strip()} fit & 5-year forecast")
fig.update_yaxes(autorange="reversed")
for t in fig.data:
    if t.name=="Forecast": t.update(line=dict(dash="dash"))
fig.add_vline(x=df_act["Date"].max(),line_dash="dot",line_color="gray")
st.plotly_chart(fig,use_container_width=True)

# ---- table & download ----
st.subheader("🗒️ 5-Year Forecast Table")
st.dataframe(df_for[["Date","Depth"]].style.format({"Depth":"{:.2f}"}),use_container_width=True)
st.download_button("Download forecast CSV",
                   df_for[["Date","Depth"]].to_csv(index=False).encode(),
                   file_name=f"{well}_{model.strip()}_forecast_{datetime.today().date()}.csv",
                   mime="text/csv")
