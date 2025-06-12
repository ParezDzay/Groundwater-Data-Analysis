# app.py – well-data dashboard + ANN page
# ────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os, base64, requests, io
from datetime import datetime
from pathlib import Path
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ───────── CONFIG ─────────
st.set_page_config(page_title="Well Data App", layout="wide")
st.sidebar.title("Navigation")

token    = st.secrets["github"]["token"]
username = st.secrets["github"]["username"]
repo     = st.secrets["github"]["repo"]
branch   = st.secrets["github"]["branch"]

file_path            = "Wells detailed data.csv"
gw_file_path         = "GW data.csv"
output_path          = "GW data (missing filled).csv"     # used elsewhere
cleaned_outlier_path = "GW data (missing filled).csv"

# NEW: relative data path for ANN
DATA_PATH = "GW data (missing filled).csv"

# ───────── GitHub push helper (unchanged) ─────────
def push_to_github(fp, msg):
    with open(fp, "r", encoding="utf-8") as f:
        content = f.read()
    payload = {
        "message": msg,
        "content": base64.b64encode(content.encode()).decode(),
        "branch": branch
    }
    url  = f"https://api.github.com/repos/{username}/{repo}/contents/{os.path.basename(fp)}"
    hdrs = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}
    sha  = requests.get(url, headers=hdrs).json().get("sha")
    if sha: payload["sha"] = sha
    r = requests.put(url, headers=hdrs, json=payload)
    return r.status_code in (200, 201)

# ───────── ANN helpers ─────────
@st.cache_data(show_spinner=False)
def load_gw(path)->pd.DataFrame:
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    df["Year"]   = df["Year"].astype(int)
    df["Months"] = df["Months"].astype(int)
    if "Date" not in df.columns:
        df["Date"] = pd.to_datetime(df["Year"].astype(str)+"-"+df["Months"].astype(str)+"-01")
    else:
        df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["month_sin"] = np.sin(2*np.pi*df["Months"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["Months"]/12)
    return df

def lag_feats(df, well, n):
    d = df.copy()
    for i in range(1, n+1):
        d[f"{well}_lag{i}"] = d[well].shift(i)
    return d.dropna().reset_index(drop=True)

def train_ann(df_feat, well, test_sz, h_layers):
    X = df_feat.drop(columns=[well,"Date"])
    y = df_feat[well]
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=test_sz,shuffle=False)
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    mdl = MLPRegressor(hidden_layer_sizes=h_layers,activation="relu",
                       solver="adam",max_iter=2000,early_stopping=True,
                       random_state=42)
    mdl.fit(Xtr,ytr)
    ytr_pred, yte_pred = mdl.predict(Xtr), mdl.predict(Xte)
    df_feat.loc[Xtr.index,"pred"] = ytr_pred
    df_feat.loc[Xte.index,"pred"] = yte_pred
    return mdl, sc, df_feat, {
        "R² train": round(r2_score(ytr,ytr_pred),4),
        "RMSE train": round(np.sqrt(mean_squared_error(ytr,ytr_pred)),4),
        "R² test":  round(r2_score(yte,yte_pred),4),
        "RMSE test": round(np.sqrt(mean_squared_error(yte,yte_pred)),4)
    }

def rec_forecast(model,sc,row,well,n_lags,h):
    out=[]; r=row.copy()
    for _ in range(h):
        for lg in range(n_lags,1,-1):
            r[f"{well}_lag{lg}"]=r[f"{well}_lag{lg-1}"]
        r[f"{well}_lag1"]=r["pred"]
        nxt=r["Date"]+pd.DateOffset(months=1)
        r.update({"Date":nxt,"Months":nxt.month,
                  "month_sin":np.sin(2*np.pi*nxt.month/12),
                  "month_cos":np.cos(2*np.pi*nxt.month/12)})
        X=sc.transform(r.drop(labels=[well,"Date"]).to_frame().T)
        pred=model.predict(X)[0]
        r[well]=r["pred"]=pred
        out.append({"Date":nxt,"Forecast":pred})
    return pd.DataFrame(out)

def ann_page():
    st.title("🔮 ANN Groundwater Prediction")
    df=load_gw(DATA_PATH)
    if df is None:
        st.error(f"CSV '{DATA_PATH}' not found in repo. "
                 "Add it or upload below.")
        up=st.file_uploader("Upload GW data CSV",type="csv")
        if up:
            Path(DATA_PATH).write_bytes(up.getvalue())
            st.success("File saved—reload.")
        return

    wells=[c for c in df.columns if c.startswith("W")]
    well   = st.sidebar.selectbox("Well",wells)
    lags   = st.sidebar.slider("Lag steps",1,24,12)
    split  = st.sidebar.slider("Test size",0.1,0.5,0.2,step=0.05)
    h_text = st.sidebar.text_input("Hidden layers","64,32")
    hidd   = tuple(int(x) for x in h_text.split(",") if x.strip())
    horiz  = st.sidebar.number_input("Forecast horizon (m)",1,60,12)

    feats  = lag_feats(df[["Date","Months","month_sin","month_cos",well]],well,lags)
    mdl,sc,dfp,met=train_ann(feats,well,split,hidd)

    st.json(met)
    st.plotly_chart(px.line(dfp,x="Date",y=[well,"pred"],
                            labels={"value":"Level (m)","variable":"Legend"},
                            title=f"{well} – actual vs. fit"),
                    use_container_width=True)

    fut=rec_forecast(mdl,sc,dfp.tail(1).iloc[0],well,lags,horiz)
    st.plotly_chart(px.line(pd.concat([dfp[["Date",well]].rename(columns={well:"Level"}),
                                       fut.rename(columns={"Forecast":"Level"})]),
                            x="Date",y="Level",
                            title=f"{well} – {horiz}-month forecast"),
                    use_container_width=True)

    st.download_button("Download forecast CSV",
                       fut.to_csv(index=False).encode(),
                       file_name=f"{well}_forecast_{datetime.today().date()}.csv",
                       mime="text/csv")

# ───────── page routing ─────────
page=st.sidebar.radio("Go to",[
    "Home","Well Map Viewer","📈 Groundwater Data",
    "📉 Groundwater Level Trends for Wells",
    "🔮 ANN Groundwater Prediction","📊 ARIMA Prediction"
])

# --- existing pages (unchanged code elided for brevity) ---
if page=="Home":
    st.title("Erbil Central Sub-Basin CSB Groundwater Data Analysis")
    st.markdown("Explore well data, visualize maps, analyse trends & forecasts.")

elif page=="Well Map Viewer":
    # … (all your original Map Viewer code unchanged) …
    pass

elif page=="📈 Groundwater Data":
    # … (original cleaning/editor UI unchanged) …
    pass

elif page=="📉 Groundwater Level Trends for Wells":
    st.title("📉 Groundwater Level Trends for Wells")
    st.info("Trend analysis page is under construction.")

elif page=="🔮 ANN Groundwater Prediction":
    ann_page()

elif page=="📊 ARIMA Prediction":
    st.title("📊 ARIMA Prediction Models")
    st.markdown("Placeholder for future ARIMA forecasts.")
