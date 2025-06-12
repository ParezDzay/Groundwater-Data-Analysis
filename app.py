# app.py  – Groundwater-data dashboard with depth-oriented ANN plots
# ──────────────────────────────────────────────────────────────────
# Pages
#   • Home
#   • Well Map Viewer
#   • 📈 Groundwater Data
#   • 📉 Groundwater Level Trends for Wells   (placeholder)
#   • 🔮 ANN Groundwater Prediction           (depth plot, y-axis reversed)
#   • 📊 ARIMA Prediction                     (placeholder)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os, base64, requests
from datetime import datetime
from pathlib import Path

# Extra libs for ANN
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics         import r2_score, mean_squared_error

# ───────────── CONFIG ─────────────
st.set_page_config(page_title="Well Data App", layout="wide")
st.sidebar.title("Navigation")

token    = st.secrets["github"]["token"]
username = st.secrets["github"]["username"]
repo     = st.secrets["github"]["repo"]
branch   = st.secrets["github"]["branch"]

file_path            = "Wells detailed data.csv"
gw_file_path         = "GW data.csv"
output_path          = "GW data (missing filled).csv"
cleaned_outlier_path = "GW data (missing filled).csv"
GW_CSV               = "GW data (missing filled).csv"   # ANN input

# ───────── GitHub push helper ─────────
def push_to_github(fp, msg):
    with open(fp, "r", encoding="utf-8") as f:
        content = f.read()
    payload = {
        "message": msg,
        "content": base64.b64encode(content.encode()).decode(),
        "branch":  branch
    }
    url  = f"https://api.github.com/repos/{username}/{repo}/contents/{os.path.basename(fp)}"
    hdrs = {"Authorization": f"token {token}",
            "Accept": "application/vnd.github+json"}
    sha  = requests.get(url, headers=hdrs).json().get("sha")
    if sha:
        payload["sha"] = sha
    r = requests.put(url, headers=hdrs, json=payload)
    return r.status_code in (200, 201)

# ─────── ANN HELPERS (unchanged logic) ───────
@st.cache_data(show_spinner=False)
def _load_gw_csv(path: str) -> pd.DataFrame | None:
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Year"].astype(str) + "-" +
                                df["Months"].astype(str) + "-01")
    df["month_sin"] = np.sin(2 * np.pi * df["Months"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Months"] / 12)
    return df.sort_values("Date").reset_index(drop=True)

def _add_lags(df: pd.DataFrame, well: str, n: int) -> pd.DataFrame:
    out = df.copy()
    for k in range(1, n + 1):
        out[f"{well}_lag{k}"] = out[well].shift(k)
    return out.dropna().reset_index(drop=True)

def _fit_ann(df: pd.DataFrame, well: str, tst: float, layers: tuple[int, ...]):
    X = df.drop(columns=[well, "Date"]);  y = df[well]
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=tst,shuffle=False)
    sc = StandardScaler(); Xtr = sc.fit_transform(Xtr); Xte = sc.transform(Xte)
    mdl = MLPRegressor(hidden_layer_sizes=layers,activation="relu",
                       solver="adam",max_iter=2000,early_stopping=True,
                       random_state=42).fit(Xtr,ytr)
    ytr_pred, yte_pred = mdl.predict(Xtr), mdl.predict(Xte)
    df.loc[Xtr.index,"pred"], df.loc[Xte.index,"pred"] = ytr_pred, yte_pred
    return mdl, sc, df, {
        "R² train":round(r2_score(ytr,ytr_pred),4),
        "RMSE train":round(np.sqrt(mean_squared_error(ytr,ytr_pred)),4),
        "R² test":round(r2_score(yte,yte_pred),4),
        "RMSE test":round(np.sqrt(mean_squared_error(yte,yte_pred)),4)
    }

def _recur_forecast(m,sc,row,well,lags,h):
    out=[]; r=row.copy()
    for _ in range(h):
        for k in range(lags,1,-1):
            r[f"{well}_lag{k}"]=r[f"{well}_lag{k-1}"]
        r[f"{well}_lag1"]=r["pred"]
        nxt = r["Date"] + pd.DateOffset(months=1)
        r.update({"Date":nxt,"Months":nxt.month,
                  "month_sin":np.sin(2*np.pi*nxt.month/12),
                  "month_cos":np.cos(2*np.pi*nxt.month/12)})
        X=sc.transform(r.drop(labels=[well,"Date"]).to_frame().T)
        val=m.predict(X)[0]
        r[well]=r["pred"]=val
        out.append({"Date":nxt,"Forecast":val})
    return pd.DataFrame(out)

# ─────── ANN PAGE (depth axis reversed) ───────
def ann_page():
    st.title("🔮 ANN Groundwater Prediction (Depth View)")

    df=_load_gw_csv(GW_CSV)
    if df is None:
        st.error(f"CSV “{GW_CSV}” not found.  Upload below ⬇️")
        up = st.file_uploader("Upload groundwater CSV",type="csv")
        if up:
            Path(GW_CSV).write_bytes(up.getvalue())
            st.success("File saved—reload.")
        return

    wells=[c for c in df.columns if c.startswith("W")]
    well   = st.sidebar.selectbox("Well",wells)
    lags   = st.sidebar.slider("Lag steps",1,24,12)
    frac   = st.sidebar.slider("Test size",0.1,0.5,0.2,step=0.05)
    hl_txt = st.sidebar.text_input("Hidden layers","64,32")
    layers = tuple(int(x) for x in hl_txt.split(",") if x.strip())
    horiz  = st.sidebar.number_input("Forecast horizon (months)",1,60,12)

    feat=_add_lags(df[["Date","Months","month_sin","month_cos",well]],well,lags)
    mdl,sc,dfp,met=_fit_ann(feat,well,frac,layers)

    st.subheader("Performance");  st.json(met)

    # Actual vs predicted (y axis reversed for depth)
    fig1 = px.line(dfp,x="Date",y=[well,"pred"],
                   labels={"value":"Water table depth (m)",
                           "variable":"Legend"},
                   title=f"{well} – fit")
    fig1.update_yaxes(autorange="reversed")   # invert axis
    st.plotly_chart(fig1,use_container_width=True)

    future=_recur_forecast(mdl,sc,dfp.tail(1).iloc[0],well,lags,horiz)
    fig2 = px.line(
        pd.concat([dfp[["Date",well]].rename(columns={well:"Depth"}),
                   future.rename(columns={"Forecast":"Depth"})]),
        x="Date",y="Depth",
        title=f"{well} – {horiz}-month forecast"
    )
    fig2.update_yaxes(autorange="reversed")   # invert axis
    st.plotly_chart(fig2,use_container_width=True)

    st.download_button("Download forecast CSV",
                       future.to_csv(index=False).encode(),
                       file_name=f"{well}_forecast_{datetime.today().date()}.csv",
                       mime="text/csv")

# ───────── OTHER PAGES (unchanged) ─────────
def well_map_viewer():
    if not os.path.exists(file_path):
        st.error("Well CSV file not found."); st.stop()
    df=pd.read_csv(file_path,on_bad_lines='skip')
    df.columns=[c.strip().replace('\n',' ').replace('\r','') for c in df.columns]
    df["Coordinate X"]=pd.to_numeric(df.get("Coordinate X"),errors="coerce")
    df["Coordinate Y"]=pd.to_numeric(df.get("Coordinate Y"),errors="coerce")
    df["Depth (m)"]   =pd.to_numeric(df.get("Depth (m)"),errors="coerce")
    df.rename(columns={"Coordinate X":"lat","Coordinate Y":"lon"},inplace=True)
    df=df.dropna(subset=["lat","lon"])

    st.title("Well Map Viewer")
    tab_d,tab_m,tab_f,tab_u=st.tabs(
        ["📊 Data Table","🗺️ Map View","🔍 Filters","⬆️ Upload CSV"])
    with tab_f:
        bas=st.multiselect("Select Basin(s):",df["Basin"].unique(),
                           default=df["Basin"].unique())
        sub=st.multiselect("Select Sub-District(s):",df["sub district"].unique(),
                           default=df["sub district"].unique())
        filt=df[df["Basin"].isin(bas)&df["sub district"].isin(sub)]
        st.success("Filters applied.")
    with tab_d: st.dataframe(filt)
    with tab_m:
        fig=px.scatter_mapbox(filt,lat="lat",lon="lon",color="Basin",
                              hover_name="Well Name",
                              hover_data={"Depth (m)":True},
                              zoom=10,height=600)
        fig.update_layout(mapbox_style="open-street-map",
                          margin=dict(r=0,l=0,t=0,b=0))
        st.plotly_chart(fig,use_container_width=True)
    with tab_u:
        up=st.file_uploader("Upload a CSV",type="csv")
        if up:
            new=pd.read_csv(up,on_bad_lines='skip')
            new.columns=[c.strip().replace('\n',' ').replace('\r','') for c in new.columns]
            st.dataframe(new)
            if st.button("Append Uploaded Data to Dataset"):
                new.to_csv(file_path,mode='a',header=False,index=False)
                push_to_github(file_path,"Appended new well data")
                st.success("Data uploaded and pushed to GitHub.")

def groundwater_data_page():
    if not os.path.exists(output_path):
        st.error("Processed groundwater data not found."); st.stop()
    gw=pd.read_csv(output_path)
    gw["Year"]=gw["Year"].astype(int); gw["Months"]=gw["Months"].astype(int)
    gw["Date"]=pd.to_datetime(gw["Year"].astype(str)+"-"+gw["Months"].astype(str)+"-01")

    st.title("📈 Groundwater Data Over 20 Years")
    t_raw,t_edit,t_clean,t_out,t_save=st.tabs(
        ["📉 Data with Missing","✏️ Edit Raw Data","✅ Cleaned Data",
         "⚠️ Outlier %","🧹 Clean & Save"])
    wells=[c for c in gw.columns if c not in ("Year","Months","Date")]

    with t_raw: st.dataframe(gw)
    with t_edit:
        ed=st.data_editor(gw,num_rows="dynamic",use_container_width=True)
        if st.button("Save Edited Data and Push to GitHub"):
            ed.to_csv(gw_file_path,index=False)
            if push_to_github(gw_file_path,f"Edited on {datetime.now()}"):
                st.success("Data saved & pushed."); 
            else: st.error("GitHub push failed.")
    with t_clean: st.dataframe(gw)
    with t_out:
        def pct(df):
            out={}
            for w in wells:
                s=df[w].dropna()
                q1,q3=s.quantile(0.25),s.quantile(0.75)
                iqr=q3-q1; lb,ub=q1-1.5*iqr,q3+1.5*iqr
                out[w]=round(((s<lb)|(s>ub)).mean()*100,2)
            return out
        before=pct(gw); clean=gw.copy()
        for w in wells:
            s=clean[w]; q1,q3=s.quantile(0.25),s.quantile(0.75)
            iqr=q3-q1; lb,ub=q1-1.5*iqr,q3+1.5*iqr
            clean.loc[(s<lb)|(s>ub),w]=np.nan
            clean[w]=clean[w].interpolate(method='linear',limit_direction='both')
        after=pct(clean)
        st.dataframe(pd.DataFrame({"Before %":before,"After %":after}))
    with t_save:
        st.dataframe(clean)
        if st.button("Save Cleaned Data to GitHub"):
            clean.to_csv(cleaned_outlier_path,index=False)
            if push_to_github(cleaned_outlier_path,"Cleaned groundwater data"):
                st.success("Pushed to GitHub.")
            else: st.error("Push failed.")

# ─────── ROUTER ───────
page=st.sidebar.radio("Go to",
    ["Home","Well Map Viewer","📈 Groundwater Data",
     "📉 Groundwater Level Trends for Wells",
     "🔮 ANN Groundwater Prediction","📊 ARIMA Prediction"])

if page=="Home":
    st.title("Erbil Central Sub-Basin Groundwater Dashboard")
    st.markdown("Explore wells, clean datasets, and generate depth forecasts.")
elif page=="Well Map Viewer":
    well_map_viewer()
elif page=="📈 Groundwater Data":
    groundwater_data_page()
elif page=="📉 Groundwater Level Trends for Wells":
    st.title("📉 Groundwater Level Trends for Wells")
    st.info("Trend analysis page is under construction.")
elif page=="🔮 ANN Groundwater Prediction":
    ann_page()
elif page=="📊 ARIMA Prediction":
    st.title("📊 ARIMA Prediction Models")
    st.markdown("Placeholder for future ARIMA forecasts.")
