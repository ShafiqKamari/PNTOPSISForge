
import io
import math
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet


# ============================================================
# PNTOPSISForge v1.0
# A Pythagorean Neutrosophic Decision Support Engine
# ============================================================

st.set_page_config(page_title="PNTOPSISForge v1.0", layout="wide")
st.title("PNTOPSISForge v1.0")
st.caption("A Pythagorean Neutrosophic TOPSIS Decision Support System")


# ------------------------------------------------------------
# Linguistic Tables
# ------------------------------------------------------------
PNS_TABLES: Dict[int, Dict[int, Tuple[float, float, float]]] = {
    5: {1:(0.10,0.85,0.90),2:(0.30,0.65,0.70),3:(0.50,0.45,0.45),4:(0.70,0.25,0.20),5:(0.90,0.10,0.05)},
    7: {1:(0.10,0.80,0.90),2:(0.20,0.70,0.80),3:(0.35,0.60,0.60),4:(0.50,0.40,0.45),
        5:(0.65,0.30,0.25),6:(0.80,0.20,0.15),7:(0.90,0.10,0.10)},
    9: {1:(0.05,0.90,0.95),2:(0.10,0.85,0.90),3:(0.20,0.80,0.75),4:(0.35,0.65,0.60),
        5:(0.50,0.50,0.45),6:(0.65,0.35,0.30),7:(0.80,0.25,0.20),
        8:(0.90,0.15,0.10),9:(0.95,0.05,0.05)},
    11:{1:(0.05,0.90,0.95),2:(0.10,0.80,0.85),3:(0.20,0.70,0.75),4:(0.30,0.60,0.65),
        5:(0.40,0.50,0.55),6:(0.50,0.45,0.45),7:(0.60,0.40,0.35),
        8:(0.70,0.30,0.25),9:(0.80,0.20,0.15),10:(0.90,0.15,0.10),11:(0.95,0.05,0.05)}
}


# ------------------------------------------------------------
# Distance Functions
# ------------------------------------------------------------
def dist_euclidean(t,x,e,tp,xp,ep):
    n=len(t)
    return math.sqrt((1/(3*n))*np.sum((t-tp)**2+(x-xp)**2+(e-ep)**2))

def dist_hamming(t,x,e,tp,xp,ep):
    n=len(t)
    return (1/(3*n))*np.sum(np.abs(t-tp)+np.abs(x-xp)+np.abs(e-ep))

def dist_fiq(t,x,e,tp,xp,ep):
    n=len(t)
    d_tau=np.abs(t-tp); d_xi=np.abs(x-xp); d_eta=np.abs(e-ep)

    w_tau=1-(x*xp)
    w_eta=1-(x*xp)
    w_xi=1+(np.abs(t-ep)+np.abs(e-tp))/2

    p_tau=1+(x+xp)/2
    p_eta=1+(x+xp)/2
    p_xi=2-np.abs(t-ep)
    p=2-(x*xp)

    inner=w_tau*(d_tau**p_tau)+w_xi*(d_xi**p_xi)+w_eta*(d_eta**p_eta)
    return (1/(3*n))*np.sum(inner**(1/p))

DIST_FUNCS={"PN-Euclidean":dist_euclidean,"PN-Hamming":dist_hamming,"FIQ":dist_fiq}


# ------------------------------------------------------------
# Sidebar Settings
# ------------------------------------------------------------
st.sidebar.header("Settings")
scale=st.sidebar.selectbox("Linguistic Scale",[5,7,9,11],index=3)
distance_choice=st.sidebar.selectbox("Primary Distance",list(DIST_FUNCS.keys()))
sensitivity_toggle=st.sidebar.checkbox("Enable Sensitivity Analysis")
weight_mode=st.sidebar.radio("Weights",["Equal","Manual"])


# ------------------------------------------------------------
# Data Upload
# ------------------------------------------------------------
uploaded=st.file_uploader("Upload Excel/CSV Decision Matrix")
if not uploaded:
    st.stop()

df=pd.read_excel(uploaded) if uploaded.name.endswith("xlsx") else pd.read_csv(uploaded)

alt_names=df.iloc[:,0].astype(str).tolist()
crit_names=list(df.columns[1:])
crisp=df.iloc[:,1:].astype(int).values
m,n=crisp.shape


# ------------------------------------------------------------
# Convert Crisp -> PNS
# ------------------------------------------------------------
tau=np.zeros((m,n)); xi=np.zeros((m,n)); eta=np.zeros((m,n))
for i in range(m):
    for j in range(n):
        tau[i,j],xi[i,j],eta[i,j]=PNS_TABLES[scale][crisp[i,j]]


# ------------------------------------------------------------
# Criterion Types (Assume B/C Editable)
# ------------------------------------------------------------
crit_types=["B"]*n
type_df=pd.DataFrame([crit_types],columns=crit_names,index=["Type"])
edited=st.data_editor(type_df)
crit_types=[edited.iloc[0,j] for j in range(n)]


# ------------------------------------------------------------
# Weights
# ------------------------------------------------------------
if weight_mode=="Equal":
    w=np.ones(n)/n
else:
    w_df=pd.DataFrame([[1/n]*n],columns=crit_names,index=["w"])
    w_edit=st.data_editor(w_df)
    w=np.array([float(w_edit.iloc[0,j]) for j in range(n)])
    if abs(np.sum(w)-1)>1e-3:
        st.warning("Sum of weights is not equal to 1.")


# ------------------------------------------------------------
# Normalization
# ------------------------------------------------------------
tau_n=tau.copy(); xi_n=xi.copy(); eta_n=eta.copy()
for j in range(n):
    if crit_types[j]=="B":
        tau_n[:,j]/=np.max(tau[:,j])
        xi_n[:,j]/=np.max(xi[:,j])
        eta_n[:,j]/=np.max(eta[:,j])
    else:
        tau_n[:,j]=np.min(tau[:,j])/tau[:,j]
        xi_n[:,j]=np.min(xi[:,j])/xi[:,j]
        eta_n[:,j]=np.min(eta[:,j])/eta[:,j]

tau_w=tau_n*w; xi_w=xi_n*w; eta_w=eta_n*w

tau_p=np.max(tau_w,axis=0); xi_p=np.min(xi_w,axis=0); eta_p=np.min(eta_w,axis=0)
tau_nis=np.min(tau_w,axis=0); xi_nis=np.max(xi_w,axis=0); eta_nis=np.max(eta_w,axis=0)


# ------------------------------------------------------------
# Ranking Function
# ------------------------------------------------------------
def compute_ranking(dist_name):
    dist_fn=DIST_FUNCS[dist_name]
    S_plus=[]; S_minus=[]
    for i in range(m):
        S_plus.append(dist_fn(tau_w[i],xi_w[i],eta_w[i],tau_p,xi_p,eta_p))
        S_minus.append(dist_fn(tau_w[i],xi_w[i],eta_w[i],tau_nis,xi_nis,eta_nis))
    S_plus=np.array(S_plus); S_minus=np.array(S_minus)
    Pi=S_minus/(S_plus+S_minus)
    res=pd.DataFrame({"P_i":Pi},index=alt_names)
    res["Rank"]=(-res["P_i"]).rank(method="dense").astype(int)
    return res.sort_values("Rank")


# ------------------------------------------------------------
# Primary Ranking + Timing
# ------------------------------------------------------------
start=time.time()
primary_result=compute_ranking(distance_choice)
elapsed=time.time()-start

st.subheader("Primary Ranking Result")
st.dataframe(primary_result)
st.info(f"Computation Time: {elapsed:.6f} seconds")


# ------------------------------------------------------------
# Sensitivity Analysis
# ------------------------------------------------------------
if sensitivity_toggle:
    st.subheader("Sensitivity Analysis")
    compare=pd.DataFrame(index=alt_names)
    for name in DIST_FUNCS.keys():
        compare[name]=compute_ranking(name)["P_i"]
    st.dataframe(compare)
    st.bar_chart(compare)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
best=primary_result.index[0]
best_val=primary_result.iloc[0]["P_i"]
st.success(f"{best} is the best alternative with P_i = {best_val:.6f}")
