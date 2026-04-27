import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(layout="wide")

st.title("📊 Economic Stability & Crisis Dashboard")

# =========================
# Safe Load Function
# =========================
def load_csv(file, **kwargs):
    if os.path.exists(file):
        return pd.read_csv(file, **kwargs)
    else:
        st.warning(f"{file} not found")
        return None

# =========================
# Load Data (SAFE)
# =========================
df = load_csv("processed_data.csv", parse_dates=['Date'], index_col='Date')
probs = load_csv("xgb_probs.csv")
imp = load_csv("feature_importance.csv")
corr = load_csv("correlation_matrix.csv", index_col=0)

# =========================
# Sidebar
# =========================
section = st.sidebar.radio("Select Section", [
    "Overview",
    "Stationarity (ADF)",
    "Cointegration (Johansen)",
    "VECM",
    "IRF",
    "FEVD",
    "Economic Stability",
    "Shock Detection",
    "Correlation Heatmap",
    "Crisis Prediction (ML)"
])

# =========================
# 1. Overview
# =========================
if section == "Overview":
    st.subheader("Dataset Overview")

    if df is not None:
        st.dataframe(df.head())
        st.write("### Summary Statistics")
        st.write(df.describe())
    else:
        st.error("Dataset not available")

# =========================
# 2. ADF
# =========================
elif section == "Stationarity (ADF)":
    st.subheader("ADF Test Results")

    st.write("""
    - Used to test stationarity
    - Variables became stationary after differencing
    - Suitable for VECM modeling
    """)

# =========================
# 3. Johansen
# =========================
elif section == "Cointegration (Johansen)":
    st.subheader("Johansen Cointegration Test")

    st.write("""
    - Long-run relationships exist between variables
    - Justifies using VECM instead of VAR
    """)

# =========================
# 4. VECM
# =========================
elif section == "VECM":
    st.subheader("Vector Error Correction Model")

    st.write("""
    - Captures short-run and long-run dynamics
    - Adjustment toward equilibrium confirmed
    """)

# =========================
# 5. IRF
# =========================
elif section == "IRF":
    st.subheader("Impulse Response Function")

    st.write("""
    - Shows response of variables to shocks
    - Some effects are temporary, others persistent
    """)

# =========================
# 6. FEVD
# =========================
elif section == "FEVD":
    st.subheader("Forecast Error Variance Decomposition")

    st.write("""
    - Identifies most influential variables
    - Inflation and GDP are major drivers
    """)

# =========================
# 7. Stability
# =========================
elif section == "Economic Stability":

    if df is not None and "Economic_Stability_Index" in df.columns:
        fig = px.line(df,
                      y="Economic_Stability_Index",
                      title="Economic Stability Over Time")

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Economic Stability data not available")

# =========================
# 8. Shock Detection
# =========================
elif section == "Shock Detection":

    if df is not None and "Shock" in df.columns:
        fig = px.scatter(df,
                         y="Shock",
                         title="Shock Occurrence")

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Shock data not available")

# =========================
# 9. Correlation Heatmap
# =========================
elif section == "Correlation Heatmap":

    st.subheader("Correlation Matrix")

    if corr is not None:
        fig = px.imshow(corr,
                        text_auto=True,
                        aspect="auto",
                        title="Feature Correlation Heatmap")

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Correlation data not available")

# =========================
# 10. ML Prediction
# =========================
elif section == "Crisis Prediction (ML)":

    st.subheader("Model Performance")

    # لو عندك auc ضيفيه هنا
    if 'auc' in globals():
        fig_auc = px.bar(auc,
                         x="Model",
                         y="AUC",
                         title="Model Comparison (AUC)")
        st.plotly_chart(fig_auc, use_container_width=True)
    else:
        st.warning("AUC data not available")

    st.write("""
    - Random Forest outperformed XGBoost
    - Better ability to detect shocks
    """)

    # =====================
    # Feature Importance
    # =====================
    st.subheader("Feature Importance")

    if imp is not None:
        fig_imp = px.bar(imp.head(10),
                         x='Importance',
                         y='Feature',
                         orientation='h',
                         title="Top Important Features")

        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.warning("Feature importance not available")

    # =====================
    # Crisis Probability
    # =====================
    st.subheader("Early Warning Signal")

    if probs is not None and 'prob' in probs.columns:
        fig_prob = go.Figure()
        fig_prob.add_trace(go.Scatter(
            y=probs['prob'],
            name="Crisis Probability"
        ))

        fig_prob.add_hline(y=0.5, line_dash="dash")

        st.plotly_chart(fig_prob, use_container_width=True)
    else:
        st.warning("Probability data not available")

    st.write("""
    - Values above 0.5 indicate high crisis risk
    - Used as early warning system
    """)
