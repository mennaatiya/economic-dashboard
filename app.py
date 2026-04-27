import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")

st.title("📊 Economic Stability & Crisis Dashboard")

# =========================
# Load Data
# =========================
df = pd.read_csv("processed_data.csv", parse_dates=['Date'], index_col='Date')
auc = pd.read_csv("shock_prediction_auc.csv")
probs = pd.read_csv("xgb_probs.csv")
imp = pd.read_csv("feature_importance.csv")
corr = pd.read_csv("correlation_matrix.csv", index_col=0)

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
    st.dataframe(df.head())

    st.write("### Summary Statistics")
    st.write(df.describe())

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

    fig = px.line(df,
                  y="Economic_Stability_Index",
                  title="Economic Stability Over Time")

    st.plotly_chart(fig, use_container_width=True)

    st.write("""
    - Stability fluctuates over time
    - Drops correspond to economic shocks
    """)

# =========================
# 8. Shock Detection
# =========================
elif section == "Shock Detection":

    if "Shock" in df.columns:
        fig = px.scatter(df,
                         y="Shock",
                         title="Shock Occurrence")

        st.plotly_chart(fig, use_container_width=True)

        st.write("""
        - Shocks detected using residual threshold
        - Represents abnormal economic behavior
        """)

# =========================
# 9. Correlation Heatmap
# =========================
elif section == "Correlation Heatmap":

    st.subheader("Correlation Matrix")

    fig = px.imshow(corr,
                    text_auto=True,
                    aspect="auto",
                    title="Feature Correlation Heatmap")

    st.plotly_chart(fig, use_container_width=True)

    st.write("""
    - Shows relationships between variables
    - High correlation may indicate multicollinearity
    """)

# =========================
# 10. ML Prediction
# =========================
elif section == "Crisis Prediction (ML)":

    st.subheader("Model Performance")

    st.dataframe(auc)

    fig_auc = px.bar(auc,
                     x="Model",
                     y="AUC",
                     title="Model Comparison (AUC)")

    st.plotly_chart(fig_auc, use_container_width=True)

    st.write("""
    - Random Forest outperformed XGBoost
    - Better ability to detect shocks
    """)

    # =====================
    # Feature Importance
    # =====================
    st.subheader("Feature Importance")

    fig_imp = px.bar(imp.head(10),
                     x='Importance',
                     y='Feature',
                     orientation='h',
                     title="Top Important Features")

    st.plotly_chart(fig_imp, use_container_width=True)

    st.write("""
    - Shows most influential variables in prediction
    """)

    # =====================
    # Crisis Probability
    # =====================
    st.subheader("Early Warning Signal")

    fig_prob = go.Figure()
    fig_prob.add_trace(go.Scatter(
        y=probs['prob'],
        name="Crisis Probability"
    ))

    fig_prob.add_hline(y=0.5, line_dash="dash")

    st.plotly_chart(fig_prob, use_container_width=True)

    st.write("""
    - Values above 0.5 indicate high crisis risk
    - Used as early warning system
    """)
