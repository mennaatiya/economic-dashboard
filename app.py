import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Economic Stability Dashboard",
    layout="wide"
)

# =========================
# 🎯 TITLE
# =========================
st.title("📊 Economic Stability & Crisis Prediction Dashboard")
st.markdown("End-to-End Machine Learning + Time Series Project")

# =========================
# 📂 LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("processed_data.csv", index_col=0, parse_dates=True)
    shock = pd.read_csv("shock_data.csv", index_col=0, parse_dates=True)
    importance = pd.read_csv("importance.csv")
    model_comp = pd.read_csv("model_comparison.csv")
    probs = pd.read_csv("probs.csv")
    return df, shock, importance, model_comp, probs

df, shock, importance, model_comp, probs = load_data()

# =========================
# 🧭 SIDEBAR NAVIGATION
# =========================
section = st.sidebar.radio("📌 Navigation", [
    "Overview",
    "Data Analysis",
    "Models Performance",
    "Feature Importance",
    "Forecast",
    "Crisis Prediction"
])

# =========================
# 🏁 OVERVIEW
# =========================
if section == "Overview":
    st.subheader("📌 Project Overview")

    st.markdown("""
    This project analyzes **Economic Stability** using:
    
    - Machine Learning Models
    - Time Series (ARIMA)
    - Lag-based Features
    - Crisis Prediction System
    
    🎯 Goal:
    - Predict future stability
    - Detect early warning signals for crises
    """)

# =========================
# 📈 DATA ANALYSIS
# =========================
elif section == "Data Analysis":
    st.subheader("📈 Economic Stability Over Time")

    target_col = df.columns[0]

    fig = px.line(df, y=target_col, title="Economic Stability Trend")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📊 Dataset Preview")
    st.dataframe(df.tail())

# =========================
# 🤖 MODELS PERFORMANCE
# =========================
elif section == "Models Performance":
    st.subheader("🤖 Model Comparison")

    # Detect if RMSE or AUC exists
    if "RMSE" in model_comp.columns:
        metric = "RMSE"
    else:
        metric = "AUC"

    fig = px.bar(
        model_comp,
        x="Model",
        y=metric,
        color="Model",
        title=f"Model Comparison ({metric})"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(model_comp)

    best_model = model_comp.sort_values(by=metric).iloc[0]
    st.success(f"🏆 Best Model: {best_model['Model']}")

# =========================
# 📊 FEATURE IMPORTANCE
# =========================
elif section == "Feature Importance":
    st.subheader("📊 Most Important Features")

    importance_sorted = importance.sort_values(by="Importance", ascending=False)

    top_n = st.slider("Select Top Features", 5, 20, 10)

    fig = px.bar(
        importance_sorted.head(top_n),
        x="Importance",
        y="Feature",
        orientation='h',
        title="Top Important Features"
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# 🔮 FORECAST
# =========================
elif section == "Forecast":
    st.subheader("🔮 Future Forecast")

    df_temp = df.copy()
    df_temp['t'] = range(len(df_temp))

    from sklearn.linear_model import LinearRegression

    X = df_temp[['t']]
    y = df_temp.iloc[:, 0]

    model = LinearRegression()
    model.fit(X, y)

    future_steps = st.slider("Forecast Steps", 5, 20, 10)

    future_t = pd.DataFrame({
        't': range(len(df_temp), len(df_temp) + future_steps)
    })

    future_pred = model.predict(future_t)

    future_dates = pd.date_range(
        start=df_temp.index[-1],
        periods=future_steps + 1,
        freq='QE'
    )[1:]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_temp.index,
        y=y,
        name="Actual",
        line=dict(width=3)
    ))

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_pred,
        name="Forecast",
        line=dict(dash="dash")
    ))

    st.plotly_chart(fig, use_container_width=True)

# =========================
# ⚠️ CRISIS PREDICTION
# =========================
elif section == "Crisis Prediction":
    st.subheader("⚠️ Early Warning System")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=probs.iloc[:, 0],
        mode='lines',
        name='Crisis Probability'
    ))

    fig.add_hline(
        y=0.5,
        line_dash="dash",
        annotation_text="Risk Threshold"
    )

    st.plotly_chart(fig, use_container_width=True)

    last_prob = probs.iloc[-1, 0]

    if last_prob > 0.5:
        st.error(f"🚨 High Crisis Risk ({last_prob:.2f})")
    else:
        st.success(f"✅ Stable ({last_prob:.2f})")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("👩‍💻 Developed for Graduation Project | Machine Learning & Econometrics")
