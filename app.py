import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Economic Stability Dashboard", layout="wide")

st.title("📊 Economic Stability & Crisis Prediction Dashboard")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("processed_data.csv", index_col=0, parse_dates=True)
shock = pd.read_csv("shock_data.csv", index_col=0, parse_dates=True)
importance = pd.read_csv("importance.csv")
model_comp = pd.read_csv("model_comparison.csv")
probs = pd.read_csv("probs.csv")

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Data",
    "🤖 Models",
    "⚠️ Crisis",
    "📊 Importance",
    "🔮 Forecast"
])

# =========================
# 📈 DATA TAB
# =========================
with tab1:
    st.subheader("Economic Stability Over Time")

    fig = px.line(df, y=df.columns[0])
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df.tail())

# =========================
# 🤖 MODELS TAB
# =========================
with tab2:
    st.subheader("Model Comparison")

    fig = px.bar(model_comp, x="Model", y="AUC")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(model_comp)

# =========================
# ⚠️ CRISIS TAB
# =========================
with tab3:
    st.subheader("Crisis Probability")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=probs.iloc[:,0],
        mode='lines',
        name='Probability'
    ))

    fig.add_hline(y=0.5, line_dash="dash")

    st.plotly_chart(fig, use_container_width=True)

# =========================
# 📊 IMPORTANCE TAB
# =========================
with tab4:
    st.subheader("Feature Importance")

    importance_sorted = importance.sort_values(by="Importance", ascending=False)

    fig = px.bar(
        importance_sorted.head(15),
        x="Importance",
        y="Feature",
        orientation='h'
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# 🔮 FORECAST TAB
# =========================
with tab5:
    st.subheader("Future Forecast")

    df['t'] = range(len(df))

    from sklearn.linear_model import LinearRegression

    X = df[['t']]
    y = df.iloc[:,0]

    model = LinearRegression()
    model.fit(X, y)

    future_steps = 10
    future_t = pd.DataFrame({'t': range(len(df), len(df)+future_steps)})

    future_pred = model.predict(future_t)

    future_dates = pd.date_range(
        start=df.index[-1],
        periods=future_steps+1,
        freq='QE'
    )[1:]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=y,
        name="Actual"
    ))

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_pred,
        name="Forecast",
        line=dict(dash="dash")
    ))

    st.plotly_chart(fig, use_container_width=True)
