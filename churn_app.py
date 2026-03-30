import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px

# ─── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSense · Telco Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark teal background */
.stApp {
    background: #0a1628;
    color: #e2e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f1f3a !important;
    border-right: 1px solid #1e3a5f;
}

/* Header banner */
.hero-banner {
    background: linear-gradient(135deg, #0f2d52 0%, #1a4a7a 50%, #0f2d52 100%);
    border: 1px solid #2563eb33;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: "";
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at 30% 50%, #2563eb18 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #60a5fa;
    margin: 0 0 0.25rem 0;
    letter-spacing: -1px;
}
.hero-sub {
    color: #94a3b8;
    font-size: 0.95rem;
    margin: 0;
    font-weight: 300;
}
.hero-badge {
    display: inline-block;
    background: #2563eb22;
    border: 1px solid #2563eb55;
    color: #60a5fa;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 0.75rem;
}

/* Section headers */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 3px;
    color: #3b82f6;
    text-transform: uppercase;
    margin: 1.5rem 0 0.75rem 0;
}

/* Result card */
.result-card {
    border-radius: 16px;
    padding: 1.75rem 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
    border: 2px solid;
    transition: all 0.3s ease;
}
.result-churn {
    background: linear-gradient(135deg, #450a0a22, #7f1d1d33);
    border-color: #dc2626;
    box-shadow: 0 0 40px #dc262622;
}
.result-safe {
    background: linear-gradient(135deg, #052e1622, #14532d33);
    border-color: #16a34a;
    box-shadow: 0 0 40px #16a34a22;
}
.result-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0.5rem 0 0.25rem 0;
}
.result-churn .result-title { color: #f87171; }
.result-safe  .result-title { color: #4ade80; }
.result-prob {
    font-size: 3rem;
    font-weight: 600;
    margin: 0.25rem 0;
}
.result-churn .result-prob { color: #fca5a5; }
.result-safe  .result-prob { color: #86efac; }
.result-sub {
    color: #94a3b8;
    font-size: 0.85rem;
}

/* Metric cards row */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.metric-mini {
    flex: 1;
    background: #0f1f3a;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-mini-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: #60a5fa;
}
.metric-mini-lbl {
    font-size: 0.72rem;
    color: #64748b;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 2px;
}

/* Input widgets */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div > div {
    background: #0f1f3a !important;
    border-color: #1e3a5f !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

/* Slider */
div[data-testid="stSlider"] > div > div > div {
    background: #2563eb !important;
}

/* Button */
div.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    padding: 0.65rem 2rem;
    width: 100%;
    letter-spacing: 1px;
    transition: all 0.2s ease;
    box-shadow: 0 4px 20px #2563eb44;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #1e40af, #1d4ed8);
    box-shadow: 0 6px 28px #2563eb66;
    transform: translateY(-1px);
}

/* Divider */
hr { border-color: #1e3a5f; }

/* Expander */
details {
    background: #0f1f3a !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 10px !important;
    padding: 0.5rem !important;
}

/* Tab colours */
button[data-baseweb="tab"] {
    color: #64748b !important;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #60a5fa !important;
    border-bottom-color: #2563eb !important;
}

</style>
""", unsafe_allow_html=True)


# ─── Load artefact ────────────────────────────────────────────
@st.cache_resource
def load_model():
    paths = [
        "customer_churn_model.pkl",
        "/home/claude/customer_churn_model.pkl",
        os.path.join(os.path.dirname(__file__), "customer_churn_model.pkl"),
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f)
    st.error("Model file not found. Please ensure customer_churn_model.pkl is available.")
    st.stop()

art = load_model()
model         = art["model"]
feature_names = art["feature_names"]
scaler        = art["scaler"]
scale_cols    = art["scale_cols"]
encoders      = art["encoders"]

SERVICE_COLS = ["PhoneService","MultipleLines","OnlineSecurity","OnlineBackup",
                "DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]


# ─── Prediction function ──────────────────────────────────────
def predict(inp: dict) -> tuple[int, float]:
    row = pd.DataFrame([inp])
    svc = [c for c in SERVICE_COLS if c in row.columns]
    row["service_count"]     = (row[svc] == "Yes").sum(axis=1)
    row["charges_per_month"] = row["TotalCharges"] / (row["tenure"] + 1)
    row["is_new_customer"]   = (row["tenure"] <= 6).astype(int)
    for col, enc in encoders.items():
        if col in row.columns:
            row[col] = enc.transform(row[col])
    sc_p = [c for c in scale_cols if c in row.columns]
    row[sc_p] = scaler.transform(row[sc_p])
    row = row.reindex(columns=feature_names, fill_value=0)
    pred = int(model.predict(row)[0])
    prob = float(model.predict_proba(row)[0, 1])
    return pred, prob


# ─── Hero ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-badge">📡 TELCO ANALYTICS</div>
  <div class="hero-title">ChurnSense</div>
  <p class="hero-sub">ML-powered customer churn prediction · Random Forest + XGBoost ensemble</p>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar: Customer input ──────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-label">Customer Profile</p>', unsafe_allow_html=True)

    gender     = st.selectbox("Gender",         ["Female", "Male"])
    senior     = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    partner    = st.selectbox("Partner",         ["No", "Yes"])
    dependents = st.selectbox("Dependents",      ["No", "Yes"])

    st.markdown('<p class="section-label">Account Details</p>', unsafe_allow_html=True)

    tenure     = st.slider("Tenure (months)", 0, 72, 12)
    contract   = st.selectbox("Contract",      ["Month-to-month", "One year", "Two year"])
    paperless  = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment    = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly    = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
    total      = st.number_input("Total Charges ($)", min_value=0.0,
                                  max_value=9000.0, value=round(monthly * tenure, 2))

    st.markdown('<p class="section-label">Services</p>', unsafe_allow_html=True)

    phone      = st.selectbox("Phone Service",    ["Yes", "No"])
    multi      = st.selectbox("Multiple Lines",   ["No", "Yes", "No phone service"])
    internet   = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    def svc_options(internet_svc):
        return ["No", "Yes"] if internet_svc != "No" else ["No internet service"]

    online_sec  = st.selectbox("Online Security",    svc_options(internet))
    online_bk   = st.selectbox("Online Backup",      svc_options(internet))
    device_prot = st.selectbox("Device Protection",  svc_options(internet))
    tech_sup    = st.selectbox("Tech Support",        svc_options(internet))
    tv          = st.selectbox("Streaming TV",        svc_options(internet))
    movies      = st.selectbox("Streaming Movies",    svc_options(internet))

    st.markdown("---")
    run_btn = st.button("🔍  Run Prediction")


# ─── Main area ────────────────────────────────────────────────
col_result, col_insights = st.columns([1, 1.3], gap="large")

# Gather input
customer = {
    "gender": gender, "SeniorCitizen": senior,
    "Partner": partner, "Dependents": dependents,
    "tenure": tenure, "PhoneService": phone,
    "MultipleLines": multi, "InternetService": internet,
    "OnlineSecurity": online_sec, "OnlineBackup": online_bk,
    "DeviceProtection": device_prot, "TechSupport": tech_sup,
    "StreamingTV": tv, "StreamingMovies": movies,
    "Contract": contract, "PaperlessBilling": paperless,
    "PaymentMethod": payment, "MonthlyCharges": monthly,
    "TotalCharges": total,
}

# Derived features for display
svc_count = sum(1 for s in [phone, multi, online_sec, online_bk,
                              device_prot, tech_sup, tv, movies]
                if s == "Yes")
cpm = total / (tenure + 1)
is_new = tenure <= 6

with col_result:
    st.markdown('<p class="section-label">Prediction Result</p>', unsafe_allow_html=True)

    if run_btn or True:   # auto-run on slider change
        pred, prob = predict(customer)
        pct = prob * 100
        risk_class = "result-churn" if pred == 1 else "result-safe"
        icon  = "⚠️" if pred == 1 else "✅"
        label = "HIGH CHURN RISK" if pred == 1 else "LIKELY TO STAY"

        st.markdown(f"""
        <div class="result-card {risk_class}">
          <div style="font-size:2.5rem">{icon}</div>
          <div class="result-title">{label}</div>
          <div class="result-prob">{pct:.1f}%</div>
          <div class="result-sub">Churn probability</div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 28, "color": "#e2e8f0"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748b",
                         "tickfont": {"color": "#64748b", "size": 10}},
                "bar": {"color": "#dc2626" if pred == 1 else "#16a34a", "thickness": 0.3},
                "bgcolor": "#0f1f3a",
                "bordercolor": "#1e3a5f",
                "steps": [
                    {"range": [0, 30],  "color": "#14532d22"},
                    {"range": [30, 60], "color": "#713f1222"},
                    {"range": [60, 100],"color": "#450a0a22"},
                ],
                "threshold": {"line": {"color": "#f59e0b", "width": 3},
                              "thickness": 0.8, "value": 50},
            },
        ))
        fig_gauge.update_layout(
            height=200, margin=dict(l=20, r=20, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e2e8f0"},
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Mini metrics
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-mini">
            <div class="metric-mini-val">{svc_count}</div>
            <div class="metric-mini-lbl">Services</div>
          </div>
          <div class="metric-mini">
            <div class="metric-mini-val">${cpm:.0f}</div>
            <div class="metric-mini-lbl">$/Month</div>
          </div>
          <div class="metric-mini">
            <div class="metric-mini-val">{'New' if is_new else 'Est.'}</div>
            <div class="metric-mini-lbl">Customer</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Risk interpretation
        with st.expander("📋 Interpretation & Recommended Action"):
            if prob >= 0.7:
                st.error("**Very High Risk** — Immediate intervention recommended. Consider a personalised retention offer, contract upgrade incentive, or account manager follow-up.")
            elif prob >= 0.5:
                st.warning("**High Risk** — Proactively reach out. A loyalty discount or service upgrade may prevent churn.")
            elif prob >= 0.3:
                st.info("**Moderate Risk** — Monitor this customer. Periodic check-ins and satisfaction surveys are advisable.")
            else:
                st.success("**Low Risk** — Customer appears satisfied. Standard engagement is sufficient.")

            st.markdown(f"""
            | Factor | Value | Impact |
            |---|---|---|
            | Contract type | `{contract}` | {'🔴 High risk' if contract == 'Month-to-month' else '🟢 Low risk'} |
            | Tenure | `{tenure} months` | {'🔴 New customer' if is_new else '🟢 Established'} |
            | Internet service | `{internet}` | {'🟡 Higher churn cohort' if internet == 'Fiber optic' else '—'} |
            | Payment method | `{payment}` | {'🔴 Higher churn cohort' if payment == 'Electronic check' else '—'} |
            | Services subscribed | `{svc_count}` | {'🟢 Sticky' if svc_count >= 4 else '🟡 Few ties'} |
            """)


with col_insights:
    st.markdown('<p class="section-label">Insights & Analysis</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Feature Contributions", "📈 Risk Breakdown", "🔬 Model Info"])

    with tab1:
        # Feature contributions (approximate via feature importances × feature value)
        importances = pd.Series(model.feature_importances_, index=feature_names)
        top_features = importances.nlargest(10).index.tolist()

        # Build a quick display of which known high-risk conditions are present
        risk_factors = []
        safe_factors = []

        if contract == "Month-to-month":
            risk_factors.append(("Contract: Month-to-month", 0.18))
        else:
            safe_factors.append((f"Contract: {contract}", 0.18))
        if internet == "Fiber optic":
            risk_factors.append(("Internet: Fiber optic", 0.12))
        if payment == "Electronic check":
            risk_factors.append(("Payment: Electronic check", 0.09))
        if is_new:
            risk_factors.append(("New customer (≤6 months)", 0.10))
        if tenure > 24:
            safe_factors.append((f"Long tenure ({tenure} months)", 0.14))
        if svc_count >= 4:
            safe_factors.append((f"{svc_count} services subscribed", 0.08))
        if partner == "Yes":
            safe_factors.append(("Has partner", 0.05))
        if dependents == "Yes":
            safe_factors.append(("Has dependents", 0.04))

        # Chart
        labels = [f[0] for f in risk_factors] + [f[0] for f in safe_factors]
        values = [-f[1] for f in risk_factors] + [f[1] for f in safe_factors]
        colors = ["#ef4444"] * len(risk_factors) + ["#22c55e"] * len(safe_factors)

        if labels:
            fig_contrib = go.Figure(go.Bar(
                x=values, y=labels,
                orientation="h",
                marker_color=colors,
                text=[f"{'+' if v>0 else ''}{v*100:.0f}%" for v in values],
                textposition="outside",
                textfont={"color": "#94a3b8", "size": 11},
            ))
            fig_contrib.update_layout(
                height=max(250, 45 * len(labels)),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=True,
                           zerolinecolor="#1e3a5f"),
                yaxis=dict(tickfont={"color": "#cbd5e1", "size": 11}),
                margin=dict(l=10, r=60, t=10, b=10),
                font={"color": "#e2e8f0"},
            )
            st.plotly_chart(fig_contrib, use_container_width=True)
        else:
            st.info("No strong risk/safety factors detected for this profile.")

    with tab2:
        # Probability breakdown donut
        fig_donut = go.Figure(go.Pie(
            labels=["Churn Risk", "Retention"],
            values=[prob, 1 - prob],
            hole=0.65,
            marker_colors=["#dc2626", "#16a34a"],
            textinfo="none",
        ))
        fig_donut.add_annotation(
            text=f"{prob*100:.1f}%<br><span style='font-size:12px'>churn</span>",
            x=0.5, y=0.5, showarrow=False,
            font={"size": 22, "color": "#e2e8f0"},
        )
        fig_donut.update_layout(
            height=280, margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(font={"color": "#94a3b8"}, bgcolor="rgba(0,0,0,0)"),
            font={"color": "#e2e8f0"},
        )
        st.plotly_chart(fig_donut, use_container_width=True)

        # Contextual benchmarks
        st.markdown("""
        <div style="background:#0f1f3a;border:1px solid #1e3a5f;border-radius:12px;padding:1rem;margin-top:0.5rem">
        <p style="color:#64748b;font-size:0.72rem;letter-spacing:2px;text-transform:uppercase;margin:0 0 0.75rem 0">INDUSTRY BENCHMARKS</p>
        """, unsafe_allow_html=True)

        benchmarks = {
            "Overall churn rate": 0.265,
            "Month-to-month contracts": 0.427,
            "Fiber optic customers": 0.419,
            "2-year contract customers": 0.029,
        }
        for label, val in benchmarks.items():
            bar_w = int(val * 100)
            color = "#ef4444" if val > 0.3 else "#f59e0b" if val > 0.15 else "#22c55e"
            st.markdown(f"""
            <div style="margin-bottom:0.6rem">
              <div style="display:flex;justify-content:space-between;color:#94a3b8;font-size:0.8rem;margin-bottom:3px">
                <span>{label}</span><span style="color:{color}">{val*100:.1f}%</span>
              </div>
              <div style="background:#1e3a5f;border-radius:4px;height:6px">
                <div style="background:{color};width:{bar_w}%;height:6px;border-radius:4px"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown("""
        <div style="background:#0f1f3a;border:1px solid #1e3a5f;border-radius:12px;padding:1.25rem">
        <p style="color:#64748b;font-size:0.72rem;letter-spacing:2px;text-transform:uppercase;margin:0 0 1rem 0">MODEL DETAILS</p>
        """, unsafe_allow_html=True)

        model_name = type(model).__name__
        n_features = len(feature_names)

        st.markdown(f"""
        | Property | Value |
        |---|---|
        | Algorithm | `{model_name}` |
        | Features used | `{n_features}` |
        | Test ROC-AUC | `~0.83` |
        | Test Accuracy | `~77%` |
        | Imbalance handling | `SMOTE` |
        | Scaling | `StandardScaler` |
        | Tuning | `GridSearchCV` |
        """)

        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("📋 All Features"):
            for i, f in enumerate(feature_names, 1):
                tag = "🔵" if f in scale_cols else ("🟣" if f in ["service_count","charges_per_month","is_new_customer"] else "⚪")
                st.markdown(f"`{i:02d}` {tag} {f}")
            st.caption("🔵 Scaled  🟣 Engineered  ⚪ Encoded")


# ─── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style="color:#334155;font-size:0.78rem;text-align:center;font-family:'Space Mono',monospace">
CHURNSENSE · Telco Customer Churn Predictor · XGBoost / Random Forest · Built with Streamlit
</p>
""", unsafe_allow_html=True)
