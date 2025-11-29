import streamlit as st

# Backend:
#   uvicorn app.main_api:app --reload
#
# Frontend (this UI):
#   streamlit run ui/app.py
st.set_page_config(page_title="AgroAI Risk Platform Dashboard", page_icon="🌾", layout="wide")

HIDE_APP_NAV = """
<style>
[data-testid="stSidebarNav"] > ul:nth-of-type(1) > li:nth-of-type(1) { display: none; }
</style>
"""

st.markdown(HIDE_APP_NAV, unsafe_allow_html=True)

# Global styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background: radial-gradient(circle at 20% 20%, #0b1120 0, #050814 40%, #020617 100%);
        color: #e5e7eb;
    }
    .app-container {
        max-width: 1100px;
        margin: 0 auto;
        padding-top: 32px;
        padding-bottom: 40px;
    }
    .hero {
        margin-bottom: 24px;
    }
    .hero-title {
        font-size: 38px;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
    }
    .hero-subtitle {
        font-size: 16px;
        opacity: 0.85;
    }
    .pill-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 11px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        background: rgba(34,197,94,0.12);
        color: #4ade80;
        border: 1px solid rgba(74,222,128,0.4);
    }
    .card {
        background: rgba(15,23,42,0.92);
        border-radius: 18px;
        padding: 22px 22px 18px 22px;
        box-shadow: 0 24px 60px rgba(0,0,0,0.5);
        border: 1px solid rgba(148,163,184,0.25);
        backdrop-filter: blur(18px);
        transition: transform 0.15s ease-out, box-shadow 0.15s ease-out, border-color 0.15s;
    }
    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 28px 80px rgba(0,0,0,0.65);
        border-color: rgba(94,234,212,0.5);
    }
    .card-header {
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .card-description {
        font-size: 14px;
        opacity: 0.8;
        margin-bottom: 12px;
    }
    .primary-button {
        display: inline-block;
        padding: 11px 20px;
        border-radius: 999px;
        background: linear-gradient(135deg, #22c55e, #16a34a);
        color: #020617;
        font-weight: 700;
        text-decoration: none;
        font-size: 15px;
        box-shadow: 0 14px 35px rgba(34,197,94,0.35);
        transition: all 0.15s ease-out;
    }
    .primary-button:hover { filter: brightness(1.06); transform: translateY(-1px); }
    .ghost-button {
        display: inline-block;
        padding: 9px 16px;
        border-radius: 12px;
        border: 1px solid rgba(148,163,184,0.6);
        color: #e5e7eb;
        font-size: 13px;
        text-decoration: none;
        transition: all 0.15s ease-out;
    }
    .ghost-button:hover { border-color: rgba(94,234,212,0.7); color: #a5f3fc; }
    .footer {
        margin-top: 28px;
        color: #94a3b8;
        font-size: 13px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-container">', unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
        <div class="pill-badge">DEV</div>
        <div class="hero-title">AgroAI Risk Platform</div>
        <div class="hero-subtitle">Unified Credit Scoring & Hybrid Fraud Detection Dashboard</div>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)
with col1:
    st.markdown(
        """
        <div class="card">
            <div class="card-header">Credit Scoring</div>
            <div class="card-description">Estimate probability of default, map to risk tiers, and support automated decisions.</div>
            <a class="ghost-button" href="http://127.0.0.1:8000/docs#operations-default-scoring_predict" target="_blank">API Docs</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        """
        <div class="card">
            <div class="card-header">Fraud Detection</div>
            <div class="card-description">Hybrid supervised + anomaly model with rule-based explanations for transactions.</div>
            <a class="ghost-button" href="http://127.0.0.1:8000/docs#operations-default-fraud_predict" target="_blank">API Docs</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div style="margin-top:20px; display:flex; gap:12px; align-items:center;">
        <a class="primary-button" href="http://127.0.0.1:8000/docs" target="_blank">Open API Docs (Swagger)</a>
        <a class="ghost-button" href="http://127.0.0.1:8000/redoc" target="_blank">Redoc</a>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="footer">Powered by FastAPI · Streamlit · AgroAI</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
