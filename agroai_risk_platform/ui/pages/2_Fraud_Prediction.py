import requests
import streamlit as st
from datetime import datetime

API_BASE = "http://127.0.0.1:8000"

HIDE_APP_NAV = """
<style>
[data-testid="stSidebarNav"] > ul:nth-of-type(1) > li:nth-of-type(1) { display: none; }
</style>
"""

st.markdown(HIDE_APP_NAV, unsafe_allow_html=True)


def render_risk_badge(score: float) -> str:
    if score < 0.3:
        return '<span class="pill-badge pill-success">Low</span>'
    if score < 0.7:
        return '<span class="pill-badge pill-warning">Medium</span>'
    return '<span class="pill-badge pill-danger">High</span>'


st.markdown('<div class="app-container">', unsafe_allow_html=True)
st.title("Fraud Prediction & Explanation")
st.caption("Hybrid supervised + anomaly model with rule-based explanations.")

col_left, col_right = st.columns([1.1, 0.9])

with col_left:
    with st.form("fraud_form"):
        st.markdown("### Transaction basics")
        c1, c2 = st.columns(2)
        with c1:
            transaction_id = st.text_input("Transaction ID", "T100")
            customer_id = st.text_input("Customer ID", "C10")
            amount = st.number_input("Amount", min_value=0.01, value=220.0, step=10.0, format="%.2f")
            currency = st.selectbox("Currency", ["USD", "EUR", "GBP"])
        with c2:
            timestamp = st.text_input("Timestamp (ISO)", datetime.utcnow().isoformat(timespec="minutes"))
            country = st.text_input("Country", "US")
            merchant_id = st.text_input("Merchant ID", "M50")
            merchant_category = st.selectbox(
                "Merchant category",
                ["grocery", "electronics", "travel", "gaming", "luxury", "services", "fuel", "jewelry"],
            )

        with st.expander("Advanced signals", expanded=False):
            c3, c4 = st.columns(2)
            with c3:
                channel = st.selectbox("Channel", ["POS", "ATM", "ONLINE", "MOBILE"])
                device_id = st.text_input("Device ID", "D10")
                same_day_transactions_count = st.number_input("Same-day tx count", min_value=0, value=5, step=1)
            with c4:
                is_new_country = st.selectbox("Is new country", [0, 1], format_func=lambda x: "Yes" if x else "No")
                is_new_device = st.selectbox("Is new device", [0, 1], format_func=lambda x: "Yes" if x else "No")
                average_customer_spend = st.number_input(
                    "Average customer spend", min_value=0.01, value=180.0, step=10.0, format="%.2f"
                )

        submitted = st.form_submit_button("Run fraud detection")

with col_right:
    result_area = st.empty()

payload = None
if submitted:
    payload = {
        "transaction_id": transaction_id,
        "customer_id": customer_id,
        "amount": amount,
        "currency": currency,
        "timestamp": timestamp,
        "country": country,
        "merchant_id": merchant_id,
        "merchant_category": merchant_category,
        "channel": channel,
        "device_id": device_id,
        "is_new_country": int(is_new_country),
        "is_new_device": int(is_new_device),
        "same_day_transactions_count": int(same_day_transactions_count),
        "average_customer_spend": average_customer_spend,
    }
    try:
        with st.spinner("Calling fraud API..."):
            resp_predict = requests.post(f"{API_BASE}/fraud/predict", json=payload, timeout=15)
            resp_explain = requests.post(f"{API_BASE}/fraud/explain", json=payload, timeout=15)
        if resp_predict.status_code != 200:
            st.error(f"Predict API error: {resp_predict.status_code} - {resp_predict.text}")
        else:
            pred = resp_predict.json()
            score = pred.get("combined_fraud_score", 0)
            sup = pred.get("supervised_score", 0)
            ano = pred.get("anomaly_score", 0)
            badge = render_risk_badge(score)
            with result_area.container():
                st.markdown(
                    f"""
                    <div class="card">
                        <div class="card-header">Fraud Scores</div>
                        <div class="card-description">Combined hybrid score with supervised and anomaly components.</div>
                        <div style="display:flex; gap:16px; align-items:center; flex-wrap:wrap;">
                            <div style="font-size:22px; font-weight:700;">{score:.3f}</div>
                            {badge}
                        </div>
                        <div style="margin-top:8px;">{"Low fraud risk" if score < 0.4 else "Monitor closely" if score < 0.8 else "High fraud likelihood — block or review immediately."}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            c1, c2, c3 = st.columns(3)
            c1.metric("Combined fraud score", f"{score:.3f}")
            c2.metric("Supervised score", f"{sup:.3f}")
            c3.metric("Anomaly score", f"{ano:.3f}")

        if resp_explain.status_code == 200:
            exp = resp_explain.json()
            reasons = exp.get("explanation", [])
            st.markdown(
                """
                <div class="card">
                    <div class="card-header">Explanations</div>
                    <div class="card-description">Rule-based drivers flagged by the hybrid model.</div>
                """,
                unsafe_allow_html=True,
            )
            if reasons:
                for r in reasons:
                    st.markdown(
                        f'<div class="pill-badge pill-warning" style="margin:4px 4px 0 0; display:inline-block;">{r}</div>',
                        unsafe_allow_html=True,
                    )
            st.markdown("</div>", unsafe_allow_html=True)
            st.json(exp)
        else:
            st.error(f"Explain API error: {resp_explain.status_code} - {resp_explain.text}")
    except Exception as exc:
        st.error(f"Request failed: {exc}")

st.markdown("</div>", unsafe_allow_html=True)
