import requests
import streamlit as st

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
st.title("Credit Scoring")
st.caption("Submit borrower data to the AgroAI credit scoring API and view the risk score.")

with st.form("credit_form"):
    st.markdown("### Applicant profile")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=18, max_value=90, value=32)
        past_due = st.selectbox("Past due history", options=["no", "yes"])
        employment_type = st.selectbox("Employment type", options=["employed", "self-employed", "unemployed"])
    with c2:
        monthly_income = st.number_input("Monthly income", min_value=500, max_value=50000, value=5500, step=100)
        has_mortgage = st.selectbox("Has mortgage", options=["no", "yes"])
        num_prev_loans = st.number_input("Number of previous loans", min_value=0, max_value=30, value=2, step=1)
    with c3:
        credit_utilization = st.slider("Credit utilization", min_value=0.0, max_value=1.0, value=0.35, step=0.01)

    st.markdown("### Loan details")
    l1, l2 = st.columns(2)
    with l1:
        loan_amount = st.number_input("Loan amount", min_value=500, max_value=200000, value=15000, step=500)
    with l2:
        loan_term_months = st.number_input("Loan term (months)", min_value=6, max_value=96, value=36, step=6)

    submitted = st.form_submit_button("Run scoring")

result_card = st.empty()

if submitted:
    payload = {
        "age": age,
        "monthly_income": monthly_income,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "past_due": past_due,
        "employment_type": employment_type,
        "has_mortgage": has_mortgage,
        "num_prev_loans": num_prev_loans,
        "credit_utilization": credit_utilization,
    }
    with st.spinner("Calling scoring API..."):
        try:
            resp = requests.post(f"{API_BASE}/scoring/predict", json=payload, timeout=10)
            if resp.status_code != 200:
                st.error(f"API error: {resp.status_code} - {resp.text}")
            else:
                data = resp.json()
                score = data.get("credit_risk_score", 0)
                badge = render_risk_badge(score)
                msg = (
                    "This applicant is likely safe to approve."
                    if score < 0.2
                    else "Borderline case — consider manual review."
                    if score < 0.5
                    else "High default risk — decline recommended."
                )
                with result_card.container():
                    st.markdown(
                        f"""
                        <div class="card">
                            <div class="card-header">Result</div>
                            <div class="card-description">Aggregated risk indicators from the credit model.</div>
                            <div style="display:flex; gap:16px; align-items:center; flex-wrap:wrap;">
                                <div style="font-size:20px; font-weight:700;">{score:.3f}</div>
                                {badge}
                            </div>
                            <div style="margin-top:8px;">{msg}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                c1, c2, c3 = st.columns(3)
                c1.metric("Credit risk score", f"{score:.3f}")
                c2.metric("Loan amount", f"{loan_amount:,.0f}")
                c3.metric("Income", f"{monthly_income:,.0f}")
                st.json(data)
        except Exception as exc:
            st.error(f"Request failed: {exc}")

st.markdown("</div>", unsafe_allow_html=True)
