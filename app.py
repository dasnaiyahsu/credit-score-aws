"""
app_aws.py — Credit Score Predictor (AWS SageMaker Version)
===========================================================
Tampilan identik dengan app.py lokal, tapi prediksi via SageMaker endpoint.
Run: streamlit run app_aws.py
"""

import json
import os

import boto3
import pandas as pd
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError

#Config
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "credit-score-uas-daz")
REGION        = os.environ.get("AWS_REGION", "us-east-1")

st.set_page_config(
    page_title="Credit Score Predictor",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Urutan fitur 
NUMERIC_FEATURES = [
    "Age", "Num_of_Loan", "Num_of_Delayed_Payment", "Annual_Income",
    "Outstanding_Debt", "Changed_Credit_Limit", "Amount_invested_monthly",
    "Monthly_Balance", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
    "Num_Credit_Card", "Interest_Rate", "Delay_from_due_date",
    "Num_Credit_Inquiries", "Credit_Utilization_Ratio",
    "Credit_History_Age_Months", "Total_EMI_per_month",
]
CATEGORICAL_FEATURES = [
    "Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

#design token
INK        = "#0B1220"
INK_SOFT   = "#141B2D"
PAPER      = "#F7F8FA"
LINE       = "#E4E7EC"
MUTED      = "#6B7280"
EMERALD    = "#0F9D6C"
EMERALD_BG = "#E7F7F1"
AMBER      = "#C8860D"
AMBER_BG   = "#FBF1DD"
ROSE       = "#D14343"
ROSE_BG    = "#FBEAEA"
ACCENT     = "#5B6CFF"

#custom css
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

.stApp {{
    background:
        radial-gradient(circle at 10% 0%, rgba(91,108,255,0.07) 0%, transparent 40%),
        radial-gradient(circle at 90% 100%, rgba(15,157,108,0.07) 0%, transparent 40%),
        {PAPER};
    background-attachment: fixed;
}}

h1, h2, h3, .display-font {{ font-family: 'Sora', sans-serif !important; letter-spacing: -0.02em; }}
.mono {{ font-family: 'JetBrains Mono', monospace; }}

.hero {{
    background: linear-gradient(135deg, {INK} 0%, {INK_SOFT} 55%, #1A2440 100%);
    padding: 3rem 2.5rem;
    border-radius: 20px;
    margin-bottom: 1.75rem;
    position: relative;
    overflow: hidden;
    border: 1px solid #232C45;
}}
.hero::before {{
    content: "";
    position: absolute;
    top: -60%; right: -10%;
    width: 420px; height: 420px;
    background: radial-gradient(circle, rgba(91,108,255,0.18) 0%, transparent 70%);
    pointer-events: none;
}}
.hero-eyebrow {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #8B93B0;
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}
.hero-eyebrow::before {{
    content: "";
    width: 6px; height: 6px;
    border-radius: 50%;
    background: {EMERALD};
    box-shadow: 0 0 0 3px rgba(15,157,108,0.25);
}}
.hero h1 {{ color: #FFFFFF; font-size: 2.4rem; font-weight: 700; margin: 0 0 0.5rem 0; line-height: 1.15; }}
.hero p  {{ color: #A4ACC4; font-size: 1.02rem; margin: 0; max-width: 560px; }}

.stat-row {{ display: flex; gap: 1rem; margin-bottom: 0.5rem; }}
.stat-card {{
    background: #FFFFFF;
    border: 1px solid {LINE};
    border-radius: 14px;
    padding: 1.15rem 1.4rem;
    flex: 1;
    transition: border-color 0.15s ease, transform 0.15s ease;
}}
.stat-card:hover {{ border-color: #C9CEDA; transform: translateY(-1px); }}
.stat-label {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: {MUTED}; font-weight: 600; margin-bottom: 0.35rem; }}
.stat-value {{ font-family: 'Sora', sans-serif; font-size: 1.5rem; font-weight: 700; color: {INK}; }}

.section-head {{ display: flex; align-items: baseline; gap: 0.6rem; margin: 2rem 0 1rem 0; }}
.section-head .num {{ font-family: 'JetBrains Mono', monospace; color: {ACCENT}; font-size: 0.85rem; font-weight: 600; }}
.section-head h3 {{ margin: 0; font-size: 1.15rem; color: {INK}; }}

.result-wrap {{
    border-radius: 18px;
    padding: 2rem 2.2rem;
    margin: 1rem 0 1.5rem 0;
    border: 1px solid;
    position: relative;
    overflow: hidden;
}}
.result-good     {{ background: {EMERALD_BG}; border-color: #BFE6D5; }}
.result-standard {{ background: {AMBER_BG};   border-color: #EFD9A3; }}
.result-poor     {{ background: {ROSE_BG};    border-color: #EFC2C2; }}

.result-top {{ display: flex; align-items: center; gap: 1rem; margin-bottom: 0.6rem; }}
.result-badge {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.25rem 0.6rem;
    border-radius: 6px;
    font-weight: 600;
}}
.badge-good     {{ background: {EMERALD}; color: white; }}
.badge-standard {{ background: {AMBER};   color: white; }}
.badge-poor     {{ background: {ROSE};    color: white; }}

.result-title {{ font-family: 'Sora', sans-serif; font-size: 2.1rem; font-weight: 800; margin: 0; }}
.result-title.good     {{ color: {EMERALD}; }}
.result-title.standard {{ color: {AMBER};   }}
.result-title.poor     {{ color: {ROSE};    }}

.result-advice {{ color: #44495A; font-size: 0.98rem; margin-top: 0.5rem; max-width: 620px; }}
.confidence-pill {{
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: {INK};
    background: rgba(255,255,255,0.6);
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    margin-top: 0.9rem;
    font-weight: 600;
}}

div[data-testid="stForm"] {{
    background: #FFFFFF;
    border: 1px solid {LINE};
    border-radius: 16px;
    padding: 1.75rem 1.75rem 1.25rem 1.75rem;
}}

.stTabs [data-baseweb="tab-list"] {{ gap: 1.5rem; }}
.stTabs [data-baseweb="tab"] {{ font-family: 'Sora', sans-serif; font-weight: 600; font-size: 0.95rem; }}

.stButton>button {{
    width: 100%;
    background: {INK};
    color: white;
    border: none;
    padding: 0.8rem;
    border-radius: 10px;
    font-size: 0.98rem;
    font-weight: 600;
    font-family: 'Sora', sans-serif;
    transition: background 0.15s ease, transform 0.1s ease;
    letter-spacing: -0.01em;
}}
.stButton>button:hover  {{ background: {ACCENT}; transform: translateY(-1px); }}
.stButton>button:active {{ transform: translateY(0px); }}

[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid {LINE}; }}

.sidebar-card {{ background: {PAPER}; border: 1px solid {LINE}; border-radius: 12px; padding: 1rem 1.1rem; margin-bottom: 1rem; }}
.legend-row {{ display: flex; align-items: center; gap: 0.55rem; font-size: 0.88rem; padding: 0.3rem 0; color: #374151; }}
.legend-dot {{ width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }}

footer {{ visibility: hidden; }}
#MainMenu {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


#aws helper
@st.cache_resource
def get_sagemaker_clients():
    sm      = boto3.client("sagemaker",         region_name=REGION)
    runtime = boto3.client("sagemaker-runtime", region_name=REGION)
    return sm, runtime


def invoke_endpoint(raw_input: dict) -> dict:
    """
    Susun feature list sesuai urutan ALL_FEATURES, kirim ke SageMaker,
    kembalikan dict {predicted_class, confidence, probabilities}.
    """
    _, runtime = get_sagemaker_clients()
    features   = [raw_input[k] for k in ALL_FEATURES]
    payload    = {"instances": [features]}
    response   = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    result      = json.loads(response["Body"].read().decode("utf-8"))
    label       = result["labels"][0]
    probs_list  = result["probabilities"][0]
    class_names = result.get("class_names", ["Good", "Poor", "Standard"])
    confidence  = max(probs_list)
    probs_dict  = dict(zip(class_names, probs_list))
    return {"predicted_class": label, "confidence": confidence, "probabilities": probs_dict}


def invoke_endpoint_batch(df: pd.DataFrame) -> list:
    results = []
    for _, row in df.iterrows():
        r = invoke_endpoint(row.to_dict())
        results.append(r["predicted_class"])
    return results


#hero
st.markdown(f"""
<div class="hero">
    <div class="hero-eyebrow">Model live · SageMaker Endpoint · {ENDPOINT_NAME}</div>
    <h1>Credit Score Predictor</h1>
    <p>Machine learning-based credit risk assessment — 21 financial indicators
    distilled into a single, explainable verdict via AWS SageMaker.</p>
</div>
""", unsafe_allow_html=True)

#endpoint
try:
    sm_client, _ = get_sagemaker_clients()
    ep_desc      = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
    ep_status    = ep_desc["EndpointStatus"]
    status_color = EMERALD if ep_status == "InService" else ROSE
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-label">Endpoint</div>
            <div class="stat-value" style="font-size:1rem;">{ENDPOINT_NAME}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Status</div>
            <div class="stat-value" style="color:{status_color}; font-size:1.1rem;">{ep_status}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Region</div>
            <div class="stat-value" style="font-size:1rem;">{REGION}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Features</div>
            <div class="stat-value">{len(ALL_FEATURES)}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if ep_status != "InService":
        st.error(f"Endpoint status: {ep_status}. Tunggu hingga InService sebelum prediksi.")
        st.stop()

except NoCredentialsError:
    st.error("AWS credentials tidak ditemukan. Attach IAM Role ke EC2 instance ini.")
    st.stop()
except ClientError as e:
    st.error(f"Tidak bisa terhubung ke endpoint: {e.response['Error'].get('Message', str(e))}")
    st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### About")
    st.markdown(f"""
    <div class="sidebar-card">
        <div class="legend-row"><span class="legend-dot" style="background:{EMERALD}"></span><b>Good</b> — low credit risk</div>
        <div class="legend-row"><span class="legend-dot" style="background:{AMBER}"></span><b>Standard</b> — average risk</div>
        <div class="legend-row"><span class="legend-dot" style="background:{ROSE}"></span><b>Poor</b> — high credit risk</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Test Cases")
    test_cases = {
        "Good Credit Profile": {
            "Age": 35, "Annual_Income": 95000.0, "Monthly_Inhand_Salary": 7500.0,
            "Num_Bank_Accounts": 3, "Num_Credit_Card": 3, "Interest_Rate": 8,
            "Num_of_Loan": 1, "Delay_from_due_date": 0, "Num_of_Delayed_Payment": 0,
            "Changed_Credit_Limit": 5.0, "Num_Credit_Inquiries": 1.0,
            "Outstanding_Debt": 500.0, "Credit_Utilization_Ratio": 15.0,
            "Total_EMI_per_month": 200.0, "Amount_invested_monthly": 800.0,
            "Monthly_Balance": 3500.0, "Credit_History_Age_Months": 180,
            "Occupation": "Engineer", "Credit_Mix": "Good",
            "Payment_of_Min_Amount": "No", "Payment_Behaviour": "High_spent_Large_value_payments",
        },
        "Standard Credit Profile": {
            "Age": 32, "Annual_Income": 45000.0, "Monthly_Inhand_Salary": 3500.0,
            "Num_Bank_Accounts": 5, "Num_Credit_Card": 5, "Interest_Rate": 18,
            "Num_of_Loan": 3, "Delay_from_due_date": 10, "Num_of_Delayed_Payment": 5,
            "Changed_Credit_Limit": 3.0, "Num_Credit_Inquiries": 5.0,
            "Outstanding_Debt": 2500.0, "Credit_Utilization_Ratio": 35.0,
            "Total_EMI_per_month": 500.0, "Amount_invested_monthly": 200.0,
            "Monthly_Balance": 800.0, "Credit_History_Age_Months": 72,
            "Occupation": "Teacher", "Credit_Mix": "Standard",
            "Payment_of_Min_Amount": "Yes", "Payment_Behaviour": "Low_spent_Medium_value_payments",
        },
        "Poor Credit Profile": {
            "Age": 28, "Annual_Income": 18000.0, "Monthly_Inhand_Salary": 1400.0,
            "Num_Bank_Accounts": 10, "Num_Credit_Card": 9, "Interest_Rate": 32,
            "Num_of_Loan": 8, "Delay_from_due_date": 40, "Num_of_Delayed_Payment": 20,
            "Changed_Credit_Limit": 0.5, "Num_Credit_Inquiries": 18.0,
            "Outstanding_Debt": 5500.0, "Credit_Utilization_Ratio": 75.0,
            "Total_EMI_per_month": 1200.0, "Amount_invested_monthly": 20.0,
            "Monthly_Balance": 100.0, "Credit_History_Age_Months": 12,
            "Occupation": "Musician", "Credit_Mix": "Bad",
            "Payment_of_Min_Amount": "Yes", "Payment_Behaviour": "High_spent_Small_value_payments",
        },
    }
    selected_test = st.selectbox(
        "Load test case", ["(none)"] + list(test_cases.keys()), label_visibility="collapsed"
    )


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

# ── TAB 1: Single Prediction ──────────────────────────────────────────────────
with tab1:
    defaults = test_cases.get(selected_test, {}) if selected_test != "(none)" else {}

    st.markdown('<div class="section-head"><span class="num">01</span><h3>Customer Information</h3></div>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Personal & Income**")
            age            = st.number_input("Age", 18, 100, int(defaults.get("Age", 30)))
            annual_income  = st.number_input("Annual Income ($)", 0.0, 1_000_000.0,
                                             float(defaults.get("Annual_Income", 45000.0)), step=1000.0)
            monthly_salary = st.number_input("Monthly Inhand Salary ($)", 0.0, 100_000.0,
                                             float(defaults.get("Monthly_Inhand_Salary", 3500.0)), step=100.0)
            occ_list   = ["Accountant","Architect","Developer","Doctor","Engineer",
                          "Entrepreneur","Journalist","Lawyer","Manager","Mechanic",
                          "Media_Manager","Musician","Scientist","Teacher","Writer"]
            occupation = st.selectbox(
                "Occupation", occ_list,
                index=occ_list.index(defaults.get("Occupation","Engineer"))
                      if defaults.get("Occupation") in occ_list else 4,
            )

        with c2:
            st.markdown("**Credit Behavior**")
            num_bank    = st.number_input("Num Bank Accounts",        0, 20,  int(defaults.get("Num_Bank_Accounts", 3)))
            num_card    = st.number_input("Num Credit Cards",         0, 20,  int(defaults.get("Num_Credit_Card", 3)))
            interest    = st.number_input("Interest Rate (%)",        1, 50,  int(defaults.get("Interest_Rate", 15)))
            num_loan    = st.number_input("Num of Loans",             0, 20,  int(defaults.get("Num_of_Loan", 2)))
            delay_days  = st.number_input("Delay from Due Date (days)", 0, 100, int(defaults.get("Delay_from_due_date", 5)))
            num_delayed = st.number_input("Num Delayed Payments",     0, 50,  int(defaults.get("Num_of_Delayed_Payment", 3)))

        with c3:
            st.markdown("**Financial Metrics**")
            changed_limit    = st.number_input("Changed Credit Limit",      0.0, 50.0,
                                               float(defaults.get("Changed_Credit_Limit", 5.0)), step=0.5)
            num_inquiries    = st.number_input("Num Credit Inquiries",      0.0, 30.0,
                                               float(defaults.get("Num_Credit_Inquiries", 3.0)), step=1.0)
            outstanding_debt = st.number_input("Outstanding Debt ($)",      0.0, 10_000.0,
                                               float(defaults.get("Outstanding_Debt", 1500.0)), step=100.0)
            utilization      = st.number_input("Credit Utilization (%)",    0.0, 100.0,
                                               float(defaults.get("Credit_Utilization_Ratio", 30.0)), step=1.0)
            emi              = st.number_input("Total EMI/month ($)",       0.0, 5000.0,
                                               float(defaults.get("Total_EMI_per_month", 300.0)), step=10.0)
            amount_invested  = st.number_input("Amount Invested/month ($)", 0.0, 3000.0,
                                               float(defaults.get("Amount_invested_monthly", 200.0)), step=10.0)
            monthly_balance  = st.number_input("Monthly Balance ($)",       0.0, 10_000.0,
                                               float(defaults.get("Monthly_Balance", 1000.0)), step=100.0)
            credit_age       = st.number_input("Credit History Age (months)", 0, 400,
                                               int(defaults.get("Credit_History_Age_Months", 60)))

        st.markdown("**Credit Profile**")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            mix_list   = ["Bad", "Standard", "Good"]
            credit_mix = st.selectbox(
                "Credit Mix", mix_list,
                index=mix_list.index(defaults.get("Credit_Mix","Standard"))
                      if defaults.get("Credit_Mix") in mix_list else 1,
            )
        with cc2:
            pay_list    = ["Yes", "No"]
            payment_min = st.selectbox(
                "Payment of Min Amount", pay_list,
                index=pay_list.index(defaults.get("Payment_of_Min_Amount","No"))
                      if defaults.get("Payment_of_Min_Amount") in pay_list else 1,
            )
        with cc3:
            behav_list = [
                "High_spent_Large_value_payments", "High_spent_Medium_value_payments",
                "High_spent_Small_value_payments", "Low_spent_Large_value_payments",
                "Low_spent_Medium_value_payments", "Low_spent_Small_value_payments",
            ]
            payment_behaviour = st.selectbox(
                "Payment Behaviour", behav_list,
                index=behav_list.index(defaults.get("Payment_Behaviour","High_spent_Large_value_payments"))
                      if defaults.get("Payment_Behaviour") in behav_list else 0,
            )

        submitted = st.form_submit_button("Predict Credit Score")

    if submitted:
        raw_input = {
            "Age": age, "Annual_Income": annual_income,
            "Monthly_Inhand_Salary": monthly_salary, "Num_Bank_Accounts": num_bank,
            "Num_Credit_Card": num_card, "Interest_Rate": interest,
            "Num_of_Loan": num_loan, "Delay_from_due_date": delay_days,
            "Num_of_Delayed_Payment": num_delayed, "Changed_Credit_Limit": changed_limit,
            "Num_Credit_Inquiries": num_inquiries, "Outstanding_Debt": outstanding_debt,
            "Credit_Utilization_Ratio": utilization, "Total_EMI_per_month": emi,
            "Amount_invested_monthly": amount_invested, "Monthly_Balance": monthly_balance,
            "Credit_History_Age_Months": credit_age,
            "Occupation": occupation, "Credit_Mix": credit_mix,
            "Payment_of_Min_Amount": payment_min, "Payment_Behaviour": payment_behaviour,
        }

        try:
            with st.spinner("Calling SageMaker endpoint..."):
                result = invoke_endpoint(raw_input)
        except NoCredentialsError:
            st.error("AWS credentials tidak ditemukan. Attach IAM Role ke instance ini.")
            st.stop()
        except ClientError as e:
            st.error(f"AWS error: {e.response['Error'].get('Message', str(e))}")
            st.stop()

        predicted  = result["predicted_class"]
        confidence = result["confidence"]
        probs      = result["probabilities"]

        style_map  = {"Good": "result-good",     "Standard": "result-standard",   "Poor": "result-poor"}
        badge_map  = {"Good": "badge-good",       "Standard": "badge-standard",    "Poor": "badge-poor"}
        title_cls  = {"Good": "good",             "Standard": "standard",          "Poor": "poor"}
        advice_map = {
            "Good":     "Excellent credit profile. Eligible for premium financial products and preferential rates.",
            "Standard": "Average credit profile. May qualify for standard loans with moderate interest rates.",
            "Poor":     "Credit profile needs improvement. Focus on debt reduction and timely payments.",
        }

        st.markdown('<div class="section-head"><span class="num">02</span><h3>Result</h3></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="result-wrap {style_map.get(predicted, 'result-standard')}">
            <div class="result-top">
                <span class="result-badge {badge_map.get(predicted, 'badge-standard')}">{predicted}</span>
            </div>
            <p class="result-title {title_cls.get(predicted, 'standard')}">{predicted} Credit Score</p>
            <p class="result-advice">{advice_map.get(predicted, '')}</p>
            <span class="confidence-pill">Confidence&nbsp;&nbsp;{confidence:.1%}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-head"><span class="num">03</span><h3>Probability Distribution</h3></div>', unsafe_allow_html=True)
        prob_df = pd.DataFrame({"Class": list(probs.keys()), "Probability": list(probs.values())})
        st.bar_chart(prob_df.set_index("Class"), color=ACCENT)


# ── TAB 2: Batch Prediction ───────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-head"><span class="num">01</span><h3>Upload CSV</h3></div>', unsafe_allow_html=True)
    st.caption("Upload CSV dengan kolom fitur yang sama dengan data training.")

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], label_visibility="collapsed")
    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-card"><div class="stat-label">Rows</div><div class="stat-value">{df_up.shape[0]:,}</div></div>
            <div class="stat-card"><div class="stat-label">Columns</div><div class="stat-value">{df_up.shape[1]}</div></div>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(df_up.head(5), use_container_width=True)

        if st.button("Run Batch Prediction"):
            missing = [f for f in ALL_FEATURES if f not in df_up.columns]
            if missing:
                st.error(f"Kolom tidak lengkap di CSV: {missing}")
            else:
                try:
                    with st.spinner(f"Predicting {len(df_up)} records via SageMaker..."):
                        preds = invoke_endpoint_batch(df_up[ALL_FEATURES])
                    df_up["Predicted_Credit_Score"] = preds
                    st.success(f"Predicted {len(preds)} records.")
                    st.markdown('<div class="section-head"><span class="num">02</span><h3>Results</h3></div>', unsafe_allow_html=True)
                    st.dataframe(
                        df_up[["Predicted_Credit_Score"] + [c for c in df_up.columns if c != "Predicted_Credit_Score"]],
                        use_container_width=True,
                    )
                    st.bar_chart(df_up["Predicted_Credit_Score"].value_counts(), color=ACCENT)
                    csv_out = df_up.to_csv(index=False).encode("utf-8")
                    st.download_button("Download Predictions", csv_out, "predictions.csv", "text/csv")
                except (NoCredentialsError, ClientError) as e:
                    st.error(f"AWS error saat batch prediction: {e}")

st.markdown(f"""
<div style="text-align:center; padding: 2rem 0 1rem 0; color: {MUTED}; font-size: 0.85rem;">
    Credit Score Predictor · Dataset B · Built with Streamlit & AWS SageMaker
</div>
""", unsafe_allow_html=True)
