"""
app.py — Streamlit Frontend for Credit Score Prediction
========================================================
Disesuaikan dengan aws_pipeline_v2.py:
- Fitur sesuai DataPreprocessor pipeline (bukan list lama)
- Credit_History_Age di-parse ke bulan di sisi client
- Tidak ada Month, Spending_Level, Payments_Value_Level, loan one-hot
- Cat cols: Occupation, Credit_Mix, Payment_of_Min_Amount, Payment_Behaviour
- Label kelas dinamis dari response endpoint (class_names)
"""

import json
import os

import boto3
import pandas as pd
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "credit-score-uas-daz")
REGION        = os.environ.get("AWS_REGION", "us-east-1")

# Urutan harus identik dengan FEATURE_NAMES di inference.py
NUMERIC_FEATURES = [
    "Age",
    "Num_of_Loan",
    "Num_of_Delayed_Payment",
    "Annual_Income",
    "Outstanding_Debt",
    "Changed_Credit_Limit",
    "Amount_invested_monthly",
    "Monthly_Balance",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Delay_from_due_date",
    "Num_Credit_Inquiries",
    "Credit_Utilization_Ratio",
    "Credit_History_Age_Months",
    "Total_EMI_per_month",
]

CATEGORICAL_FEATURES = [
    "Occupation",
    "Credit_Mix",
    "Payment_of_Min_Amount",
    "Payment_Behaviour",
]


@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)


def invoke_endpoint(features: list) -> dict:
    runtime = get_runtime_client()
    payload  = {"instances": [features]}
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode("utf-8"))


# ─────────────────────────────────────────
# Helper: parse Credit_History_Age ke bulan
# (sama dengan _parse_credit_age di pipeline)
# ─────────────────────────────────────────
def parse_credit_age_months(years: int, months: int) -> int:
    return years * 12 + months


# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────
st.title("Credit Score Prediction")
st.caption("Model: aws_pipeline_v2 · SageMaker Endpoint")

with st.sidebar:
    st.header("Data Nasabah")

    # ── Numerik ──────────────────────────
    Age = st.slider("Usia nasabah (tahun)", 18, 100, value=30, step=1)

    Annual_Income = st.number_input(
        "Pendapatan tahunan kotor (Annual Income)", 0.0, 25_000_000.0, step=100.0
    )
    Monthly_Inhand_Salary = st.number_input(
        "Gaji bersih bulanan (Monthly Inhand Salary)", 0.0, 100_000.0, step=100.0
    )
    Num_Bank_Accounts = st.slider("Jumlah rekening bank aktif", 0, 15, step=1)
    Num_Credit_Card   = st.slider("Jumlah kartu kredit aktif", 0, 15, step=1)
    Interest_Rate     = st.slider("Suku bunga rata-rata (%)", 0, 40, step=1)
    Num_of_Loan       = st.slider("Jumlah pinjaman berjalan", 0, 10, step=1)

    Delay_from_due_date = st.slider(
        "Hari keterlambatan dari jatuh tempo", 0, 100, step=1
    )
    Num_of_Delayed_Payment = st.slider(
        "Frekuensi keterlambatan bayar", 0, 50, step=1
    )
    Changed_Credit_Limit = st.number_input(
        "Perubahan limit kartu kredit (%)", -50.0, 50.0, step=0.01
    )
    Num_Credit_Inquiries = st.slider(
        "Jumlah pengecekan riwayat kredit", 0, 20, step=1
    )
    Outstanding_Debt = st.number_input(
        "Total sisa utang belum lunas", 0.0, 100_000.0, step=100.0
    )
    Credit_Utilization_Ratio = st.slider(
        "Utilisasi limit kredit (%)", 0.0, 100.0, step=0.1
    )

    st.markdown("**Umur riwayat kredit**")
    credit_age_years  = st.number_input("Tahun", 0, 100, step=1)
    credit_age_months = st.number_input("Bulan", 0, 11, step=1)
    Credit_History_Age_Months = parse_credit_age_months(
        int(credit_age_years), int(credit_age_months)
    )
    st.caption(f"→ {Credit_History_Age_Months} bulan")

    Total_EMI_per_month    = st.number_input("Total cicilan bulanan (EMI)", 0.0, 90_000.0, step=100.0)
    Amount_invested_monthly = st.number_input("Investasi bulanan", 0.0, 10_000.0, step=10.0)
    Monthly_Balance        = st.number_input("Saldo akhir bulan", 0.0, 10_000.0, step=10.0)

    # ── Kategorikal ──────────────────────
    st.markdown("---")
    Occupation = st.selectbox(
        "Pekerjaan / Profesi",
        [
            "Accountant", "Architect", "Developer", "Doctor", "Engineer",
            "Entrepreneur", "Journalist", "Lawyer", "Manager", "Mechanic",
            "Media_Manager", "Musician", "Scientist", "Teacher", "Writer",
        ],
    )
    Credit_Mix = st.radio(
        "Credit Mix (keragaman jenis kredit)",
        ["Bad", "Standard", "Good"],
    )
    Payment_of_Min_Amount = st.radio(
        "Sering bayar hanya minimum tagihan?",
        ["No", "Yes", "NM"],
    )
    Payment_Behaviour = st.selectbox(
        "Payment Behaviour",
        [
            "High_spent_Large_value_payments",
            "High_spent_Medium_value_payments",
            "High_spent_Small_value_payments",
            "Low_spent_Large_value_payments",
            "Low_spent_Medium_value_payments",
            "Low_spent_Small_value_payments",
        ],
    )

# ─────────────────────────────────────────
# Predict
# ─────────────────────────────────────────
if st.button("Predict", type="primary"):
    # Susun fitur sesuai urutan NUMERIC_FEATURES + CATEGORICAL_FEATURES
    features = [
        Age,
        Num_of_Loan,
        Num_of_Delayed_Payment,
        Annual_Income,
        Outstanding_Debt,
        Changed_Credit_Limit,
        Amount_invested_monthly,
        Monthly_Balance,
        Monthly_Inhand_Salary,
        Num_Bank_Accounts,
        Num_Credit_Card,
        Interest_Rate,
        Delay_from_due_date,
        Num_Credit_Inquiries,
        Credit_Utilization_Ratio,
        Credit_History_Age_Months,
        Total_EMI_per_month,
        # categorical
        Occupation,
        Credit_Mix,
        Payment_of_Min_Amount,
        Payment_Behaviour,
    ]

    try:
        result = invoke_endpoint(features)
    except NoCredentialsError:
        st.error(
            "AWS credentials tidak ditemukan. "
            "Jika di EC2, attach LabInstanceProfile. "
            "Jika lokal, konfigurasikan ~/.aws/credentials."
        )
    except ClientError as e:
        st.error(f"AWS error: {e.response['Error'].get('Message', str(e))}")
    else:
        label       = result["labels"][0]
        probs       = result["probabilities"][0]
        class_names = result.get("class_names", ["Good", "Poor", "Standard"])

        st.success(f"Predicted Credit Score: **{label}**")

        st.write("#### Probabilitas per Kelas")
        chart_data = pd.DataFrame(
            {"Probability": probs},
            index=class_names,
        )
        st.bar_chart(chart_data)

        with st.expander("Detail input yang dikirim ke endpoint"):
            all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
            st.json(dict(zip(all_features, features)))
