import streamlit as st
import numpy as np
import pandas as pd
import joblib
import random
import time
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from lime.lime_tabular import LimeTabularExplainer


# =========================
# LOAD MODEL & SCALER
# =========================
MODEL_PATH = "fraud_model_isotonic.pkl" if Path("fraud_model_isotonic.pkl").exists() else "fraud_model.pkl"
model = joblib.load(MODEL_PATH)
scaler = joblib.load("scaler.pkl")
background_data = joblib.load("explain_background.pkl")
feature_names = joblib.load("feature_names.pkl")


@st.cache_resource
def load_lime_explainer():
    return LimeTabularExplainer(
        training_data=background_data,
        feature_names=feature_names,
        class_names=["Genuine", "Fraud"],
        mode="classification",
        random_state=42
    )


lime_explainer = load_lime_explainer()


def explain_with_lime(step, txn):
    txn_type = txn["type"]

    raw_row = np.array([[
        step,
        txn["amount"],
        txn["oldOrg"],
        txn["newOrg"],
        txn["oldDest"],
        txn["newDest"],
        1 if txn_type == "CASH_OUT" else 0,
        1 if txn_type == "DEBIT" else 0,
        1 if txn_type == "PAYMENT" else 0,
        1 if txn_type == "TRANSFER" else 0
    ]], dtype=float)

    def predict_from_raw(data):
        data_scaled = scaler.transform(data)
        return model.predict_proba(data_scaled)

    explanation = lime_explainer.explain_instance(
        data_row=raw_row[0],
        predict_fn=predict_from_raw,
        labels=(1,),
        num_features=8,
        num_samples=1000
    )

    return explanation.as_list(label=1)

# =========================
# PAGE CONFIG

# =========================
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("Online Transaction Fraud Detection System")
st.caption("Predictive Analytics with Explainable Machine Learning")

# =========================
# SESSION STATE DEFAULTS
# =========================
defaults = {
    "running": False,
    "paused": False,
    "txn_id": 1,
    "current_txn": None,
    "prob_history": [],
    "transactions": [],  # stores recent transactions for table & metrics
    "y_true": [],
    "y_pred": []
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================
# HELPER FUNCTIONS
# =========================


def predict(step, amount, oldOrg, newOrg, oldDest, newDest, txn_type):
    X = [[
        step, amount, oldOrg, newOrg,
        oldDest, newDest,
        int(txn_type == "CASH_OUT"),
        int(txn_type == "DEBIT"),
        int(txn_type == "PAYMENT"),
        int(txn_type == "TRANSFER")
    ]]

    X_scaled = scaler.transform(X)
    probabilities = model.predict_proba(X_scaled)

    fraud_index = list(model.classes_).index(1)
    prob = float(probabilities[0][fraud_index])
    prob = max(0.0, min(1.0, prob))

    label = 1 if prob >= 0.5 else 0
    return prob, label


def generate_transaction(step):
    scenario = random.choices(
        ["normal", "suspicious", "fraud"],
        weights=[70, 20, 10]
    )[0]

    if scenario == "normal":
        amount = random.randint(1000, 50000)
        oldOrg = random.randint(amount + 10000, amount + 200000)
        newOrg = oldOrg - amount

        oldDest = random.randint(10000, 200000)
        newDest = oldDest + amount

    elif scenario == "suspicious":
        amount = random.randint(50000, 200000)
        oldOrg = random.randint(amount, amount + 100000)
        newOrg = oldOrg - random.randint(0, amount)

        oldDest = random.randint(0, 100000)
        newDest = oldDest + amount + random.randint(10000, 50000)

    else:
        amount = random.randint(200000, 500000)
        oldOrg = random.randint(10000, 100000)
        newOrg = random.randint(0, oldOrg)

        oldDest = random.randint(0, 10000)
        newDest = oldDest + amount

    txn_type = random.choice([
        "CASH_OUT",
        "TRANSFER",
        "PAYMENT",
        "DEBIT"
    ])

    return amount, oldOrg, newOrg, oldDest, newDest, txn_type

def risk_label(prob):
    if prob > 0.5:
        return "FRAUDULENT TRANSACTION", "error"
    elif prob > 0.2:
        return "SUSPICIOUS TRANSACTION", "warning"
    else:
        return "GENUINE TRANSACTION", "success"

def explain_transaction(txn):
    reasons = []
    if txn["amount"] > 200000:
        reasons.append("Very high transaction amount")
    if abs((txn["oldOrg"] - txn["newOrg"]) - txn["amount"]) > 1:
        reasons.append("Sender balance mismatch")
    if txn["type"] in ["CASH_OUT", "TRANSFER"]:
        reasons.append("High-risk transaction type")
    if (txn["newDest"] - txn["oldDest"]) > 200000:
        reasons.append("Sudden increase in receiver balance")
    if txn["prob"] > 0.8:
        reasons.append("Strong ML fraud pattern detected")
    return reasons

# =========================
# SIDEBAR: Monitoring Mode
# =========================
mode = st.sidebar.radio(
    "Select Monitoring Mode",
    ["Real-Time Monitoring", "Manual Transaction Check"]
)
st.sidebar.success(f"Model Loaded: {MODEL_PATH}")

# =========================
# REAL-TIME MONITORING
# =========================
if mode == "Real-Time Monitoring":

    st.subheader("Real-Time Transaction Monitoring")

    # ---- Controls ----
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("▶ Start"):
            st.session_state.running = True
            st.session_state.paused = False
    with c2:
        if st.button("⏸ Pause"):
            st.session_state.paused = True
    with c3:
        if st.button("▶ Resume"):
            st.session_state.paused = False

    st.divider()

    txn_box = st.empty()
    explain_box = st.empty()
    graph_box = st.empty()
    metrics_box = st.empty()
    table_box = st.empty()
    status_box = st.empty()

    if st.session_state.running:

        if not st.session_state.paused:
            step = st.session_state.txn_id

            amount, oldOrg, newOrg, oldDest, newDest, txn_type = generate_transaction(step)

            prob, label = predict(
                step,
                amount,
                oldOrg,
                newOrg,
                oldDest,
                newDest,
                txn_type
            )

            txn = {
                "id": step,
                "amount": amount,
                "type": txn_type,
                "oldOrg": oldOrg,
                "newOrg": newOrg,
                "oldDest": oldDest,
                "newDest": newDest,
                "prob": prob,
                "risk_score": round(prob * 100, 2),
                "label": label
            }

            st.session_state.current_txn = txn
            st.session_state.transactions.append(txn)
            st.session_state.prob_history.append(prob)
            st.session_state.txn_id += 1

        txn = st.session_state.current_txn
        label_text, label_color = risk_label(txn["prob"])

        # ---- Current Transaction ----
        with txn_box.container():
            st.markdown("Current Transaction")
            a, b, c, d = st.columns(4)
            a.metric("Transaction ID", txn["id"])
            b.metric("Amount", f"₹{txn['amount']:,}")
            c.metric("Type", txn["type"])
            d.metric("Risk Score", f"{txn['risk_score']:.2f}/100")
            st.progress(max(0, min(100, int(txn["prob"] * 100))))
            st.write(f"Fraud Probability: **{txn['prob']:.2%}**")
            if label_color == "error":
                st.error(label_text)
            elif label_color == "warning":
                st.warning(label_text)
            else:
                st.success(label_text)

        # ---- Explanation ----
        with explain_box.container():
            st.markdown("Risk Explanation")

            if txn["prob"] > 0.4:
                reasons = explain_transaction(txn)
                if reasons:
                    for r in reasons:
                        st.warning(r)
                else:
                    st.info("The model detected elevated risk, but no predefined rule matched.")
            else:
                st.success("No abnormal behavior detected.")

            # ---- Explainable AI (LIME) ----
            st.markdown("### Explainable AI (LIME)")

            with st.spinner("Generating local explanation..."):
                lime_results = explain_with_lime(txn["id"], txn)

            lime_df = pd.DataFrame(
                lime_results,
                columns=["Feature contribution", "Impact on fraud"]
            )

            st.dataframe(lime_df, use_container_width=True)

            st.bar_chart(
                lime_df.set_index("Feature contribution")
            )

            st.caption(
                "Positive contributions support the fraud class; "
                "negative contributions support the genuine class. "
                "LIME provides a local approximation of the model."
            )
        # ---- Dashboard Metrics ----
        with metrics_box.container():
            st.markdown("Cumulative Metrics")
            total_txns = len(st.session_state.transactions)
            total_fraud = sum(t["label"] for t in st.session_state.transactions)
            avg_prob = np.mean(st.session_state.prob_history) if st.session_state.prob_history else 0

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Transactions", total_txns)
            m2.metric("Total Frauds", total_fraud)
            m3.metric("Avg Fraud Probability", f"{avg_prob:.2%}")

        # ---- Recent Transactions Table ----
        with table_box.container():
            st.markdown("Recent Transactions (Last 10)")
            df_recent = pd.DataFrame(st.session_state.transactions[-10:])
            df_recent["Risk Score (%)"] = df_recent["prob"].apply(lambda p: round(p * 100, 2))
            df_recent["Risk Level"] = df_recent["prob"].apply(lambda p: risk_label(p)[0])
            st.dataframe(
                df_recent[["id", "amount", "type", "Risk Score (%)", "Risk Level"]].rename(
                    columns={"id": "Txn ID", "amount": "Amount (₹)", "type": "Type"}
                ),
                use_container_width=True
            )

        # ---- Trend Chart ----
        with graph_box.container():
            st.markdown("Fraud Probability Trend")
            df_trend = pd.DataFrame({
                "Txn ID": range(1, len(st.session_state.prob_history)+1),
                "Fraud Probability": st.session_state.prob_history
            })
            df_trend["Rolling Avg"] = df_trend["Fraud Probability"].rolling(5).mean()
            st.line_chart(df_trend.set_index("Txn ID"))

        # ---- Predictive Insight ----
        last_rolling = df_trend["Rolling Avg"].iloc[-1] if len(df_trend)>4 else np.nan
        trend_msg = "➖ Stable Risk"
        if len(df_trend) > 4:
            if last_rolling > df_trend["Rolling Avg"].iloc[-5]:
                trend_msg = "📈 Risk Increasing"
            elif last_rolling < df_trend["Rolling Avg"].iloc[-5]:
                trend_msg = "📉 Risk Decreasing"

        st.markdown("Predictive Insight")
        st.write(f"Trend Direction: **{trend_msg}**")
        if not np.isnan(last_rolling):
            st.write(f"Predicted Next Transaction Fraud Probability: `{last_rolling:.2%}`")
            if last_rolling > 0.7:
                st.error("High probability of fraud in upcoming transactions")
            elif last_rolling > 0.5:
                st.warning("Fraud risk increasing")
            else:
                st.success("Low predicted risk")

        # ---- Status ----
        with status_box.container():
            if st.session_state.paused:
                st.info("⏸ Monitoring paused")
            else:
                st.success("▶ Monitoring live")

        # Controlled refresh
        if not st.session_state.paused:
            time.sleep(1.5)
            st.rerun()

    else:
        st.info("Click ▶ Start to begin monitoring")

# =========================
# MANUAL TRANSACTION CHECK
# =========================
elif mode == "Manual Transaction Check":

    st.subheader("Manual Transaction Fraud Analysis")

    with st.form("manual_form"):
        step = st.number_input("Transaction Step", value=1)
        amount = st.number_input("Amount", value=1000.0)
        oldOrg = st.number_input("Sender Balance Before", value=50000.0)
        newOrg = st.number_input("Sender Balance After", value=49000.0)
        oldDest = st.number_input("Receiver Balance Before", value=20000.0)
        newDest = st.number_input("Receiver Balance After", value=21000.0)
        txn_type = st.selectbox("Transaction Type", ["CASH_OUT","TRANSFER","PAYMENT","DEBIT"])
        submitted = st.form_submit_button("Check Fraud")

    
    if submitted:
        prob, label = predict(
            step, amount, oldOrg, newOrg, oldDest, newDest, txn_type
        )
        label_text, label_color = risk_label(prob)

        st.markdown("## Prediction Result")
        st.progress(max(0, min(100, int(prob * 100))))
        score_col, probability_col = st.columns(2)
        score_col.metric("Risk Score", f"{prob * 100:.2f}/100")
        probability_col.metric("Fraud Probability", f"{prob:.2%}")
        st.caption("Risk Score is the model's estimated fraud probability expressed on a 0–100 scale.")

        if label_color == "error":
            st.error(label_text)
        elif label_color == "warning":
            st.warning(label_text)
        else:
            st.success(label_text)

        txn = {
            "amount": amount,
            "oldOrg": oldOrg,
            "newOrg": newOrg,
            "oldDest": oldDest,
            "newDest": newDest,
            "type": txn_type,
            "prob": prob,
            "risk_score": round(prob * 100, 2)
        }

        if prob > 0.4:
            st.subheader("Risk Explanation")
            reasons = explain_transaction(txn)

            if reasons:
                for r in reasons:
                    st.warning(r)
            else:
                st.info("Elevated risk detected, but no predefined rule matched.")
        else:
            st.success("Transaction is normal. No fraud indicators detected.")

        # Explainable AI (LIME)
        st.markdown("### Explainable AI (LIME)")

        with st.spinner("Generating local explanation..."):
            lime_results = explain_with_lime(step, txn)

        lime_df = pd.DataFrame(
            lime_results,
            columns=["Feature contribution", "Impact on fraud"]
        )

        st.dataframe(lime_df, use_container_width=True)
        st.bar_chart(lime_df.set_index("Feature contribution"))

        st.caption(
            "Positive contributions support the fraud class, "
            "while negative contributions support the genuine class. "
            "LIME provides a local explanation of the prediction."
        )
