import streamlit as st
import pandas as pd

from model_utils import train_model, unlearn, evaluate

st.set_page_config(page_title="Machine Unlearning App", layout="wide")

st.title("🧠 Multi-Criteria Machine Unlearning System")

# ---------------------------
# Upload Dataset
# ---------------------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    df = df[['Text', 'label', 'UserId']]

    st.success("Dataset Loaded!")

    # ---------------------------
    # Train Baseline
    # ---------------------------
    st.subheader("🔹 Baseline Model")

    model, tfidf, acc, X_train_tf, X_test_tf = train_model(df)

    st.write("Accuracy:", round(acc, 4))

    # ---------------------------
    # User Selection
    # ---------------------------
    st.subheader("🧩 Select Users to Forget")

    all_users = sorted(df['UserId'].unique())

    selected_users = st.multiselect(
        "Select Users",
        options=all_users
    )

    # ---------------------------
    # Run Unlearning
    # ---------------------------
    if st.button("🚀 Run Unlearning"):

        if len(selected_users) == 0:
            st.warning("Select users first")
        else:

            un_model, tfidf_u, X_tr_tf, X_te_tf = unlearn(df, selected_users)

            pred_change, conf_drop, mia, auc, score = evaluate(
                model, tfidf, un_model, tfidf_u, df, selected_users
            )

            # ---------------------------
            # Results
            # ---------------------------
            st.subheader("📊 Results")

            col1, col2, col3 = st.columns(3)

            col1.metric("Prediction Change", round(pred_change, 4))
            col2.metric("Confidence Drop", round(conf_drop, 4))
            col3.metric("MIA", round(mia, 4))

            col4, col5 = st.columns(2)

            col4.metric("Re-ID AUC", round(auc, 4))
            col5.metric("Final Score", round(score, 4))

            # Verdict
            if score > 0.4:
                st.success("✅ Strong Privacy")
            elif score > 0.25:
                st.warning("⚠️ Moderate Privacy")
            else:
                st.error("❌ Weak Privacy")
