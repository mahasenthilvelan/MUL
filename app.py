import streamlit as st
import time
import pandas as pd

from model_utils import prepare_data, train_model, unlearn, evaluate

st.set_page_config(page_title="UnlearnIQ", layout="wide")

# =========================
# SESSION STATE
# =========================
if "page" not in st.session_state:
    st.session_state.page = "splash"

# =========================
# SPLASH SCREEN
# =========================
if st.session_state.page == "splash":
    st.markdown(
        """
        <h1 style='text-align:center;'>🧠 UnlearnIQ</h1>
        <h3 style='text-align:center;'>Intelligent Machine Unlearning System</h3>
        """,
        unsafe_allow_html=True
    )
    time.sleep(2)
    st.session_state.page = "upload"
    st.rerun()

# =========================
# PAGE 1: UPLOAD
# =========================
elif st.session_state.page == "upload":

    st.title("📂 Upload Dataset")

    file = st.file_uploader("Upload CSV")

    if file:
        df = pd.read_csv(file)
        df = prepare_data(df)

        st.session_state.df = df
        st.success("Dataset Loaded!")

        if st.button("Next → Train Model"):
            st.session_state.page = "train"
            st.rerun()

# =========================
# PAGE 2: TRAIN
# =========================
elif st.session_state.page == "train":

    st.title("🤖 Training Model")

    df = st.session_state.df

    model, tfidf, acc = train_model(df)

    st.session_state.model = model
    st.session_state.tfidf = tfidf

    st.metric("Accuracy", round(acc,4))

    if st.button("Next → Select Users"):
        st.session_state.page = "select"
        st.rerun()

# =========================
# PAGE 3: SELECT USERS
# =========================
elif st.session_state.page == "select":

    st.title("🧩 Select Users to Forget")

    df = st.session_state.df

    users = sorted(df['UserId'].unique())

    selected = st.multiselect("Select Users", users)

    st.session_state.selected = selected

    if st.button("Next → Run Unlearning"):
        st.session_state.page = "unlearn"
        st.rerun()

# =========================
# PAGE 4: UNLEARNING
# =========================
elif st.session_state.page == "unlearn":

    st.title("⚙️ Running Unlearning...")

    df = st.session_state.df
    selected = st.session_state.selected

    model = st.session_state.model
    tfidf = st.session_state.tfidf

    un_model, tfidf_u = unlearn(df, selected)

    st.session_state.un_model = un_model
    st.session_state.tfidf_u = tfidf_u

    st.success("Unlearning Completed!")

    if st.button("View Results"):
        st.session_state.page = "results"
        st.rerun()

# =========================
# PAGE 5: RESULTS
# =========================
elif st.session_state.page == "results":

    st.title("📊 Results Dashboard")

    df = st.session_state.df
    selected = st.session_state.selected

    pred, conf, auc, score = evaluate(
        st.session_state.model,
        st.session_state.tfidf,
        st.session_state.un_model,
        st.session_state.tfidf_u,
        df,
        selected
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Prediction Change", round(pred,4))
    col2.metric("Confidence Drop", round(conf,4))
    col3.metric("Re-ID AUC", round(auc,4))

    st.markdown("---")

    st.subheader("🔐 Privacy Score")
    st.progress(score)

    if score > 0.4:
        st.success("✅ Strong Privacy")
    elif score > 0.25:
        st.warning("⚠️ Moderate Privacy")
    else:
        st.error("❌ Weak Privacy")
