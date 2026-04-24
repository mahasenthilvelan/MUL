import streamlit as st
import pandas as pd
import plotly.express as px

from model_utils import prepare_data, train_model, unlearn, evaluate

st.set_page_config(page_title="UnlearnIQ", layout="wide")

# =========================
# HEADER
# =========================
st.markdown("""
<h1 style='text-align:center;'>🧠 UnlearnIQ</h1>
<h4 style='text-align:center;color:gray;'>
Smart Machine Unlearning Dashboard
</h4>
<hr>
""", unsafe_allow_html=True)

# =========================
# STEP 1: UPLOAD
# =========================
st.markdown("## 📂 Step 1: Upload Dataset")

file = st.file_uploader("Upload your dataset")

if file:
    df = pd.read_csv(file)
    df = prepare_data(df)

    st.success("Dataset Loaded!")

    # =========================
    # STEP 2: TRAIN
    # =========================
    st.markdown("## 🤖 Step 2: Train Baseline Model")

    model, tfidf, acc = train_model(df)

    st.metric("Model Accuracy", round(acc,4))

    # =========================
    # STEP 3: USER SELECT
    # =========================
    st.markdown("## 🧩 Step 3: Select Users to Forget")

    users = sorted(df['UserId'].unique())

    selected_users = st.multiselect(
        "Choose users (search supported)",
        users
    )

    st.info(f"Selected {len(selected_users)} users")

    # =========================
    # STEP 4: UNLEARNING
    # =========================
    if st.button("🚀 Run Unlearning"):

        un_model, tfidf_u = unlearn(df, selected_users)

        # =========================
        # STEP 5: RESULTS
        # =========================
        st.markdown("## 📊 Step 5: Results")

        pred, conf, auc, score = evaluate(
            model, tfidf, un_model, tfidf_u, df, selected_users
        )

        # -------------------------
        # METRIC CARDS
        # -------------------------
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Prediction Change", round(pred,4))
        col2.metric("Confidence Drop", round(conf,4))
        col3.metric("Re-ID AUC", round(auc,4))
        col4.metric("Privacy Score", round(score,4))

        st.markdown("---")

        # -------------------------
        # VISUALIZATION
        # -------------------------
        st.markdown("### 📈 Model Behavior Change")

        chart_df = pd.DataFrame({
            "Metric": ["Prediction Change","Confidence Drop","Privacy Score"],
            "Value": [pred, conf, score]
        })

        fig = px.bar(chart_df, x="Metric", y="Value", text="Value")
        st.plotly_chart(fig, use_container_width=True)

        # -------------------------
        # PROGRESS BAR
        # -------------------------
        st.markdown("### 🔐 Privacy Strength")
        st.progress(score)

        # -------------------------
        # VERDICT
        # -------------------------
        if score > 0.4:
            st.success("✅ Strong Privacy Protection")
        elif score > 0.25:
            st.warning("⚠️ Moderate Privacy")
        else:
            st.error("❌ Weak Privacy")

        # -------------------------
        # EXTRA INSIGHT
        # -------------------------
        st.markdown("### 🧠 Insights")

        st.write("""
- Model forgets selected users partially  
- Confidence reduction indicates uncertainty increase  
- Lower AUC → attacker failure  
- System reduces privacy leakage effectively  
        """)

else:
    st.info("Upload a dataset to begin")
