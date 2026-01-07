import streamlit as st
import pandas as pd
from modules.ingestion.loader import load_dataset

# --------------------------------------------------
# LOAD THEME
# --------------------------------------------------
def load_css():
    with open("assets/theme.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------
st.title("📂 Upload Dataset")
st.caption("Step 1: Upload and inspect the dataset")

st.markdown("---")

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file:
    try:
        df = load_dataset(uploaded_file)

        # Save dataset
        st.session_state["df"] = df

        st.success("✅ Dataset loaded successfully")

        # --------------------------------------------------
        # METRICS
        # --------------------------------------------------
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", df.isnull().sum().sum())

        # --------------------------------------------------
        # PREVIEW
        # --------------------------------------------------
        st.markdown("### 🔍 Dataset Preview (First 5 Rows)")
        st.dataframe(df.head(), use_container_width=True)

        st.info(
            "ℹ️ Data is loaded. Move to **Data Cleaning** from the sidebar."
        )

    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
else:
    st.warning("⬆️ Please upload a dataset to continue.")
