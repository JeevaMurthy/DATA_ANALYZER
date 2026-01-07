import streamlit as st

# --------------------------------------------------
# APP CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Data Analyzer",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# LOAD GLOBAL CSS
# --------------------------------------------------
def load_css():
    with open("assets/theme.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# --------------------------------------------------
# APP TITLE
# --------------------------------------------------
st.title("📊 Data Analyzer")
st.caption("End-to-End Tool for Data Analysts")

st.markdown(
    """
    👉 Use the sidebar to move through the analysis steps.
    """
