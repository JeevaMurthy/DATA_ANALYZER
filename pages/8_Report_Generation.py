import streamlit as st
import pandas as pd
import io

from modules.report.nlp_report import NLPReportGenerator

def load_css():
    with open("assets/theme.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="NLP Report",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 NLP-Based Analysis Report")
st.caption("Human-readable summary of the entire data analysis pipeline")

st.markdown("---")

# --------------------------------------------------
# CHECK REQUIRED DATA
# --------------------------------------------------
if "df" not in st.session_state:
    st.warning("⚠️ No processed dataset found. Please complete previous steps.")
    st.stop()

df = st.session_state["df"]

ml_results = st.session_state.get("ml_results")
problem_type = st.session_state.get("ml_problem_type")

best_model = None
if ml_results:
    best_model = list(ml_results.keys())[0]

# --------------------------------------------------
# GENERATE REPORT
# --------------------------------------------------
st.markdown("## 📄 Generate NLP Report")

if st.button("🧠 Generate Report"):
    generator = NLPReportGenerator(
        df=df,
        ml_results=ml_results,
        problem_type=problem_type,
        best_model=best_model
    )

    report_text = generator.generate_full_report()
    st.session_state["nlp_report"] = report_text

    st.success("✅ NLP report generated successfully")

# --------------------------------------------------
# DISPLAY REPORT
# --------------------------------------------------
if "nlp_report" in st.session_state:
    st.markdown("## 📑 Report Preview")
    st.markdown(st.session_state["nlp_report"])

    st.markdown("---")

    # --------------------------------------------------
    # REPORT DOWNLOAD
    # --------------------------------------------------
    st.markdown("## ⬇️ Download Report")

    report_bytes = st.session_state["nlp_report"].encode("utf-8")

    st.download_button(
        label="📄 Download Report (TXT)",
        data=report_bytes,
        file_name="analysis_report.txt",
        mime="text/plain"
    )

# --------------------------------------------------
# DATASET DOWNLOAD
# --------------------------------------------------
st.markdown("---")
st.markdown("## 📥 Download Processed Dataset")

csv_data = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Dataset (CSV)",
    data=csv_data,
    file_name="processed_dataset.csv",
    mime="text/csv"
)

# Optional Excel download
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
    df.to_excel(writer, index=False, sheet_name="Processed_Data")

st.download_button(
    label="⬇️ Download Dataset (Excel)",
    data=buffer.getvalue(),
    file_name="processed_dataset.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# --------------------------------------------------
# FINAL MESSAGE
# --------------------------------------------------
st.markdown("---")
st.success(
    "🎉 Analysis complete! You can now download the report and processed dataset."
)
