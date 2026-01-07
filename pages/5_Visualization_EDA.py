import streamlit as st
import pandas as pd

from modules.visualization.eda_visuals import *
def load_css():
    with open("assets/theme.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="EDA Visualization",
    page_icon="📊",
    layout="wide"
)

st.title("📊 EDA Visualization")
st.caption("Interactive, filter-based exploratory visualizations")

st.markdown("---")

# --------------------------------------------------
# CHECK DATA
# --------------------------------------------------
if "df" not in st.session_state:
    st.warning("⚠️ Please upload and clean a dataset first.")
    st.stop()

df = st.session_state["df"]

num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(exclude="number").columns.tolist()
date_cols = df.select_dtypes(include="datetime").columns.tolist()

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------
st.sidebar.header("📌 Visualization Controls")

viz_category = st.sidebar.selectbox(
    "Select Visualization Category",
    [
        "Distribution",
        "Categorical",
        "Correlation",
        "Missing Data",
        "Outliers",
        "Time-Series"
    ]
)

st.sidebar.markdown("---")

# ==================================================
# 🅰️ DISTRIBUTION VISUALS
# ==================================================
if viz_category == "Distribution":

    st.subheader("🅰 Distribution Visualizations")

    col = st.selectbox("Select numeric column", num_cols)

    viz_type = st.selectbox(
        "Select visualization",
        ["Histogram", "KDE", "Box Plot", "Violin Plot", "CDF", "Q-Q Plot"]
    )

    if st.button("Generate Visualization"):
        if viz_type == "Histogram":
            st.pyplot(histogram(df, col))

        elif viz_type == "KDE":
            st.pyplot(kde_plot(df, col))

        elif viz_type == "Box Plot":
            st.pyplot(box_plot(df, col))

        elif viz_type == "Violin Plot":
            st.pyplot(violin_plot(df, col))

        elif viz_type == "CDF":
            st.pyplot(cdf_plot(df, col))

        elif viz_type == "Q-Q Plot":
            st.pyplot(qq_plot(df, col))


# ==================================================
# 🅱️ CATEGORICAL VISUALS
# ==================================================
elif viz_category == "Categorical":

    st.subheader("🅱 Categorical Visualizations")

    col = st.selectbox("Select categorical column", cat_cols)

    viz_type = st.selectbox(
        "Select visualization",
        ["Frequency Bar", "Percentage Bar", "Dominant Category", "Cross-Tab Heatmap"]
    )

    if viz_type == "Cross-Tab Heatmap":
        col2 = st.selectbox("Select second categorical column", cat_cols)

    if st.button("Generate Visualization"):
        if viz_type == "Frequency Bar":
            st.pyplot(frequency_bar(df, col))

        elif viz_type == "Percentage Bar":
            st.pyplot(percentage_bar(df, col))

        elif viz_type == "Dominant Category":
            st.pyplot(dominant_category_plot(df, col))

        elif viz_type == "Cross-Tab Heatmap":
            st.pyplot(crosstab_heatmap(df, col, col2))


# ==================================================
# 🅲 CORRELATION VISUALS
# ==================================================
elif viz_category == "Correlation":

    st.subheader("🅲 Correlation Visualizations")

    viz_type = st.selectbox(
        "Select visualization",
        ["Correlation Heatmap", "Scatter Plot", "Pair Plot", "Hexbin Plot"]
    )

    if viz_type == "Correlation Heatmap":
        method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"])

    if viz_type in ["Scatter Plot", "Hexbin Plot"]:
        x_col = st.selectbox("X-axis", num_cols)
        y_col = st.selectbox("Y-axis", num_cols)

    if viz_type == "Pair Plot":
        cols = st.multiselect("Select numeric columns", num_cols)

    if st.button("Generate Visualization"):
        if viz_type == "Correlation Heatmap":
            st.pyplot(correlation_heatmap(df, method))

        elif viz_type == "Scatter Plot":
            st.pyplot(scatter_plot(df, x_col, y_col))

        elif viz_type == "Pair Plot" and cols:
            st.pyplot(pair_plot(df, cols))

        elif viz_type == "Hexbin Plot":
            st.pyplot(hexbin_plot(df, x_col, y_col))


# ==================================================
# 🅳 MISSING DATA VISUALS
# ==================================================
elif viz_category == "Missing Data":

    st.subheader("🅳 Missing Data Visualizations")

    viz_type = st.selectbox(
        "Select visualization",
        ["Missing Heatmap", "Missing Percentage Bar"]
    )

    if st.button("Generate Visualization"):
        if viz_type == "Missing Heatmap":
            st.pyplot(missing_heatmap(df))

        elif viz_type == "Missing Percentage Bar":
            st.pyplot(missing_percentage_bar(df))


# ==================================================
# 🅴 OUTLIER VISUALS
# ==================================================
elif viz_category == "Outliers":

    st.subheader("🅴 Outlier Visualizations")

    col = st.selectbox("Select numeric column", num_cols)

    viz_type = st.selectbox(
        "Select visualization",
        ["Boxplot with Outliers", "Z-Score Plot"]
    )

    if viz_type == "Z-Score Plot":
        threshold = st.slider("Z-score threshold", 2.0, 5.0, 3.0)

    if st.button("Generate Visualization"):
        if viz_type == "Boxplot with Outliers":
            st.pyplot(boxplot_outliers(df, col))

        elif viz_type == "Z-Score Plot":
            st.pyplot(zscore_outlier_plot(df, col, threshold))


# ==================================================
# 🅵 TIME-SERIES VISUALS
# ==================================================
elif viz_category == "Time-Series":

    st.subheader("🅵 Time-Series Visualizations")

    if not date_cols:
        st.warning("No datetime columns detected.")
    else:
        date_col = st.selectbox("Select date column", date_cols)
        target_col = st.selectbox("Select numeric target", num_cols)

        viz_type = st.selectbox(
            "Select visualization",
            ["Time-Series Line", "Rolling Mean", "Lag Plot"]
        )

        if viz_type == "Rolling Mean":
            window = st.slider("Window size", 2, 30, 7)

        if viz_type == "Lag Plot":
            lag = st.slider("Lag value", 1, 20, 1)

        if st.button("Generate Visualization"):
            if viz_type == "Time-Series Line":
                st.pyplot(timeseries_line(df, date_col, target_col))

            elif viz_type == "Rolling Mean":
                st.pyplot(rolling_mean_plot(df, date_col, target_col, window))

            elif viz_type == "Lag Plot":
                st.pyplot(lag_plot(df, target_col, lag))
