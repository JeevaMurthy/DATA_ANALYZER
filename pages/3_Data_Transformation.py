import streamlit as st
import pandas as pd

# --------------------------------------------------
# IMPORT BACKEND MODULES
# --------------------------------------------------
from modules.transformation.type_conversion import *
from modules.transformation.encoding import *
from modules.transformation.scaling import *
from modules.transformation.feature_engineering import *
from modules.transformation.binning import *
from modules.transformation.aggregation import *

# --------------------------------------------------
# LOAD THEME
# --------------------------------------------------
def load_css():
    with open("assets/theme.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Data Transformation",
    page_icon="🔄",
    layout="wide"
)

st.title("🔄 Data Transformation")
st.caption("Step 3: Convert, Encode, Scale, Engineer & Aggregate Data")

st.markdown("---")

# --------------------------------------------------
# CHECK DATA AVAILABILITY
# --------------------------------------------------
if "df" not in st.session_state:
    st.warning("⚠️ Please upload and clean the dataset first.")
    st.stop()

df = st.session_state["df"]

# --------------------------------------------------
# TRANSFORMATION CATEGORY
# --------------------------------------------------
category = st.selectbox(
    "Select Transformation Category",
    [
        "Step 1: Data Type Conversion",
        "Step 2: Encoding",
        "Step 3: Scaling & Normalization",
        "Step 4: Feature Engineering",
        "Step 5: Binning / Discretization",
        "Step 6: Aggregation & Summarization"
    ]
)

st.markdown("---")

# ==================================================
# STEP 1: DATA TYPE CONVERSION
# ==================================================
if category == "Step 1: Data Type Conversion":
    col = st.selectbox("Select column", df.columns)

    method = st.selectbox(
        "Conversion Technique",
        [
            "String → Numeric",
            "String → DateTime",
            "Numeric → String",
            "Boolean Conversion",
            "Convert to Category"
        ]
    )

    if st.button("Apply Conversion"):
        if method == "String → Numeric":
            df = string_to_numeric(df, col)
        elif method == "String → DateTime":
            df = string_to_datetime(df, col)
        elif method == "Numeric → String":
            df = numeric_to_string(df, col)
        elif method == "Boolean Conversion":
            df = to_boolean(df, col)
        elif method == "Convert to Category":
            df = to_category(df, col)

        st.success("✅ Conversion applied successfully")

# ==================================================
# STEP 2: ENCODING
# ==================================================
elif category == "Step 2: Encoding":
    col = st.selectbox("Select categorical column", df.columns)

    method = st.selectbox(
        "Encoding Technique",
        [
            "Label Encoding",
            "One-Hot Encoding",
            "Ordinal Encoding",
            "Binary Encoding",
            "Frequency Encoding",
            "Target Encoding"
        ]
    )

    if method == "Ordinal Encoding":
        categories = st.text_input(
            "Enter ordered categories (comma-separated)",
            "Low,Medium,High"
        ).split(",")

    if method == "Target Encoding":
        target_col = st.selectbox("Select target column", df.columns)

    if st.button("Apply Encoding"):
        if method == "Label Encoding":
            df = label_encoding(df, col)
        elif method == "One-Hot Encoding":
            df = one_hot_encoding(df, col)
        elif method == "Ordinal Encoding":
            df = ordinal_encoding(df, col, categories)
        elif method == "Binary Encoding":
            df = binary_encoding(df, col)
        elif method == "Frequency Encoding":
            df = frequency_encoding(df, col)
        elif method == "Target Encoding":
            df = target_encoding(df, col, target_col)

        st.success("✅ Encoding applied successfully")

# ==================================================
# STEP 3: SCALING & NORMALIZATION
# ==================================================
elif category == "Step 3: Scaling & Normalization":
    numeric_cols = df.select_dtypes(include="number").columns

    if len(numeric_cols) == 0:
        st.warning("No numeric columns available.")
    else:
        col = st.selectbox("Select numeric column", numeric_cols)

        method = st.selectbox(
            "Scaling Technique",
            [
                "Min–Max Scaling",
                "Standardization (Z-score)",
                "Robust Scaling",
                "Unit Vector Scaling",
                "Log Transformation",
                "Box–Cox Transformation",
                "Yeo–Johnson Transformation"
            ]
        )

        if st.button("Apply Scaling"):
            if method == "Min–Max Scaling":
                df = minmax_scaling(df, col)
            elif method == "Standardization (Z-score)":
                df = standard_scaling(df, col)
            elif method == "Robust Scaling":
                df = robust_scaling(df, col)
            elif method == "Unit Vector Scaling":
                df = unit_vector_scaling(df, col)
            elif method == "Log Transformation":
                df = log_transform(df, col)
            elif method == "Box–Cox Transformation":
                df = boxcox_transform(df, col)
            elif method == "Yeo–Johnson Transformation":
                df = yeojohnson_transform(df, col)

            st.success("✅ Scaling applied successfully")

# ==================================================
# STEP 4: FEATURE ENGINEERING
# ==================================================
elif category == "Step 4: Feature Engineering":
    fe_method = st.selectbox(
        "Feature Engineering Technique",
        [
            "Arithmetic Feature",
            "Polynomial Features",
            "Date Feature Extraction",
            "Rolling Feature",
            "Lag Feature",
            "Aggregated Feature",
            "Domain Feature (BMI / Profit)"
        ]
    )

    if fe_method == "Arithmetic Feature":
        c1 = st.selectbox("Select first column", df.columns)
        c2 = st.selectbox("Select second column", df.columns)
        new_col = st.text_input("New feature name", "new_feature")

        if st.button("Create Feature"):
            df = add_product_feature(df, c1, c2, new_col)
            st.success("✅ Arithmetic feature created")

    elif fe_method == "Polynomial Features":
        cols = st.multiselect("Select columns", df.select_dtypes(include="number").columns)
        degree = st.slider("Degree", 2, 4, 2)

        if st.button("Apply Polynomial Features"):
            df = polynomial_features(df, cols, degree)
            st.success("✅ Polynomial features created")

    elif fe_method == "Date Feature Extraction":
        col = st.selectbox("Select datetime column", df.columns)

        if st.button("Extract Date Features"):
            df = extract_date_features(df, col)
            st.success("✅ Date features extracted")

    elif fe_method == "Rolling Feature":
        col = st.selectbox("Select column", df.columns)
        window = st.number_input("Window size", 2, 30, 3)
        new_col = st.text_input("New feature name", "rolling_mean")

        if st.button("Apply Rolling Mean"):
            df = rolling_mean(df, col, window, new_col)
            st.success("✅ Rolling feature created")

    elif fe_method == "Lag Feature":
        col = st.selectbox("Select column", df.columns)
        lag = st.number_input("Lag value", 1, 10, 1)
        new_col = st.text_input("New feature name", "lag_feature")

        if st.button("Apply Lag"):
            df = lag_feature(df, col, lag, new_col)
            st.success("✅ Lag feature created")

    elif fe_method == "Aggregated Feature":
        gcol = st.selectbox("Group column", df.columns)
        tcol = st.selectbox("Target column", df.columns)
        func = st.selectbox("Aggregation function", ["mean", "sum", "count"])
        new_col = st.text_input("New feature name", "agg_feature")

        if st.button("Apply Aggregation Feature"):
            df = aggregated_feature(df, gcol, tcol, func, new_col)
            st.success("✅ Aggregated feature created")

    elif fe_method == "Domain Feature (BMI / Profit)":
        option = st.selectbox("Select domain feature", ["BMI", "Profit"])

        if option == "BMI":
            w = st.selectbox("Weight column", df.columns)
            h = st.selectbox("Height column", df.columns)

            if st.button("Create BMI"):
                df = bmi_feature(df, w, h)
                st.success("✅ BMI feature created")

        else:
            r = st.selectbox("Revenue column", df.columns)
            c = st.selectbox("Cost column", df.columns)

            if st.button("Create Profit"):
                df = profit_feature(df, r, c)
                st.success("✅ Profit feature created")

# ==================================================
# STEP 5: BINNING
# ==================================================
elif category == "Step 5: Binning / Discretization":
    col = st.selectbox("Select numeric column", df.select_dtypes(include="number").columns)

    method = st.selectbox(
        "Binning Technique",
        [
            "Equal Width",
            "Quantile",
            "Custom",
            "K-Means",
            "Decision Tree"
        ]
    )

    if method == "Equal Width":
        bins = st.number_input("Number of bins", 2, 10, 4)
        if st.button("Apply Binning"):
            df = equal_width_binning(df, col, bins)
            st.success("✅ Binning applied")

    elif method == "Quantile":
        q = st.number_input("Number of quantiles", 2, 10, 4)
        if st.button("Apply Quantile Binning"):
            df = quantile_binning(df, col, q)
            st.success("✅ Quantile binning applied")

    elif method == "Custom":
        edges = st.text_input("Bin edges (comma-separated)", "0,18,35,60,100")
        labels = st.text_input("Labels (comma-separated)", "Child,Young,Adult,Senior")

        if st.button("Apply Custom Binning"):
            df = custom_binning(
                df,
                col,
                list(map(float, edges.split(","))),
                labels.split(",")
            )
            st.success("✅ Custom binning applied")

    elif method == "K-Means":
        k = st.number_input("Number of clusters", 2, 10, 3)
        if st.button("Apply K-Means Binning"):
            df = kmeans_binning(df, col, k)
            st.success("✅ K-Means binning applied")

    elif method == "Decision Tree":
        target = st.selectbox("Target column", df.columns)
        if st.button("Apply Decision Tree Binning"):
            df = decision_tree_binning(df, col, target)
            st.success("✅ Decision Tree binning applied")

# ==================================================
# STEP 6: AGGREGATION
# ==================================================
elif category == "Step 6: Aggregation & Summarization":
    gcol = st.selectbox("Group by column", df.columns)
    tcol = st.selectbox("Target column", df.columns)
    func = st.selectbox("Aggregation function", available_aggregations())

    if st.button("Apply Aggregation"):
        agg_df = groupby_aggregate(df, gcol, tcol, func)
        st.dataframe(agg_df, use_container_width=True)
        st.stop()

# --------------------------------------------------
# SAVE & PREVIEW
# --------------------------------------------------
st.session_state["df"] = df

st.markdown("---")
st.markdown("### 🔍 Transformed Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

st.info("ℹ️ You can repeat transformations multiple times before moving to EDA.")
