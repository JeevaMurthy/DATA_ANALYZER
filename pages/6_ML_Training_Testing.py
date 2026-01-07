import streamlit as st
import pandas as pd

from modules.ml.trainer import (
    detect_problem_type,
    get_model_registry,
    train_supervised_models,
    train_unsupervised_models
)
from modules.ml.evaluator import (
    evaluate_classification,
    evaluate_regression,
    evaluate_unsupervised,
    build_comparison_table
)


def load_css():
    with open("assets/theme.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="ML Training & Testing",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Machine Learning – Training & Testing")
st.caption("Hybrid approach: Auto-detect problem type + user-controlled model selection")

st.markdown("---")

# --------------------------------------------------
# CHECK DATA
# --------------------------------------------------
if "df" not in st.session_state:
    st.warning("⚠️ Please complete previous steps (Upload, Cleaning, Transformation).")
    st.stop()

df = st.session_state["df"]

# --------------------------------------------------
# TARGET SELECTION
# --------------------------------------------------
st.markdown("## 🎯 Step 1: Select Target Variable")

target_col = st.selectbox(
    "Select target column (leave empty for unsupervised learning)",
    ["None"] + df.columns.tolist()
)

target_col = None if target_col == "None" else target_col

# --------------------------------------------------
# AUTO-DETECT PROBLEM TYPE
# --------------------------------------------------
problem_type = detect_problem_type(df, target_col)

st.info(f"🔍 Detected Problem Type: **{problem_type.upper()}**")

# --------------------------------------------------
# MODEL SELECTION (HYBRID)
# --------------------------------------------------
st.markdown("## 🧠 Step 2: Select Models")

model_registry = get_model_registry(problem_type)

default_models = list(model_registry.keys())[:3]

selected_model_names = st.multiselect(
    "Choose ML models (you may select multiple)",
    options=list(model_registry.keys()),
    default=default_models
)

selected_models = {
    name: model_registry[name]
    for name in selected_model_names
}

if not selected_models:
    st.warning("⚠️ Please select at least one model.")
    st.stop()

# --------------------------------------------------
# TRAINING OPTIONS
# --------------------------------------------------
st.markdown("## ⚙️ Step 3: Training Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

with col2:
    scale_data = st.checkbox("Apply Feature Scaling", value=True)

with col3:
    random_state = st.number_input("Random State", value=42)

# --------------------------------------------------
# TRAIN MODELS
# --------------------------------------------------
if st.button("▶️ Train Selected Models"):

    with st.spinner("Training models... Please wait"):
        evaluation_results = {}

        # ---------------- SUPERVISED ----------------
        if problem_type in ["classification", "regression"]:
            trained = train_supervised_models(
                df=df,
                target_col=target_col,
                selected_models=selected_models,
                test_size=test_size,
                scale_data=scale_data,
                random_state=random_state
            )

            for model_name, obj in trained.items():
                y_test = obj["y_test"]
                y_pred = obj["y_pred"]

                if problem_type == "classification":
                    y_prob = None
                    try:
                        y_prob = obj["model"].predict_proba(obj["X_test"])
                    except:
                        pass

                    evaluation_results[model_name] = evaluate_classification(
                        y_test, y_pred, y_prob
                    )

                else:  # regression
                    evaluation_results[model_name] = evaluate_regression(
                        y_test, y_pred
                    )

        # ---------------- UNSUPERVISED ----------------
        else:
            trained = train_unsupervised_models(df, selected_models)

            for model_name, obj in trained.items():
                evaluation_results[model_name] = evaluate_unsupervised(
                    df.select_dtypes(include="number"),
                    obj["labels"]
                )

        # --------------------------------------------------
        # SAVE RESULTS FOR NEXT PAGE
        # --------------------------------------------------
        st.session_state["ml_trained"] = trained        # for visualization
        st.session_state["ml_results"] = evaluation_results  # for metrics
        st.session_state["ml_problem_type"] = problem_type


        # --------------------------------------------------
        # DISPLAY RESULTS
        # --------------------------------------------------
        st.success("✅ Model training completed!")

        st.markdown("## 📊 Model Comparison")

        comparison_df = build_comparison_table(
            evaluation_results, problem_type
        )

        st.dataframe(comparison_df, use_container_width=True)

        # Highlight best model (simple heuristic)
        if problem_type == "classification" and "Accuracy" in comparison_df.columns:
            best = comparison_df.sort_values("Accuracy", ascending=False).iloc[0]
            st.success(f"🏆 Best Model: **{best['Model']}**")

        if problem_type == "regression" and "RMSE" in comparison_df.columns:
            best = comparison_df.sort_values("RMSE").iloc[0]
            st.success(f"🏆 Best Model: **{best['Model']}**")

        st.info("➡️ Proceed to **Page 7 – ML Visualization** for detailed analysis.")
