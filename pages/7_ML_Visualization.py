import streamlit as st
import pandas as pd
import numpy as np

from modules.visualization.ml_visuals import *


def load_css():
    with open("assets/theme.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="ML Visualization",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Machine Learning Visualization")
st.caption("Model performance, diagnostics, and interpretability")

st.markdown("---")

# --------------------------------------------------
# CHECK TRAINED RESULTS
# --------------------------------------------------
if "ml_results" not in st.session_state:
    st.warning("⚠️ Please train models in Page 6 first.")
    st.stop()

trained_models = st.session_state["ml_trained"]
problem_type = st.session_state["ml_problem_type"]


model_names = list(trained_models.keys())

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------
st.sidebar.header("📌 ML Visualization Controls")

selected_model = st.sidebar.selectbox(
    "Select Model",
    model_names
)

viz_type = None

# ==================================================
# CLASSIFICATION VISUALS
# ==================================================
if problem_type == "classification":

    viz_type = st.sidebar.selectbox(
        "Select Visualization",
        [
            "Confusion Matrix",
            "Normalized Confusion Matrix",
            "ROC Curve",
            "Precision–Recall Curve",
            "Probability Distribution",
            "Misclassification Plot"
        ]
    )

    model_data = trained_models[selected_model]
    y_true = model_data["y_test"]
    y_pred = model_data["y_pred"]

    y_prob = None
    try:
        y_prob = model_data["model"].predict_proba(model_data["X_test"])
    except:
        pass

    if st.button("Generate Visualization"):
        if viz_type == "Confusion Matrix":
            st.pyplot(confusion_matrix_plot(y_true, y_pred))

        elif viz_type == "Normalized Confusion Matrix":
            st.pyplot(normalized_confusion_matrix_plot(y_true, y_pred))

        elif viz_type == "ROC Curve" and y_prob is not None:
            st.pyplot(roc_curve_plot(y_true, y_prob))

        elif viz_type == "Precision–Recall Curve" and y_prob is not None:
            st.pyplot(precision_recall_plot(y_true, y_prob))

        elif viz_type == "Probability Distribution" and y_prob is not None:
            st.pyplot(class_probability_distribution(y_prob))

        elif viz_type == "Misclassification Plot":
            st.pyplot(misclassification_plot(y_true, y_pred))

        else:
            st.warning("⚠️ Selected model does not support this visualization.")


# ==================================================
# REGRESSION VISUALS
# ==================================================
elif problem_type == "regression":

    viz_type = st.sidebar.selectbox(
        "Select Visualization",
        [
            "Actual vs Predicted",
            "Residual Plot",
            "Residual Distribution",
            "Residual Q-Q Plot"
        ]
    )

    model_data = trained_models[selected_model]
    y_true = model_data["y_test"]
    y_pred = model_data["y_pred"]

    if st.button("Generate Visualization"):
        if viz_type == "Actual vs Predicted":
            st.pyplot(actual_vs_predicted(y_true, y_pred))

        elif viz_type == "Residual Plot":
            st.pyplot(residual_plot(y_true, y_pred))

        elif viz_type == "Residual Distribution":
            st.pyplot(residual_distribution(y_true, y_pred))

        elif viz_type == "Residual Q-Q Plot":
            st.pyplot(residual_qq_plot(y_true, y_pred))


# ==================================================
# UNSUPERVISED VISUALS
# ==================================================
elif problem_type == "unsupervised":

    viz_type = st.sidebar.selectbox(
        "Select Visualization",
        [
            "PCA Explained Variance",
            "PCA Scatter Plot",
            "Cluster Assignment Plot",
            "Elbow Method"
        ]
    )

    model_data = trained_models[selected_model]
    labels = model_data["labels"]

    X = pd.get_dummies(st.session_state["df"]).values

    if st.button("Generate Visualization"):
        if viz_type == "PCA Explained Variance":
            st.pyplot(pca_explained_variance(X))

        elif viz_type == "PCA Scatter Plot":
            st.pyplot(pca_scatter(X, labels))

        elif viz_type == "Cluster Assignment Plot":
            st.pyplot(cluster_assignment_plot(X[:, :2], labels))

        elif viz_type == "Elbow Method":
            st.pyplot(elbow_method_plot(X))


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption(
    "ℹ️ ML visualizations help diagnose model performance, "
    "errors, interpretability, and structure."
)
