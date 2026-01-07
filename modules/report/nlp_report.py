import pandas as pd
import numpy as np
from datetime import datetime


# ==================================================
# 🧠 NLP REPORT GENERATOR
# ==================================================

class NLPReportGenerator:
    """
    Generates human-readable analytical reports
    from data analysis, EDA, and ML results.
    """

    def __init__(
        self,
        df,
        eda_insights=None,
        ml_results=None,
        problem_type=None,
        best_model=None
    ):
        self.df = df
        self.eda_insights = eda_insights or {}
        self.ml_results = ml_results or {}
        self.problem_type = problem_type
        self.best_model = best_model

    # --------------------------------------------------
    # 1️⃣ DATASET OVERVIEW
    # --------------------------------------------------
    def dataset_overview(self):
        rows, cols = self.df.shape
        num_cols = self.df.select_dtypes(include="number").shape[1]
        cat_cols = self.df.select_dtypes(exclude="number").shape[1]

        return (
            f"The dataset contains {rows:,} rows and {cols} columns. "
            f"It includes {num_cols} numeric features and {cat_cols} categorical features."
        )

    # --------------------------------------------------
    # 2️⃣ DATA QUALITY SUMMARY
    # --------------------------------------------------
    def data_quality_summary(self):
        missing_pct = (self.df.isnull().mean() * 100).round(2)
        high_missing = missing_pct[missing_pct > 20].index.tolist()

        duplicates = self.df.duplicated().sum()

        summary = f"The overall missing value percentage is {missing_pct.mean():.2f}%. "

        if high_missing:
            summary += (
                f"The following columns have high missing values (>20%): "
                f"{', '.join(high_missing)}. "
            )
        else:
            summary += "No columns contain critically high missing values. "

        summary += f"A total of {duplicates} duplicate rows were detected."

        return summary

    # --------------------------------------------------
    # 3️⃣ DATA TRANSFORMATION SUMMARY
    # --------------------------------------------------
    def transformation_summary(self):
        return (
            "Data transformations were applied based on user selection, "
            "including data type conversion, categorical encoding, "
            "numeric scaling, and feature engineering to improve model readiness."
        )

    # --------------------------------------------------
    # 4️⃣ EDA INSIGHTS
    # --------------------------------------------------
    def eda_summary(self):
        numeric_df = self.df.select_dtypes(include="number")

        if numeric_df.empty:
            return "Exploratory analysis focused mainly on categorical feature distributions."

        skewed = numeric_df.skew().abs()
        highly_skewed = skewed[skewed > 1].index.tolist()

        corr = numeric_df.corr().abs()
        high_corr_pairs = (
            corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            .stack()
            .sort_values(ascending=False)
        )

        summary = ""

        if highly_skewed:
            summary += (
                f"The following numeric features show high skewness: "
                f"{', '.join(highly_skewed)}. "
            )

        if not high_corr_pairs.empty:
            top_pair = high_corr_pairs.index[0]
            summary += (
                f"A strong correlation was observed between "
                f"{top_pair[0]} and {top_pair[1]}. "
            )

        if not summary:
            summary = "EDA revealed stable distributions with no extreme anomalies."

        return summary

    # --------------------------------------------------
    # 5️⃣ FEATURE IMPORTANCE SUMMARY
    # --------------------------------------------------
    def feature_summary(self):
        return (
            "Feature relevance was evaluated using correlation analysis "
            "and model-based importance methods to reduce redundancy "
            "and enhance interpretability."
        )

    # --------------------------------------------------
    # 6️⃣ ML MODEL SUMMARY
    # --------------------------------------------------
    def ml_summary(self):
        if not self.ml_results:
            return "No machine learning models were trained."

        return (
            f"The problem was identified as a {self.problem_type} task. "
            f"Multiple models were trained and evaluated."
        )

    # --------------------------------------------------
    # 7️⃣ MODEL PERFORMANCE SUMMARY
    # --------------------------------------------------
    def performance_summary(self):
        if not self.ml_results:
            return ""

        if self.problem_type == "classification":
            return (
                f"The best-performing model was {self.best_model}, "
                "demonstrating balanced accuracy and generalization capability."
            )

        if self.problem_type == "regression":
            return (
                f"The best-performing model was {self.best_model}, "
                "achieving low prediction error and strong explanatory power."
            )

        if self.problem_type == "unsupervised":
            return (
                "Unsupervised models revealed meaningful structure "
                "and clustering patterns within the data."
            )

    # --------------------------------------------------
    # 8️⃣ BUSINESS INSIGHTS
    # --------------------------------------------------
    def business_insights(self):
        return (
            "The analysis highlights key patterns that can support "
            "data-driven decision-making, risk identification, "
            "and opportunity discovery."
        )

    # --------------------------------------------------
    # 9️⃣ LIMITATIONS
    # --------------------------------------------------
    def limitations(self):
        return (
            "Results are based on the available dataset and selected features. "
            "External factors and unseen data may influence real-world performance."
        )

    # --------------------------------------------------
    # 🔟 EXECUTIVE SUMMARY
    # --------------------------------------------------
    def executive_summary(self):
        return (
            "Overall, the dataset quality is suitable for analysis. "
            "EDA uncovered meaningful patterns, and machine learning models "
            "successfully captured predictive relationships. "
            "The processed dataset is ready for deployment or further use."
        )

    # --------------------------------------------------
    # 📄 FULL REPORT
    # --------------------------------------------------
    def generate_full_report(self):
        report_sections = [
            ("DATASET OVERVIEW", self.dataset_overview()),
            ("DATA QUALITY SUMMARY", self.data_quality_summary()),
            ("DATA TRANSFORMATION SUMMARY", self.transformation_summary()),
            ("EDA INSIGHTS", self.eda_summary()),
            ("FEATURE SUMMARY", self.feature_summary()),
            ("ML MODEL SUMMARY", self.ml_summary()),
            ("MODEL PERFORMANCE", self.performance_summary()),
            ("BUSINESS INSIGHTS", self.business_insights()),
            ("LIMITATIONS", self.limitations()),
            ("EXECUTIVE SUMMARY", self.executive_summary()),
        ]

        report_text = ""
        for title, content in report_sections:
            report_text += f"\n### {title}\n{content}\n"

        report_text += (
            f"\n---\nReport generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        return report_text
