import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ---------------- CLASSIFICATION MODELS ----------------
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.svm import SVC

# ---------------- REGRESSION MODELS ----------------
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# ---------------- UNSUPERVISED MODELS ----------------
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


# ==================================================
# 🔍 PROBLEM TYPE DETECTION
# ==================================================
def detect_problem_type(df, target_col=None):
    """
    Automatically detects ML problem type.
    """
    if target_col is None:
        return "unsupervised"

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        return "classification"

    unique_vals = df[target_col].nunique()
    return "classification" if unique_vals <= 20 else "regression"


# ==================================================
# 🧠 MODEL REGISTRY
# ==================================================
def get_model_registry(problem_type):
    """
    Returns available models based on problem type.
    """
    if problem_type == "classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC(probability=True),
            "Gradient Boosting": GradientBoostingClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "Extra Trees": ExtraTreesClassifier(),
            "Ridge Classifier": RidgeClassifier(),
            "SGD Classifier": SGDClassifier()
        }

    if problem_type == "regression":
        return {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "ElasticNet": ElasticNet(),
            "KNN Regressor": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "SVR": SVR(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "AdaBoost": AdaBoostRegressor(),
            "Extra Trees": ExtraTreesRegressor()
        }

    if problem_type == "unsupervised":
        return {
            "KMeans": KMeans(),
            "DBSCAN": DBSCAN(),
            "Hierarchical Clustering": AgglomerativeClustering(),
            "PCA": PCA(n_components=2),
            "Isolation Forest": IsolationForest(),
            "One-Class SVM": OneClassSVM()
        }


# ==================================================
# ⚙️ TRAIN SUPERVISED MODELS
# ==================================================
def train_supervised_models(
    df,
    target_col,
    selected_models,
    test_size=0.2,
    scale_data=True,
    random_state=42
):
    """
    Trains classification or regression models.
    Returns trained models and predictions.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() <= 20 else None
    )

    results = {}

    for name, model in selected_models.items():
        steps = []

        if scale_data:
            steps.append(("scaler", StandardScaler()))

        steps.append(("model", model))
        pipeline = Pipeline(steps)

        pipeline.fit(X_train, y_train)

        results[name] = {
            "model": pipeline,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": pipeline.predict(X_test)
        }

    return results


# ==================================================
# ⚙️ TRAIN UNSUPERVISED MODELS
# ==================================================
def train_unsupervised_models(df, selected_models):
    """
    Trains clustering / anomaly detection models.
    """
    X = pd.get_dummies(df)
    X = StandardScaler().fit_transform(X)

    results = {}

    for name, model in selected_models.items():
        labels = model.fit_predict(X)
        results[name] = {
            "model": model,
            "labels": labels
        }

    return results
