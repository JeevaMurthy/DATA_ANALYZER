# 📊 Data Analyzer – End-to-End Data Analysis & ML Tool

## 🔍 Project Overview

**Data Analyzer** is a complete, interactive data analysis and machine learning platform built using **Python and Streamlit**.  
It allows users to perform the **entire data science pipeline** — from dataset upload to **machine learning insights and NLP-based reporting** — without writing code.

This project is designed as a **main academic / career project** and demonstrates real-world data analyst workflows.

---

## 🚀 Key Features

- Upload and preview datasets (CSV / Excel)
- Data cleaning & missing value handling
- Data transformation & feature engineering
- Exploratory Data Analysis (EDA)
- 20+ EDA visualizations
- 30+ Machine Learning models
- Hybrid ML model selection (auto-detect + user override)
- 25+ ML performance visualizations
- NLP-based automatic report generation
- Download processed dataset and report

---

## 🧱 Project Architecture

```
DataAnalyzer/
│
├── app.py
│
├── pages/
│ ├── 1_Upload_Data.py
│ ├── 2_Data_Cleaning.py
│ ├── 3_Data_Transformation.py
│ ├── 4_EDA.py
│ ├── 5_Visualization_EDA.py
│ ├── 6_ML_Training_Testing.py
│ ├── 7_ML_Visualization.py
│ └── 8_Report_Generation.py
│
├── modules/
│ ├── ingestion/
│ │ └── loader.py
│ │
│ ├── cleaning/
│ │ └── cleaner.py
│ │
│ ├── transformation/
│ │ ├── type_conversion.py
│ │ ├── encoding.py
│ │ ├── scaling.py
│ │ ├── feature_engineering.py
│ │ ├── binning.py
│ │ ├── aggregation.py
│ │ └── transformer.py
│ │
│ ├── eda/
│ │ ├── overview.py
│ │ ├── descriptive.py
│ │ ├── distribution.py
│ │ ├── missing.py
│ │ ├── outliers.py
│ │ ├── correlation.py
│ │ ├── advanced.py
│ │ └── summary.py
│ │
│ ├── visualization/
│ │ ├── eda_visuals.py
│ │ └── ml_visuals.py
│ │
│ ├── ml/
│ │ ├── trainer.py
│ │ └── evaluator.py
│ │
│ └── report/
│ └── nlp_report.py
│
├── assets/
│ └── theme.css
│
├── outputs/
│ ├── plots/
│ └── reports/
│
├── requirements.txt
└── README.md

```


---

## 🧭 Application Workflow

1. Upload Dataset  
2. Data Cleaning & Missing Value Handling  
3. Data Transformation & Feature Engineering  
4. Exploratory Data Analysis (EDA)  
5. EDA Visualizations  
6. ML Training & Testing (Hybrid Selection)  
7. ML Performance Visualizations  
8. NLP-Based Report Generation  
9. Download Processed Dataset  

---

## 🧪 Supported Machine Learning Models

### 🔹 Classification
- Logistic Regression
- KNN
- Naive Bayes
- Decision Tree
- Random Forest
- SVM
- Gradient Boosting
- AdaBoost
- Extra Trees
- SGD Classifier
- Ridge Classifier

### 🔹 Regression
- Linear Regression
- Ridge, Lasso, ElasticNet
- Decision Tree Regressor
- Random Forest Regressor
- SVR
- Gradient Boosting Regressor
- AdaBoost Regressor
- Extra Trees Regressor
- KNN Regressor

### 🔹 Unsupervised Learning
- KMeans
- DBSCAN
- Hierarchical Clustering
- PCA
- Isolation Forest
- One-Class SVM

---

## 📊 Visualization Support

### EDA Visualizations (20+)
- Histogram, KDE, Box plot, Violin plot
- Correlation heatmaps
- Missing value heatmaps
- Outlier detection plots
- Scatter & pair plots
- Time-series plots

### ML Visualizations (25+)
- Confusion matrix
- ROC & Precision–Recall curves
- Residual plots
- Feature importance charts
- PCA visualizations
- Clustering plots
- Model comparison charts

---

## 🧠 NLP Report Highlights

The NLP report automatically generates:
- Dataset overview
- Data quality summary
- Transformation explanation
- EDA insights
- Feature importance interpretation
- ML model performance explanation
- Business insights
- Limitations & assumptions
- Executive summary

---

## 📥 Downloads

- Processed Dataset (CSV / Excel)
- NLP Analysis Report (TXT)

---

## 🛠️ Tech Stack

- Frontend: Streamlit  
- Backend: Python  
- Data Handling: Pandas, NumPy  
- Visualization: Matplotlib, Seaborn  
- Machine Learning: Scikit-learn  
- Reporting: Rule-based NLP  

---

## ▶️ How to Run the Project

### 1️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
Windows: venv\Scripts\activate
```
---

### 2️⃣ Install Dependencies
```
pip install -r requirements.txt
```

---

### 3️⃣ Run the Application
```
streamlit run app.py
```

---

