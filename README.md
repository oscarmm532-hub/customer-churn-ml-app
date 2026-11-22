# Customer Churn Prediction App

An end-to-end Machine Learning classification pipeline and interactive web application built with **Scikit-Learn** and **Streamlit** to predict customer churn risk in real time.

🚀 **Live Demo:** [Try the Interactive App Here](TU_ENLACE_DE_STREAMLIT)

## Architecture
`Telco Customer Dataset` ➔ `Scikit-Learn Pipeline (StandardScaler + OneHotEncoder + RandomForest)` ➔ `Joblib Model` ➔ `Streamlit Web App`

## Project Structure
* `src/train.py` — Loads data, handles missing values, trains a balanced Random Forest pipeline, evaluates ROC-AUC, and exports the model.
* `app.py` — Interactive Streamlit web interface for real-time churn probability inference.
* `requirements.txt` — Project dependencies.

## Model Details
* **Algorithm:** `RandomForestClassifier` (`class_weight="balanced"`) inside a Scikit-Learn `ColumnTransformer` pipeline.
* **Key Features:** Tenure, Monthly Charges, Total Charges, Contract Type, Internet Service, Payment Method, Tech Support, and Online Security[cite: 8].
* **Evaluation Metric:** Evaluated using **ROC-AUC** and **F1-Score** on an 80/20 stratified train-test split.
