import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def train_churn_model():
  # 1. Cargar dataset público real de abandono de clientes
  url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
  df = pd.read_csv(url)

  # 2. Limpieza básica y selección de variables clave
  df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
  df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
  df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

  num_features = ["tenure", "MonthlyCharges", "TotalCharges"]
  cat_features = [
      "Contract",
      "InternetService",
      "PaymentMethod",
      "TechSupport",
      "OnlineSecurity",
  ]

  X = df[num_features + cat_features]
  y = df["Churn"]

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )

  # 3. Preprocesamiento + Modelo en un único Pipeline
  preprocessor = ColumnTransformer(
      transformers=[
          ("num", StandardScaler(), num_features),
          ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
      ]
  )

  model = Pipeline(
      steps=[
          ("preprocessor", preprocessor),
          (
              "classifier",
              RandomForestClassifier(
                  n_estimators=150,
                  max_depth=8,
                  random_state=42,
                  class_weight="balanced",
              ),
          ),
      ]
  )

  # 4. Entrenamiento y evaluación
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  y_proba = model.predict_proba(X_test)[:, 1]

  auc = roc_auc_score(y_test, y_proba)
  print(f"ROC-AUC Score: {auc:.4f}")
  print("\nClassification Report:\n", classification_report(y_test, y_pred))

  # 5. Guardar el modelo entrenado
  os.makedirs("models", exist_ok=True)
  joblib.dump(model, "models/churn_model.joblib")
  print("Modelo guardado correctamente en models/churn_model.joblib")


if __name__ == "__main__":
  train_churn_model()
