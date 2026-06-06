"""
train.py
========
Entrena un modelo de clasificación que predice si un hallazgo de compliance
es de riesgo ALTO (1) o BAJO (0), y registra TODO en MLflow.

CLAVE: el modelo es un **Pipeline de scikit-learn** que incluye DENTRO el
preprocesamiento (one-hot encoding) + el clasificador (XGBoost). Así el
preprocesamiento "viaja con el modelo" y al servir le podemos mandar datos
CRUDOS sin riesgo de 'training/serving skew' (que el encoding difiera entre
entrenamiento y producción).

Uso:
    uv run python src/train.py
Ver resultados:
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db  ->  http://localhost:5000
"""

import json                                 # para guardar las métricas en un archivo

import mlflow
import mlflow.sklearn                       # flavor para guardar Pipelines de sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

# ------------------------------------------------------------------
# Configuración
# ------------------------------------------------------------------
RUTA_DATOS = "data/compliance_findings.csv"
# Backend local de MLflow (SQLite). Fijarlo explícitamente garantiza que los
# experimentos, runs y el registry SIEMPRE queden en mlflow.db, y que la UI
# (mlflow ui --backend-store-uri sqlite:///mlflow.db) lea exactamente lo mismo.
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENTO = "compliance-risk"
NOMBRE_MODELO_REGISTRO = "compliance-risk-model"
SEED = 42

# Separamos las columnas por tipo: el Pipeline tratará distinto a cada grupo.
COLUMNAS_CATEGORICAS = ["framework", "department"]   # texto -> hay que codificar
COLUMNAS_NUMERICAS = [                                 # ya son números -> pasan tal cual
    "severity",
    "days_open",
    "control_failures",
    "affected_systems",
    "is_repeat_finding",
    "has_remediation_plan",
]
COLUMNA_OBJETIVO = "risk_high"


# ------------------------------------------------------------------
# 1) Cargar datos (CRUDOS: las categóricas quedan como texto)
# ------------------------------------------------------------------
def cargar_datos() -> tuple[pd.DataFrame, pd.Series]:
    """Lee el CSV y separa features (X, crudas) del objetivo (y).

    Ojo: aquí NO codificamos nada. El one-hot encoding lo hará el Pipeline,
    para que el mismo preprocesamiento se aplique tanto al entrenar como al servir.
    """
    df = pd.read_csv(RUTA_DATOS)
    X = df.drop(columns=[COLUMNA_OBJETIVO])   # incluye framework/department como texto
    y = df[COLUMNA_OBJETIVO]
    return X, y


# ------------------------------------------------------------------
# 2) Construir el Pipeline (preprocesamiento + modelo en un solo objeto)
# ------------------------------------------------------------------
def construir_pipeline(params: dict) -> Pipeline:
    """Crea un Pipeline = [ preprocesador -> clasificador ].

    - ColumnTransformer aplica OneHotEncoder SOLO a las categóricas y deja
      pasar las numéricas tal cual (remainder='passthrough').
    - handle_unknown='ignore': si en producción llega una categoría que no se
      vio al entrenar, no rompe (la codifica como todo ceros).
    """
    preprocesador = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), COLUMNAS_CATEGORICAS),
        ],
        remainder="passthrough",   # las numéricas pasan sin cambios
    )
    return Pipeline(
        steps=[
            ("preprocesador", preprocesador),
            ("clasificador", XGBClassifier(**params)),
        ]
    )


# ------------------------------------------------------------------
# 3) Entrenar + evaluar + registrar en MLflow
# ------------------------------------------------------------------
def main() -> None:
    X, y = cargar_datos()

    # 80% entrenar / 20% probar. stratify mantiene la proporción de clases.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    # Hiperparámetros del clasificador (las "perillas" del entrenamiento).
    params = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.9,
        "random_state": SEED,
        "eval_metric": "logloss",
    }

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENTO)

    with mlflow.start_run() as run:
        print(f"MLflow run id: {run.info.run_id}")

        # --- Entrenar el Pipeline COMPLETO (preprocesa + entrena de una) ---
        modelo = construir_pipeline(params)
        modelo.fit(X_train, y_train)

        # --- Predecir sobre el set de prueba (le pasamos datos CRUDOS) ---
        y_pred = modelo.predict(X_test)                  # clase 0/1
        y_proba = modelo.predict_proba(X_test)[:, 1]     # probabilidad de clase 1

        # --- Métricas ---
        metricas = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        # --- Registrar params y métricas ---
        mlflow.log_params(params)
        mlflow.log_metrics(metricas)

        # signature = contrato del modelo. Como entrenamos con X CRUDO, el
        # contrato dice que la API recibe columnas crudas (incluye texto).
        signature = infer_signature(X_train, modelo.predict(X_train))

        # --- Guardar el Pipeline y registrarlo en el Model Registry ---
        mlflow.sklearn.log_model(
            modelo,
            name="model",
            signature=signature,
            registered_model_name=NOMBRE_MODELO_REGISTRO,
        )

        # Guardamos las métricas en un archivo JSON. El quality gate del CI/CD
        # (src/check_quality.py) lo leerá para decidir si el modelo pasa o no.
        with open("metrics.json", "w") as f:
            json.dump(metricas, f, indent=2)

        print("\n=== Métricas del modelo ===")
        for nombre, valor in metricas.items():
            print(f"  {nombre:10s}: {valor:.4f}")
        print(f"\nModelo registrado como: '{NOMBRE_MODELO_REGISTRO}'")
        print("Métricas guardadas en: metrics.json")


if __name__ == "__main__":
    main()
