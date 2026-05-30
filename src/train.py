"""
train.py
========
Entrena un modelo de clasificación (XGBoost) que predice si un hallazgo de
compliance es de riesgo ALTO (1) o BAJO (0), y registra TODO en MLflow.

¿Qué es MLflow y por qué lo usamos?
  - Tracking: guarda cada entrenamiento (parámetros, métricas, el modelo).
    Es como el "git log" de tus experimentos -> reproducibilidad.
  - Model Registry: versiona los modelos (v1, v2...) para saber cuál usar
    en producción y poder volver atrás.

Uso:
    uv run python src/train.py
Luego, para ver los resultados en el navegador:
    uv run mlflow ui   ->  abrir http://localhost:5000
"""

import mlflow
import mlflow.xgboost                       # "flavor" de MLflow para modelos XGBoost
import pandas as pd
from mlflow.models import infer_signature   # deduce el "contrato" entrada/salida del modelo
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ------------------------------------------------------------------
# Configuración
# ------------------------------------------------------------------
RUTA_DATOS = "data/compliance_findings.csv"
EXPERIMENTO = "compliance-risk"             # nombre del experimento en MLflow
NOMBRE_MODELO_REGISTRO = "compliance-risk-model"  # nombre en el Model Registry
SEED = 42                                   # semilla fija -> resultados reproducibles

# Columnas de texto que hay que convertir a números (el modelo no entiende texto)
COLUMNAS_CATEGORICAS = ["framework", "department"]
COLUMNA_OBJETIVO = "risk_high"              # lo que queremos predecir (label)


# ------------------------------------------------------------------
# 1) Cargar y preparar los datos
# ------------------------------------------------------------------
def cargar_y_preparar() -> tuple[pd.DataFrame, pd.Series]:
    """
    Lee el CSV, convierte las columnas de texto a números (one-hot encoding)
    y separa las features (X) del objetivo (y).
    Devuelve: (X, y)
    """
    df = pd.read_csv(RUTA_DATOS)

    # one-hot encoding: cada categoría de texto se vuelve una columna 0/1.
    # Ej: 'framework' -> framework_SOC2, framework_GDPR, ... (con 0 o 1).
    # El modelo solo entiende números, por eso esta conversión es necesaria.
    df = pd.get_dummies(df, columns=COLUMNAS_CATEGORICAS)

    # X = todas las columnas MENOS el objetivo. y = solo el objetivo.
    X = df.drop(columns=[COLUMNA_OBJETIVO])
    y = df[COLUMNA_OBJETIVO]
    return X, y


# ------------------------------------------------------------------
# 2) Entrenar + evaluar + registrar en MLflow
# ------------------------------------------------------------------
def main() -> None:
    X, y = cargar_y_preparar()

    # train_test_split: separa datos para ENTRENAR (80%) y para PROBAR (20%).
    # 'stratify=y' mantiene la misma proporción de clases en ambos conjuntos.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    # Hiperparámetros del modelo (las "perillas" que controlan el entrenamiento).
    params = {
        "n_estimators": 200,        # nº de árboles
        "max_depth": 4,             # profundidad máxima de cada árbol
        "learning_rate": 0.1,       # qué tanto aprende en cada paso
        "subsample": 0.9,           # % de filas usadas por árbol (evita overfitting)
        "random_state": SEED,
        "eval_metric": "logloss",
    }

    # Le decimos a MLflow en qué "experimento" agrupar esta corrida.
    mlflow.set_experiment(EXPERIMENTO)

    # 'start_run' inicia una corrida. Todo lo que registremos adentro queda
    # asociado a esta corrida (params, métricas, modelo).
    with mlflow.start_run() as run:
        print(f"MLflow run id: {run.info.run_id}")

        # --- Entrenar ---
        modelo = XGBClassifier(**params)
        modelo.fit(X_train, y_train)

        # --- Predecir sobre el set de prueba ---
        y_pred = modelo.predict(X_test)                  # clase 0/1
        y_proba = modelo.predict_proba(X_test)[:, 1]     # probabilidad de clase 1

        # --- Calcular métricas (qué tan bueno es el modelo) ---
        metricas = {
            "accuracy": accuracy_score(y_test, y_pred),    # % de aciertos
            "precision": precision_score(y_test, y_pred),  # de los que dijo "alto", cuántos lo eran
            "recall": recall_score(y_test, y_pred),        # de los "alto" reales, cuántos detectó
            "f1": f1_score(y_test, y_pred),                # balance entre precision y recall
            "roc_auc": roc_auc_score(y_test, y_proba),     # capacidad de separar clases (0.5=azar, 1=perfecto)
        }

        # --- Registrar en MLflow (TRACKING) ---
        mlflow.log_params(params)        # guarda los hiperparámetros
        mlflow.log_metrics(metricas)     # guarda las métricas

        # 'signature' = contrato del modelo: qué columnas entran y qué sale.
        # Sirve para validar entradas cuando lo sirvamos como API.
        signature = infer_signature(X_train, modelo.predict(X_train))

        # --- Guardar el modelo y REGISTRARLO en el Model Registry ---
        # 'registered_model_name' crea/incrementa la versión en el registry.
        mlflow.xgboost.log_model(
            modelo,
            name="model",
            signature=signature,
            registered_model_name=NOMBRE_MODELO_REGISTRO,
        )

        # --- Resumen en consola ---
        print("\n=== Métricas del modelo ===")
        for nombre, valor in metricas.items():
            print(f"  {nombre:10s}: {valor:.4f}")
        print(f"\nModelo registrado como: '{NOMBRE_MODELO_REGISTRO}'")
        print("Para ver todo: uv run mlflow ui  ->  http://localhost:5000")


if __name__ == "__main__":
    main()
