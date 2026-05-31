"""
import_model.py
================
Copia el modelo registrado en MLflow al "almacén de modelos" de BentoML.

¿Por qué este paso?
  - El modelo vive en MLflow (mlflow.db + mlruns/).
  - BentoML necesita el modelo en SU propio almacén para poder empaquetarlo
    como un servicio auto-contenido (que luego desplegamos en Kubernetes).
  - import_model toma el modelo de MLflow y lo guarda en el store de BentoML.

Uso:
    uv run python src/import_model.py
"""

import bentoml
import mlflow

# Apuntamos al mismo backend local donde train.py registró el modelo.
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# 'latest' = la última versión registrada del modelo.
MODEL_URI = "models:/compliance-risk-model/latest"
NOMBRE_EN_BENTOML = "compliance_risk"

# Importa el modelo de MLflow al store de BentoML.
bento_model = bentoml.mlflow.import_model(NOMBRE_EN_BENTOML, MODEL_URI)

print(f"Modelo importado a BentoML como: {bento_model}")
