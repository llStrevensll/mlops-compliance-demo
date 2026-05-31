"""
service.py
==========
Servicio BentoML que expone el modelo como una API REST.

Idea: el modelo entrenado deja de estar "dormido en un archivo" y pasa a ser
un servicio HTTP siempre encendido. Cualquier app le manda un hallazgo (JSON)
y recibe la predicción de riesgo al instante.

Levantar el servidor (local):
    uv run bentoml serve service:ComplianceRiskService
Probar:
    curl -X POST http://localhost:3000/predict \
      -H "Content-Type: application/json" \
      -d '{"finding": {"severity":5,"days_open":120,"control_failures":4,
           "affected_systems":6,"is_repeat_finding":1,"has_remediation_plan":0,
           "framework":"SOC2","department":"IT"}}'
"""

import bentoml
import pandas as pd
from pydantic import BaseModel


# ------------------------------------------------------------------
# Esquema de entrada (un hallazgo CRUDO, sin codificar).
# Pydantic valida automáticamente los tipos: si mandan texto donde va
# un número, la API responde error claro. El Pipeline del modelo se
# encarga del one-hot encoding internamente.
# ------------------------------------------------------------------
class Finding(BaseModel):
    severity: int
    days_open: int
    control_failures: int
    affected_systems: int
    is_repeat_finding: int
    has_remediation_plan: int
    framework: str
    department: str


# ------------------------------------------------------------------
# El servicio. @bentoml.service convierte la clase en un microservicio.
# ------------------------------------------------------------------
@bentoml.service(name="compliance_risk_service")
class ComplianceRiskService:
    # Referencia al modelo en el store de BentoML. Declararlo aquí hace que
    # BentoML lo incluya al construir el Bento (deployable en Kubernetes).
    bento_model = bentoml.models.get("compliance_risk:latest")

    def __init__(self) -> None:
        # Se ejecuta UNA vez al arrancar el servicio: carga el modelo en memoria.
        self.model = bentoml.mlflow.load_model("compliance_risk:latest")

    # @bentoml.api expone este método como endpoint HTTP POST /predict
    @bentoml.api
    def predict(self, finding: Finding) -> dict:
        # Convertimos el hallazgo a un DataFrame de 1 fila (lo que espera el modelo).
        df = pd.DataFrame([finding.model_dump()])

        # El Pipeline preprocesa + predice. Devuelve la clase: 0 (bajo) o 1 (alto).
        pred = int(self.model.predict(df)[0])

        return {
            "risk_high": pred,
            "risk_label": "HIGH" if pred == 1 else "LOW",
        }
