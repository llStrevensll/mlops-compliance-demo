"""
drift_report.py
===============
Detección de DATA DRIFT con Evidently.

Concepto: un modelo se entrena con datos de cierta "forma" (distribución). Con
el tiempo, los datos reales cambian (data drift) y el modelo empieza a fallar
sin que nadie lo note. Evidently compara los datos de REFERENCIA (con los que
se entrenó) contra los datos ACTUALES y detecta si la distribución cambió.

Aquí simulamos drift desplazando algunas columnas, para ver la detección en acción.

Uso:
    uv run python src/drift_report.py
Salida:
    reports/drift_report.html  (reporte visual)
"""

import os

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

RUTA_DATOS = "data/compliance_findings.csv"
RUTA_REPORTE = "reports/drift_report.html"

COLUMNAS_NUMERICAS = [
    "severity",
    "days_open",
    "control_failures",
    "affected_systems",
    "is_repeat_finding",
    "has_remediation_plan",
]
COLUMNAS_CATEGORICAS = ["framework", "department"]


def cargar_referencia() -> pd.DataFrame:
    """Datos de REFERENCIA = los mismos con los que se entrenó (sin el label)."""
    df = pd.read_csv(RUTA_DATOS)
    return df.drop(columns=["risk_high"])


def simular_datos_actuales(referencia: pd.DataFrame) -> pd.DataFrame:
    """Crea datos ACTUALES con DRIFT (distribución desplazada a propósito).

    Simulamos que en producción los hallazgos ahora son más graves y llevan
    más tiempo abiertos -> el modelo entrenado con datos 'viejos' podría fallar.
    """
    rng = np.random.default_rng(123)
    actuales = referencia.sample(frac=1.0, random_state=123).reset_index(drop=True)

    # Drift numérico: subimos days_open y severity (datos "más graves" que antes)
    actuales["days_open"] = actuales["days_open"] + rng.integers(40, 80, size=len(actuales))
    actuales["severity"] = np.clip(actuales["severity"] + 1, 1, 5)
    actuales["control_failures"] = actuales["control_failures"] + rng.integers(1, 4, size=len(actuales))
    return actuales


def main() -> None:
    referencia = cargar_referencia()
    actuales = simular_datos_actuales(referencia)

    # DataDefinition le dice a Evidently qué columnas son numéricas y cuáles categóricas.
    data_def = DataDefinition(
        numerical_columns=COLUMNAS_NUMERICAS,
        categorical_columns=COLUMNAS_CATEGORICAS,
    )

    # Envolvemos los DataFrames en Dataset de Evidently.
    ds_ref = Dataset.from_pandas(referencia, data_definition=data_def)
    ds_cur = Dataset.from_pandas(actuales, data_definition=data_def)

    # DataDriftPreset = conjunto de chequeos de drift listos para usar.
    report = Report([DataDriftPreset()])
    resultado = report.run(reference_data=ds_ref, current_data=ds_cur)

    # Guardar el reporte visual HTML.
    os.makedirs("reports", exist_ok=True)
    resultado.save_html(RUTA_REPORTE)

    print(f"Reporte de drift guardado en: {RUTA_REPORTE}")
    print("Ábrelo en el navegador para ver qué columnas tuvieron drift.")


if __name__ == "__main__":
    main()
