"""
generate_data.py
=================
Genera un dataset SINTÉTICO (ficticio) de hallazgos de auditoría de compliance.

¿Por qué sintético?
  - No usamos NINGÚN dato real ni propietario de ninguna empresa.
  - Con una 'seed' (semilla) fija, el dataset es 100% REPRODUCIBLE:
    cualquiera que corra este script obtiene exactamente las mismas filas.

Problema de ML: clasificación binaria -> predecir si un hallazgo es de
riesgo ALTO (1) o BAJO (0) a partir de sus características (features).

Uso:
    uv run python src/generate_data.py
Salida:
    data/compliance_findings.csv
"""

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# 1) Parámetros del dataset
# ------------------------------------------------------------------
N_FILAS = 5000          # cuántos hallazgos generamos
SEED = 42               # semilla fija -> reproducibilidad (siempre los mismos datos)
RUTA_SALIDA = "data/compliance_findings.csv"

# 'rng' = generador de números aleatorios de numpy, atado a la semilla.
# Todo lo aleatorio sale de aquí para que sea reproducible.
rng = np.random.default_rng(SEED)

# Valores posibles para las columnas categóricas (texto)
FRAMEWORKS = ["SOC2", "ISO27001", "GDPR", "HIPAA"]
DEPARTAMENTOS = ["Finance", "IT", "HR", "Operations", "Legal"]


# ------------------------------------------------------------------
# 2) Generar las FEATURES (las "pistas" que el modelo usará)
# ------------------------------------------------------------------
def generar_features(n: int) -> pd.DataFrame:
    """Crea un DataFrame con n filas de features sintéticas.

    --- Desglose de la SINTAXIS de la firma (la primera línea) ---
        def                -> palabra clave que DEFINE una función.
        generar_features   -> nombre de la función (cómo la llamamos).
        (n: int)           -> recibe un parámetro 'n'. El ': int' es un
                              'type hint' (pista de tipo): indica que 'n'
                              debería ser un entero. Python NO lo obliga en
                              tiempo de ejecución, pero ayuda a:
                                • legibilidad (otro dev entiende qué espera)
                                • autocompletado del editor (VS Code, PyCharm)
                                • detectar errores con herramientas (mypy, ruff)
        -> pd.DataFrame    -> 'return type hint': dice que la función DEVUELVE
                              un pandas DataFrame (una tabla). Es informativo,
                              no obligatorio, pero documenta el contrato.

    En resumen: "esta función recibe un entero n y devuelve una tabla".
    """
    datos = {
        # severity: gravedad de 1 a 5. randint(1, 6) genera enteros 1..5
        "severity": rng.integers(1, 6, size=n),

        # days_open: días abierto. Usamos exponencial para que la mayoría
        # tenga pocos días y unos pocos lleven mucho tiempo (cola larga, realista).
        "days_open": rng.exponential(scale=30, size=n).round().astype(int),

        # control_failures: nº de controles que fallaron (0 a 10, sesgado a pocos)
        "control_failures": rng.poisson(lam=2, size=n),

        # affected_systems: nº de sistemas impactados (1 a ~8)
        "affected_systems": rng.poisson(lam=2, size=n) + 1,

        # is_repeat_finding: 1 si es recurrente. ~20% de los casos.
        "is_repeat_finding": (rng.random(n) < 0.20).astype(int),

        # has_remediation_plan: 1 si ya tiene plan. ~60% de los casos.
        "has_remediation_plan": (rng.random(n) < 0.60).astype(int),

        # framework y department: categóricos, elegidos al azar de las listas.
        "framework": rng.choice(FRAMEWORKS, size=n),
        "department": rng.choice(DEPARTAMENTOS, size=n),
    }
    return pd.DataFrame(datos)


# ------------------------------------------------------------------
# 3) Derivar el LABEL (lo que queremos predecir): risk_high (0/1)
# ------------------------------------------------------------------
def calcular_riesgo(df: pd.DataFrame) -> pd.Series:
    """
    Calcula una 'puntuación de riesgo' con una REGLA LÓGICA y la convierte
    en label binario. Le agregamos RUIDO para que no sea perfectamente
    predecible (los datos reales nunca lo son) -> el modelo tiene que 'aprender'.
    """
    # Puntuación: cada feature suma riesgo con cierto peso.
    score = (
        df["severity"] * 1.5                    # más severo -> más riesgo
        + (df["days_open"] > 60) * 2.0          # llevar >60 días abierto pesa
        + df["control_failures"] * 0.8          # más fallas -> más riesgo
        + df["affected_systems"] * 0.5          # más sistemas -> más riesgo
        + df["is_repeat_finding"] * 2.0         # recurrente -> sube el riesgo
        - df["has_remediation_plan"] * 1.5      # tener plan BAJA el riesgo
    )

    # Ruido aleatorio (distribución normal) para que el patrón no sea trivial.
    ruido = rng.normal(loc=0, scale=2.0, size=len(df))
    score_con_ruido = score + ruido

    # Umbral: si la puntuación supera la mediana -> riesgo ALTO (1), si no BAJO (0).
    # Usar la mediana deja el dataset balanceado (~50/50), bueno para entrenar.
    umbral = np.median(score_con_ruido)
    return (score_con_ruido > umbral).astype(int)


# ------------------------------------------------------------------
# 4) Función principal: une todo y guarda el CSV
# ------------------------------------------------------------------
def main() -> None:
    print(f"Generando {N_FILAS} hallazgos sintéticos (seed={SEED})...")

    df = generar_features(N_FILAS)
    df["risk_high"] = calcular_riesgo(df)

    # Guardar a CSV (sin la columna de índice de pandas)
    df.to_csv(RUTA_SALIDA, index=False)

    # Resumen para entender qué generamos
    print(f"\nGuardado en: {RUTA_SALIDA}")
    print(f"Filas: {len(df)}  |  Columnas: {len(df.columns)}")
    print("\nBalance del label (risk_high):")
    print(df["risk_high"].value_counts(normalize=True).round(3).to_string())
    print("\nPrimeras 5 filas:")
    print(df.head().to_string())


if __name__ == "__main__":
    main()
