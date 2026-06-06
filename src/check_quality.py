"""
check_quality.py
================
'Quality gate' (compuerta de calidad) del pipeline de CI/CD.

Lee las métricas que dejó train.py en metrics.json y verifica que el modelo
cumpla los umbrales mínimos. Si NO los cumple, termina con código de salida 1
-> en GitHub Actions eso hace FALLAR el build, bloqueando un modelo malo.

Uso:
    uv run python src/check_quality.py
"""

import json
import sys

# Umbrales mínimos que el modelo DEBE cumplir para "pasar".
# Si en el futuro un cambio degrada el modelo por debajo de esto, el CI falla.
UMBRALES = {
    "roc_auc": 0.85,
    "accuracy": 0.78,
}

RUTA_METRICAS = "metrics.json"


def main() -> None:
    # Cargar las métricas que generó el entrenamiento.
    with open(RUTA_METRICAS) as f:
        metricas = json.load(f)

    print("=== Quality gate ===")
    fallos = []

    # Revisar cada umbral.
    for nombre, minimo in UMBRALES.items():
        valor = metricas.get(nombre)
        if valor is None:
            fallos.append(f"falta la métrica '{nombre}'")
            continue
        ok = valor >= minimo
        estado = "OK " if ok else "FALLA"
        print(f"  [{estado}] {nombre}: {valor:.4f}  (mínimo {minimo})")
        if not ok:
            fallos.append(f"{nombre}={valor:.4f} < {minimo}")

    # Decidir: si hubo fallos, salir con código 1 (hace fallar el CI).
    if fallos:
        print("\n❌ Quality gate NO superado:")
        for f in fallos:
            print(f"   - {f}")
        sys.exit(1)   # código != 0 -> GitHub Actions marca el job como fallido

    print("\n✅ Quality gate superado: el modelo cumple los umbrales.")
    sys.exit(0)


if __name__ == "__main__":
    main()
