"""
package_sagemaker.py
====================
Empaqueta el modelo entrenado en un model.tar.gz con la estructura que espera
el contenedor sklearn de AWS SageMaker:

    model.tar.gz
    ├── model.joblib        (el Pipeline sklearn serializado)
    └── code/
        ├── inference.py     (cómo cargar/servir el modelo)
        └── requirements.txt (versiones de librerías a instalar)

Uso:
    uv run python src/package_sagemaker.py
"""

import os
import shutil
import tarfile

import joblib
import mlflow

# Mismo backend local donde train.py registró el modelo.
mlflow.set_tracking_uri("sqlite:///mlflow.db")

MODEL_URI = "models:/compliance-risk-model/latest"
DIR_BUILD = "terraform/sagemaker/build"          # carpeta temporal de ensamblado
DIR_CODE_FUENTE = "terraform/sagemaker/model_src"  # inference.py + requirements.txt
RUTA_TAR = "terraform/sagemaker/model.tar.gz"     # salida final


def main() -> None:
    # 1) Cargar el Pipeline desde el Model Registry de MLflow.
    print(f"Cargando modelo desde MLflow: {MODEL_URI}")
    modelo = mlflow.sklearn.load_model(MODEL_URI)

    # 2) Preparar carpeta de ensamblado limpia.
    if os.path.exists(DIR_BUILD):
        shutil.rmtree(DIR_BUILD)
    os.makedirs(os.path.join(DIR_BUILD, "code"), exist_ok=True)

    # 3) Guardar el modelo como model.joblib.
    joblib.dump(modelo, os.path.join(DIR_BUILD, "model.joblib"))

    # 4) Copiar el código de inferencia a code/.
    for archivo in ("inference.py", "requirements.txt"):
        shutil.copy(
            os.path.join(DIR_CODE_FUENTE, archivo),
            os.path.join(DIR_BUILD, "code", archivo),
        )

    # 5) Comprimir todo en model.tar.gz (rutas relativas, como pide SageMaker).
    with tarfile.open(RUTA_TAR, "w:gz") as tar:
        tar.add(os.path.join(DIR_BUILD, "model.joblib"), arcname="model.joblib")
        tar.add(os.path.join(DIR_BUILD, "code"), arcname="code")

    tamano_kb = os.path.getsize(RUTA_TAR) / 1024
    print(f"Paquete creado: {RUTA_TAR} ({tamano_kb:.1f} KB)")
    print("Contenido: model.joblib + code/inference.py + code/requirements.txt")


if __name__ == "__main__":
    main()
