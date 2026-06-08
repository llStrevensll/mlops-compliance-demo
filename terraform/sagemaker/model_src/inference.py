"""
inference.py
============
Script que el contenedor sklearn de AWS SageMaker usa para servir el modelo.
SageMaker llama estas 4 funciones en orden:

    model_fn   -> carga el modelo desde disco (una vez al arrancar)
    input_fn   -> convierte la petición HTTP (JSON) en datos para el modelo
    predict_fn -> ejecuta la predicción
    output_fn  -> formatea la respuesta de vuelta a JSON
"""

import json
import os

import joblib
import pandas as pd


def model_fn(model_dir):
    """Carga el Pipeline de sklearn guardado (model.joblib)."""
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def input_fn(request_body, content_type):
    """Convierte el JSON de entrada en un DataFrame que el modelo entiende."""
    if content_type != "application/json":
        raise ValueError(f"Content-Type no soportado: {content_type}")

    data = json.loads(request_body)

    # Aceptamos {"finding": {...}}  o  {...}  o  [ {...}, {...} ]
    if isinstance(data, dict) and "finding" in data:
        data = [data["finding"]]
    elif isinstance(data, dict):
        data = [data]
    return pd.DataFrame(data)


def predict_fn(input_data, model):
    """Ejecuta la predicción (el Pipeline preprocesa + clasifica)."""
    return model.predict(input_data)


def output_fn(prediction, accept):
    """Formatea la salida: clase 0/1 + etiqueta legible."""
    resultado = [
        {"risk_high": int(p), "risk_label": "HIGH" if int(p) == 1 else "LOW"}
        for p in prediction
    ]
    return json.dumps(resultado), "application/json"
