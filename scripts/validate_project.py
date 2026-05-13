"""Pruebas básicas (smoke tests) para el proyecto MachineLearning-Back.

Este script comprueba que los artefactos entrenados existen, importa la app
FastAPI y realiza llamadas de ejemplo al endpoint de predicción. Está pensado
como una verificación mínima y rápida para instructores o pipelines de CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
MODEL_PATH = ROOT_DIR / "app" / "modelColab" / "modelo.pkl"
VECTORIZER_PATH = ROOT_DIR / "app" / "modelColab" / "vectorizer.pkl"


def ensure_project_root_on_path() -> None:
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))
    # Insertar ROOT_DIR y APP_DIR en sys.path asegura que importaciones como
    # `from app.main import app` funcionen al ejecutar este script directamente.


def validate_artifacts() -> None:
    missing = [path for path in (MODEL_PATH, VECTORIZER_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Faltan artefactos entrenados: " + ", ".join(str(path) for path in missing)
        )


def validate_api() -> None:
    ensure_project_root_on_path()

    # Import FastAPI app after ensuring app package directory is on sys.path
    from app.main import app

    client = TestClient(app)
    samples = [
        "me siento muy feliz hoy",
        "estoy muy triste y cansado",
        "tengo mucho miedo del futuro",
        "me tiene muy molesto esta situacion",
    ]
    allowed_labels = {"feliz", "triste", "miedo", "enojo"}

    for message in samples:
        response = client.post("/model/message", json={"message": message})
        if response.status_code != 200:
            raise AssertionError(
                f"Código de estado inesperado {response.status_code} para la muestra: {message}"
            )

        payload = response.json()
        if payload.get("message") != message:
            raise AssertionError(f"El campo 'message' en la respuesta no coincide para la muestra: {message}")

        prediction = payload.get("prediction")
        if prediction not in allowed_labels:
            raise AssertionError(
                f"Predicción inesperada '{prediction}' para la muestra: {message}"
            )


def main() -> None:
    validate_artifacts()
    validate_api()
    print("Validación exitosa: los artefactos existen y el endpoint de predicción responde correctamente.")


if __name__ == "__main__":
    main()