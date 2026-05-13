"""Router para las rutas de predicción del modelo.

Este módulo carga los artefactos (modelo y vectorizador) desde
`app/modelColab/` y expone un endpoint POST `/model/message` que
recibe un texto y devuelve la predicción.
"""

from fastapi import APIRouter
import joblib
from pydantic import BaseModel
from pathlib import Path

routerModel = APIRouter(prefix="/model", tags=["Model"])


class Message(BaseModel):
    """Modelo Pydantic para el payload JSON entrante.

    Campos:
    - message: el texto bruto que se desea clasificar
    """
    message: str


# Resolver rutas relativas a este archivo para localizar los artefactos.
BASE_DIR = Path(__file__).resolve().parent.parent
# Cargar los artefactos entrenados en el tiempo de importación (inicio rápido).
# Si faltan estos archivos la importación fallará y la API no arrancará.
model = joblib.load(BASE_DIR / "modelColab" / "modelo.pkl")
vectorizer = joblib.load(BASE_DIR / "modelColab" / "vectorizer.pkl")


@routerModel.post("/message")
def message(data: Message):
    """Endpoint: recibe un JSON y devuelve una predicción.

    La función transforma el texto entrante usando el vectorizer guardado
    y llama al modelo entrenado para obtener la etiqueta predicha.
    """
    message = data.message

    # Vectorizar el texto de entrada y predecir la etiqueta
    texto_vector = vectorizer.transform([message])
    prediccion = model.predict(texto_vector)[0]

    return {"message": message, "prediction": prediccion}
