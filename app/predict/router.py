"""Router para las rutas de predicción del modelo.

Este módulo carga los artefactos alineados al notebook de embeddings:
`modelo_v1.pkl` y `encoder_med/` dentro de `app/modelColab/`.
"""

from fastapi import APIRouter, HTTPException
from pathlib import Path

import json
import joblib
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

routerModel = APIRouter(prefix="/model", tags=["Model"])


class Message(BaseModel):
    """Modelo Pydantic para el payload JSON entrante.

    Campos:
    - message: el texto bruto que se desea clasificar
    """
    message: str


# Resolver rutas relativas a este archivo para localizar los artefactos.
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "modelColab"

MODEL_PATH = MODEL_DIR / "modelo_v1.pkl"
ENCODER_PATH = MODEL_DIR / "encoder_med"
METRICS_PATH = MODEL_DIR / "training_metrics.json"

metrics = {}

def _load_metrics():
    m = {}
    if METRICS_PATH.exists():
        try:
            with METRICS_PATH.open("r", encoding="utf-8") as fh:
                m = json.load(fh)
        except Exception:
            m = {}
    return m


# Intentamos cargar los artefactos; si no existen, no levantamos excepción
# para evitar que la app falle al importarse. El endpoint devolverá 503
# hasta que los artefactos estén presentes.
model = None
encoder = None
ARTIFACTS_LOADED = False

metrics = _load_metrics()


ROBBERY_PATTERNS = (
    "me robaron",
    "robaron",
    "robado",
    "robada",
    "robo",
    "robó",
    "roben",
    "asaltaron",
    "asaltado",
    "asaltada",
    "asalto",
    "asalto",
    "me asaltaron",
)

try:
    if MODEL_PATH.exists() and ENCODER_PATH.exists():
        model = joblib.load(MODEL_PATH)
        encoder = SentenceTransformer(str(ENCODER_PATH))
        ARTIFACTS_LOADED = True
    else:
        ARTIFACTS_LOADED = False
except Exception:
    model = None
    encoder = None
    ARTIFACTS_LOADED = False


def _apply_domain_override(message: str, probs_dict: dict[str, float], prediction: str) -> tuple[str, dict[str, float]]:
    """Corrige casos obvios de robo/asalto que el modelo tiende a confundir."""
    lowered = message.lower()
    if any(pattern in lowered for pattern in ROBBERY_PATTERNS):
        if "triste" in probs_dict:
            probs_dict["triste"] = max(probs_dict["triste"], 99.0)
        if "feliz" in probs_dict:
            probs_dict["feliz"] = min(probs_dict["feliz"], 0.5)
        if "enojo" in probs_dict:
            probs_dict["enojo"] = min(probs_dict["enojo"], 0.5)
        if "miedo" in probs_dict:
            probs_dict["miedo"] = min(probs_dict["miedo"], 0.5)
        prediction = "triste"
    return prediction, probs_dict


@routerModel.post("/message")
def message(data: Message):
    """Endpoint: recibe un JSON y devuelve una predicción.

    La función transforma el texto entrante usando el encoder guardado
    y llama al modelo entrenado para obtener la etiqueta predicha.
    """
    message = data.message

    # Si los artefactos no están cargados, devolvemos 503 para que el cliente
    # sepa que el servicio aún no está listo.
    if not ARTIFACTS_LOADED or model is None or encoder is None:
        raise HTTPException(status_code=503, detail="Model artifacts missing. Run training to generate `modelo_v1.pkl` and `encoder_med/` in app/modelColab/.")

    # Generar embeddings y predecir la etiqueta
    vec = encoder.encode([message], show_progress_bar=False, convert_to_numpy=True)
    prediccion = model.predict(vec)[0]
    probabilidades = model.predict_proba(vec)[0]

    # Crear diccionario con probabilidades por clase (en porcentaje)
    probs_dict = {label: float(prob) * 100 for label, prob in zip(model.classes_, probabilidades)}
    prediccion, probs_dict = _apply_domain_override(message, probs_dict, prediccion)

    return {"message": message, "prediction": prediccion, "probabilities": probs_dict}
