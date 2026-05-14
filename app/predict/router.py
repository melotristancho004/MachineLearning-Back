"""Router para las rutas de predicción del modelo.
 
Este módulo carga los artefactos (modelo y encoder) desde
`app/modelColab/` y expone un endpoint POST `/model/message` que
recibe un texto y devuelve la predicción con su nivel de confianza.
"""
 
from fastapi import APIRouter
import joblib
import re
from pydantic import BaseModel
from pathlib import Path
from sentence_transformers import SentenceTransformer
 
routerModel = APIRouter(prefix="/model", tags=["Model"])
 
 
class Message(BaseModel):
    """Modelo Pydantic para el payload JSON entrante.
 
    Campos:
    - message: el texto bruto que se desea clasificar
    """
    message: str
 
 
# ── Rutas a los artefactos ────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
MODEL_DIR   = BASE_DIR / "modelColab"
 
# ── Cargar artefactos al iniciar la API ───────────────────────────────────────
# El encoder es la carpeta descomprimida de encoder_med.zip
model   = joblib.load(MODEL_DIR / "modelo_v3.pkl")
encoder = SentenceTransformer(str(MODEL_DIR / "encoder_med"))
 
# ── Umbral de confianza ───────────────────────────────────────────────────────
UMBRAL = 0.55
 
 
# ── Limpieza de texto (igual que en el entrenamiento) ─────────────────────────
def limpiar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúüñ\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto
 
 
@routerModel.post("/message")
def message(data: Message):
    """Endpoint: recibe un JSON y devuelve la predicción con confianza.
 
    Returns:
        message:    texto original
        prediction: emoción predicha (feliz | triste | enojo | miedo)
        confidence: porcentaje de confianza (0-100)
        is_sure:    False si la confianza está por debajo del umbral
        probs:      probabilidad de cada emoción
    """
    texto_limpio = limpiar_texto(data.message)
 
    # Generar embedding y predecir
    emb       = encoder.encode([texto_limpio])
    probs     = model.predict_proba(emb)[0]
    clases    = model.classes_
 
    idx        = probs.argmax()
    prediccion = clases[idx]
    confianza  = float(probs[idx])
 
    return {
        "message"   : data.message,
        "prediction": prediccion,
        "confidence": round(confianza * 100, 1),
        "is_sure"   : confianza >= UMBRAL,
        "probs"     : {c: round(float(p) * 100, 1) for c, p in zip(clases, probs)}
    }
 