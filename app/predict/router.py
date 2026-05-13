from fastapi import APIRouter
import joblib
from pydantic import BaseModel
from pathlib import Path

routerModel = APIRouter(prefix="/model", tags=["Model"])


class Message(BaseModel):
    message: str

# Ruta absoluta para evitar problemas
BASE_DIR = Path(__file__).resolve().parent.parent
model = joblib.load(BASE_DIR / "modelColab" / "modelo.pkl")
vectorizer = joblib.load(BASE_DIR / "modelColab" / "vectorizer.pkl")

@routerModel.post("/message")
def message(data: Message):
    message = data.message

    texto_vector = vectorizer.transform([message])
    prediccion = model.predict(texto_vector)[0]

    return {"message": message, "prediction": prediccion}
