from pydantic import BaseModel
from fastapi import APIRouter
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
model = pickle.load(open(BASE_DIR / "modelColab" / "modelo.pkl", "rb"))
vectorizer = pickle.load(open(BASE_DIR / "modelColab" / "vectorizer.pkl", "rb"))


routerModel = APIRouter(prefix="/model", tags=["Model"])


class Message(BaseModel):
    message: str


@routerModel.post("/message")
def message(data: Message):
    message_text = data.message

    texto_vector = vectorizer.transform([message_text])
    prediccion = model.predict(texto_vector)[0]

    return {"message": message_text, "prediction": prediccion}
