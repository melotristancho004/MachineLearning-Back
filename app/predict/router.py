from pydantic import BaseModel
from fastapi import APIRouter
import pickle


class Message(BaseModel):
    message: str


# Cargar el modelo y el vectorizador
model = pickle.load(open("modelColab/modelo.pkl", "rb"))
vectorizer = pickle.load(open("modelColab/vectorizer.pkl", "rb"))


routerModel = APIRouter(prefix="/model", tags=["Model"])


@routerModel.post("/message")
def message(data: Message):
    message = data.message

    texto_vector = vectorizer.transform([message])
    prediccion = model.predict(texto_vector)[0]

    return {"message": message, "prediction": prediccion}
