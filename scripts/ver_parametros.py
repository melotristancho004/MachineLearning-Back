"""Script de utilidad para inspeccionar parámetros del modelo y del vectorizador.

Carga los artefactos entrenados y muestra un resumen JSON compacto que puede
incluirse en reportes o usarse para depuración. Es intencionalmente ligero.
"""

import json
import pickle
from pathlib import Path


def serialize(obj):
    # Ayudante para hacer objetos serializables a JSON (Paths, tipos, etc.)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, type):
        return obj.__name__
    return str(obj)


with open("app/modelColab/modelo.pkl", "rb") as f:
    model = pickle.load(f)

with open("app/modelColab/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Construir un diccionario compacto con los atributos más relevantes del modelo y vectorizer
info = {
    "model_class": model.__class__.__name__,
    "model_params": model.get_params(),
    "vectorizer_params": vectorizer.get_params(),
    "vocabulary_size": len(vectorizer.vocabulary_),
    "classes": model.classes_.tolist(),
    "coef_shape": list(model.coef_.shape),
    "intercept": model.intercept_.tolist(),
    "n_iter": model.n_iter_.tolist() if hasattr(model.n_iter_, "tolist") else model.n_iter_,
}

print(json.dumps(info, ensure_ascii=False, indent=2, default=serialize))
# También imprimir los coeficientes crudos para inspecciones más detalladas cuando sea necesario
print(model.coef_)