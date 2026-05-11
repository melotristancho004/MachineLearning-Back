import json
import pickle
from pathlib import Path

def serialize(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, type):
        return obj.__name__
    return str(obj)

with open("app/modelColab/modelo.pkl", "rb") as f:
    model = pickle.load(f)

with open("app/modelColab/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

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
print(model.coef_)