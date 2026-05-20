# MachineLearning-Back

## Overview

API en FastAPI para clasificación de texto.

- Encoder: Sentence-Transformers `paraphrase-multilingual-MiniLM-L12-v2`
- Clasificador: `LogisticRegression` (multinomial)

---

## Quick Start

1. Crear y activar entorno (Windows):

```powershell
python -m venv .venv
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Ejecutar la API:

```bash
uvicorn app.main:app --reload
# http://127.0.0.1:8000 — Swagger: /docs
```

---

## Project structure (relevant)

- `app/main.py` — FastAPI app
- `app/predict/router.py` — `/model/message` endpoint, carga de artefactos y política de decisión
- `app/modelColab/` — modelos y encoder guardados
- `scripts/train_model.py` — script de entrenamiento y exportación
- `train-model.bat` — helper para Windows
- `datos/` — datasets (ej. `med_dataset_v8.csv`)

---

## Training (generate artifacts)

- Entrenar con dataset por defecto:

```bash
python scripts/train_model.py --dataset datos/med_dataset_v8.csv
```

- Windows (batch):

```powershell
train-model.bat
```

- Reentrenar con correcciones (Windows):

```powershell
retrain-from-correcciones.bat
```

### Cuándo usar cada uno

- `train-model.bat`: cuando quieras reentrenar desde el dataset base sin correcciones manuales.
- `retrain-from-correcciones.bat`: cuando ya tengas `datos/correcciones.csv` y quieras incorporar ejemplos corregidos.

- Parámetros útiles:
  - `--decision-threshold FLOAT` (ej. `0.55`) — umbral para marcar predicciones como "safe".
  - `--overwrite` — sobrescribir artefactos existentes.

### Reentrenar con correcciones manuales

Si quieres corregir errores del modelo con ejemplos nuevos, usa:

```bash
python scripts/retrain_from_correcciones.py --corrections datos/correcciones.csv --keep-merged --overwrite
```

- `datos/correcciones.csv` debe tener las columnas `text` y `emotion`.
- La plantilla de referencia está en `datos/correcciones.csv.example`.
- El script amplifica las correcciones, combina el dataset base y vuelve a ejecutar `train_model.py`.

---

## Artifacts (output)

- `app/modelColab/modelo_v2.pkl` — modelo serializado (joblib)
- `app/modelColab/encoder_med/` — encoder Sentence-Transformers
- `app/modelColab/training_metrics.json` — métricas y metadatos (classification report, confusion matrix, training signature)

Resumen rápido (último entrenamiento): **accuracy 0.9613**, **threshold 0.55**, **7021** test rows (safe: 6792, unsafe: 229).

---

## API — `/model/message`

- Método: `POST`
- Body (JSON):

```json
{ "message": "me siento muy cansado y sin ganas" }
```

- Respuesta (principal):

```json
{
  "message": "...",
  "prediction": "triste",
  "confidence": 95.2,
  "safe": true,
  "threshold": 55.0,
  "probabilities": { "enojo": 2.1, "feliz": 1.2, "miedo": 1.5, "triste": 95.2 }
}
```

Campo `safe`: `confidence >= decision_threshold * 100` (el umbral se lee de `training_metrics.json`).

---

## Validate & quick checks

- Ejecutar verificación mínima:

```bash
python scripts/validate_project.py
```

- Si el endpoint devuelve `503 Model artifacts missing` -> ejecutar entrenamiento o colocar `modelo_v2.pkl` y `encoder_med/` en `app/modelColab/`.

---

## Troubleshooting

- Puerto ocupado: cambiar puerto de Uvicorn:

```bash
uvicorn app.main:app --reload --port 8001
```

- Activar entorno en PowerShell (Windows):

```powershell
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; .venv\Scripts\Activate.ps1
```

---

## Metrics & reports

- Detalle completo en: `app/modelColab/training_metrics.json` (contiene `classification_report` y `confusion_matrix`).

---

## Classification report

Resumen del `classification_report` (test set):

| Class       | Precision | Recall  | F1-score   | Support |
|-------------|----------:|--------:|-----------:|--------:|
| enojo       | 0.9536    | 0.9542  | 0.9539     | 1789    |
| feliz       | 0.9708    | 0.9708  | 0.9708     | 1780    |
| miedo       | 0.9777    | 0.9726  | 0.9752     | 1715    |
| triste      | 0.9433    | 0.9476  | 0.9454     | 1737    |
| **accuracy**|           |         | **0.9613** | 7021    |
| macro avg   | 0.9614    | 0.9613  | 0.9613     | 7021    |
| weighted avg| 0.9613    | 0.9613  | 0.9613     | 7021    |

(Detalle completo y matriz de confusión en `app/modelColab/training_metrics.json`.)

## Confusion matrix

Matriz de confusión (filas = etiqueta real, columnas = etiqueta predicha):

| Actual \ Predicted | enojo | feliz | miedo | triste |
|-------------------:|------:|------:|------:|-------:|
| enojo              | 1707  | 15    | 24    | 43     |
| feliz              | 16    | 1728  | 4     | 32     |
| miedo              | 21    | 2     | 1668  | 24     |
| triste             | 46    | 35    | 10    | 1646   |

(La matriz completa y los datos brutos están en `app/modelColab/training_metrics.json`.)
---