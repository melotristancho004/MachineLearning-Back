# MachineLearning-Back

API en FastAPI para clasificar mensajes de texto con un modelo entrenado con TF-IDF + Logistic Regression.

## Descripción

Este backend expone un endpoint HTTP para recibir un texto y devolver una predicción. El modelo y el vectorizador se cargan desde `app/modelColab/`.

## Estructura

- `app/main.py`: punto de entrada de la API.
- `app/predict/router.py`: router con el endpoint de predicción.
- `scripts/train_model.py`: script de entrenamiento y exportación de artefactos.
- `scripts/ver_parametros.py`: inspección de parámetros del modelo y vectorizador.
- `datos/`: datasets usados para entrenamiento.
- `app/modelColab/`: artefactos generados (`modelo.pkl`, `vectorizer.pkl`, `training_metrics.json`).

## Requisitos

- Python 3.10 o superior recomendado.
- Dependencias listadas en `requirements.txt`.

## Instalación

Crear y activar un entorno virtual:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Entrenamiento del modelo

El proyecto incluye un script para entrenar el modelo y guardar los artefactos en `app/modelColab/`.

Usando el dataset por defecto:

```bash
python scripts/train_model.py --dataset datos/med_dataset_v2.csv
```

Con el archivo por lotes de Windows:

```bash
train-model.bat
```

Opcionalmente puedes usar otro dataset, por ejemplo `datos/med_dataset_v7.csv`:

```bash
python scripts/train_model.py --dataset datos/med_dataset_v7.csv --overwrite
```

## Ejecutar la API

Levantar el servidor con Uvicorn desde la raíz de `MachineLearning-Back`:

```bash
uvicorn app.main:app --reload
```

La API quedará disponible en `http://127.0.0.1:8000`.

Documentación automática:

- Swagger: `http://127.0.0.1:8000/docs`
- Redoc: `http://127.0.0.1:8000/redoc`

## Endpoint de predicción

### `POST /model/message`

Recibe un JSON con el campo `message`.

Ejemplo de cuerpo:

```json
{
  "message": "me siento muy cansado y sin ganas"
}
```

Ejemplo de respuesta:

```json
{
  "message": "me siento muy cansado y sin ganas",
  "prediction": "triste"
}
```

## Frontend

El frontend consume el backend en `http://127.0.0.1:8000/model/message`.
Si ejecutas ambos proyectos localmente, primero inicia el backend y luego abre el frontend.

## Notas

- La ruta del modelo se resuelve de forma absoluta dentro de `app/predict/router.py`.
- La API permite CORS abierto para facilitar pruebas con el frontend.
- Si entrenas de nuevo, se sobrescriben `modelo.pkl`, `vectorizer.pkl` y `training_metrics.json` solo cuando usas `--overwrite` o cambian los parámetros/signature del entrenamiento.