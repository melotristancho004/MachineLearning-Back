"""Entrenador del clasificador de emociones con embeddings semánticos.

Este script sigue el flujo del notebook de embeddings:
- limpia texto en español
- divide train/test
- genera embeddings con Sentence-Transformers
- entrena Logistic Regression multinomial
- guarda `modelo_v1.pkl`, `encoder_med/` y `training_metrics.json`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^a-záéíóúüñ\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else base_dir / path


def sha256_of_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenar clasificador de emociones con Sentence-Transformers + Logistic Regression.",
    )
    parser.add_argument("--dataset", required=True, help="Ruta del CSV de entrenamiento.")
    parser.add_argument("--text-col", default="text", help="Columna de texto.")
    parser.add_argument("--label-col", default="emotion", help="Columna de etiqueta.")
    parser.add_argument(
        "--encoder-name",
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="Modelo de Sentence-Transformers a usar.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporción del test.")
    parser.add_argument("--random-state", type=int, default=42, help="Semilla aleatoria.")
    parser.add_argument("--c", type=float, default=2.0, help="Fuerza de regularización.")
    parser.add_argument(
        "--model-out",
        default="app/modelColab/modelo_v1.pkl",
        help="Archivo de salida para el modelo.",
    )
    parser.add_argument(
        "--encoder-out",
        default="app/modelColab/encoder_med",
        help="Carpeta de salida para el encoder guardado.",
    )
    parser.add_argument(
        "--metrics-out",
        default="app/modelColab/training_metrics.json",
        help="Archivo de métricas de salida.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Sobrescribir artefactos.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must be a float in the interval (0, 1).")
    if args.c <= 0:
        raise ValueError("--c must be > 0.")

    project_root = Path(__file__).resolve().parents[1]
    dataset_path = resolve_path(args.dataset, project_root)
    model_out = resolve_path(args.model_out, project_root)
    encoder_out = resolve_path(args.encoder_out, project_root)
    metrics_out = resolve_path(args.metrics_out, project_root)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    training_signature = {
        "dataset_checksum": sha256_of_file(dataset_path),
        "script_checksum": sha256_of_file(Path(__file__).resolve()),
        "encoder_name": args.encoder_name,
        "text_col": args.text_col,
        "label_col": args.label_col,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "c": args.c,
    }

    if not args.overwrite and model_out.exists() and encoder_out.exists() and metrics_out.exists():
        try:
            with metrics_out.open("r", encoding="utf-8") as fh:
                previous = json.load(fh)
            if previous.get("training_signature") == training_signature:
                print("Artefactos existentes y configuración sin cambios. No se reentrena.")
                return
        except Exception:
            pass

    df = pd.read_csv(dataset_path)
    missing = [col for col in (args.text_col, args.label_col) if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")

    df = df[[args.text_col, args.label_col]].dropna().copy()
    df[args.text_col] = df[args.text_col].astype(str).apply(clean_text)
    df = df[df[args.text_col].str.len() > 0]
    df = df.drop_duplicates(subset=[args.text_col, args.label_col]).reset_index(drop=True)

    X = df[args.text_col].tolist()
    y = df[args.label_col].astype(str).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    print(f"Cargando encoder: {args.encoder_name}")
    encoder = SentenceTransformer(args.encoder_name)

    print("Generando embeddings de entrenamiento...")
    X_train_emb = encoder.encode(X_train, batch_size=128, show_progress_bar=True)
    print("Generando embeddings de prueba...")
    X_test_emb = encoder.encode(X_test, batch_size=128, show_progress_bar=True)

    model = LogisticRegression(
        max_iter=1000,
        C=args.c,
        solver="lbfgs",
        multi_class="multinomial",
        random_state=args.random_state,
    )
    model.fit(X_train_emb, y_train)

    y_pred = model.predict(X_test_emb)
    acc = accuracy_score(y_test, y_pred)
    labels = list(model.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_out)

    temp_encoder_out = Path(
        tempfile.mkdtemp(prefix=f"{encoder_out.name}_", dir=str(encoder_out.parent))
    )
    try:
        encoder.save(str(temp_encoder_out), safe_serialization=False)
        if encoder_out.exists():
            shutil.rmtree(encoder_out)
        shutil.move(str(temp_encoder_out), str(encoder_out))
    finally:
        if temp_encoder_out.exists() and temp_encoder_out != encoder_out:
            shutil.rmtree(temp_encoder_out, ignore_errors=True)

    metrics = {
        "dataset_path": str(dataset_path),
        "rows_used": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "accuracy": round(float(acc), 4),
        "labels": labels,
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": cm.tolist(),
        "embedding": {
            "encoder_name": args.encoder_name,
            "encoder_out": str(encoder_out),
        },
        "model": {
            "type": "LogisticRegression",
            "C": float(args.c),
            "solver": "lbfgs",
            "multi_class": "multinomial",
            "max_iter": 1000,
        },
        "training_signature": training_signature,
    }

    with metrics_out.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)

    print("Training complete.")
    print(f"Model saved to: {model_out}")
    print(f"Encoder saved to: {encoder_out}")
    print(f"Metrics saved to: {metrics_out}")
    print(f"Summary -> accuracy={metrics['accuracy']}")


if __name__ == "__main__":
    main()
