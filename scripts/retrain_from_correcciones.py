#!/usr/bin/env python3
"""Reentrena el modelo usando un dataset base + correcciones manuales.

Este script convierte el flujo del notebook de correcciones en un flujo
reproducible dentro del proyecto.

Flujo:
1. Cargar `datos/med_dataset_v8.csv`.
2. Cargar `datos/correcciones.csv` con columnas `text` y `emotion`.
3. Normalizar texto y amplificar las correcciones si se desea.
4. Guardar un CSV combinado temporal.
5. Invocar `scripts/train_model.py` sobre ese CSV.
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


PREFIJOS = [
    "",
    "",
    "",
    "hoy ",
    "ayer ",
    "la verdad ",
    "te cuento que ",
    "resulta que ",
    "fijate que ",
    "mira que ",
]

SUFIJOS = [
    "",
    "",
    "",
    " y no sé qué hacer",
    " y me afectó mucho",
    " y estoy mal",
    " y no lo puedo creer",
    " todavía",
    " y ya",
]


def clean_text(text: str) -> str:
    import re

    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^a-záéíóúüñ\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reentrenar el clasificador usando dataset base + correcciones manuales.",
    )
    parser.add_argument(
        "--base-dataset",
        default="datos/med_dataset_v8.csv",
        help="CSV base del proyecto.",
    )
    parser.add_argument(
        "--corrections",
        default="datos/correcciones.csv",
        help="CSV con columnas text y emotion.",
    )
    parser.add_argument(
        "--amplify-factor",
        type=int,
        default=8,
        help="Cantidad de variaciones por corrección.",
    )
    parser.add_argument(
        "--corrections-weight",
        type=int,
        default=3,
        help="Cuántas veces repetir las correcciones amplificadas en el dataset final.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla aleatoria para mezcla y amplificación.",
    )
    parser.add_argument(
        "--keep-merged",
        action="store_true",
        help="Guardar el CSV combinado en app/modelColab/merged_retraining_dataset.csv.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Pasar --overwrite a train_model.py.",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.55,
        help="Umbral de seguridad para el modelo.",
    )
    parser.add_argument(
        "--encoder-name",
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="Encoder de Sentence-Transformers.",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=2.0,
        help="Fuerza de regularización del LogisticRegression.",
    )
    return parser.parse_args()


def amplify_correction(text: str, emotion: str, n: int, rng: random.Random) -> list[dict[str, str]]:
    variations = {clean_text(text)}
    attempts = 0
    while len(variations) < n and attempts < 50:
        candidate = f"{rng.choice(PREFIJOS)}{text}{rng.choice(SUFIJOS)}".strip()
        variations.add(clean_text(candidate))
        attempts += 1
    return [{"text": value, "emotion": emotion} for value in variations if value]


def load_dataset(path: Path, text_col: str = "text", label_col: str = "emotion") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    df = pd.read_csv(path)
    missing = [col for col in (text_col, label_col) if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en {path}: {missing}. Columnas disponibles: {list(df.columns)}")

    df = df[[text_col, label_col]].dropna().copy()
    df[text_col] = df[text_col].astype(str).apply(clean_text)
    df = df[df[text_col].str.len() > 0]
    df = df.drop_duplicates(subset=[text_col, label_col]).reset_index(drop=True)
    return df


def main() -> int:
    args = parse_args()
    rng = random.Random(args.random_state)

    project_root = Path(__file__).resolve().parents[1]
    base_path = (project_root / args.base_dataset).resolve() if not Path(args.base_dataset).is_absolute() else Path(args.base_dataset)
    corrections_path = (project_root / args.corrections).resolve() if not Path(args.corrections).is_absolute() else Path(args.corrections)
    train_script = project_root / "scripts" / "train_model.py"

    print("[INFO] Reentrenamiento con correcciones")
    print(f"[INFO] Base dataset : {base_path}")
    print(f"[INFO] Corrections  : {corrections_path}")

    base_df = load_dataset(base_path)
    print(f"[INFO] Filas base limpias: {len(base_df)}")

    if not corrections_path.exists():
        raise FileNotFoundError(
            f"No se encontró {corrections_path}. Crea un CSV con columnas 'text' y 'emotion'."
        )

    corrections_df = load_dataset(corrections_path)
    print(f"[INFO] Correcciones limpias: {len(corrections_df)}")

    amplified_rows: list[dict[str, str]] = []
    for _, row in corrections_df.iterrows():
        amplified_rows.extend(
            amplify_correction(
                text=row["text"],
                emotion=row["emotion"],
                n=args.amplify_factor,
                rng=rng,
            )
        )

    amplified_df = pd.DataFrame(amplified_rows).drop_duplicates(subset=["text"])
    if amplified_df.empty:
        raise ValueError("La amplificación de correcciones no generó filas válidas.")

    print(f"[INFO] Correcciones amplificadas: {len(amplified_df)}")
    print("[INFO] Conteo por emoción en correcciones amplificadas:")
    print(amplified_df["emotion"].value_counts().to_string())

    combined_parts = [base_df]
    combined_parts.extend([amplified_df] * max(args.corrections_weight, 1))
    merged_df = pd.concat(combined_parts, ignore_index=True)
    merged_df = merged_df.sample(frac=1, random_state=args.random_state).reset_index(drop=True)
    merged_df = merged_df.drop_duplicates(subset=["text", "emotion"]).reset_index(drop=True)

    print(f"[INFO] Dataset combinado final: {len(merged_df)} filas")
    print("[INFO] Conteo por emoción en dataset combinado:")
    print(merged_df["emotion"].value_counts().to_string())

    merged_path = None
    temp_dir = None
    if args.keep_merged:
        out_dir = project_root / "app" / "modelColab"
        out_dir.mkdir(parents=True, exist_ok=True)
        merged_path = out_dir / "merged_retraining_dataset.csv"
        merged_df.to_csv(merged_path, index=False)
        print(f"[INFO] Dataset combinado guardado en: {merged_path}")
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="med_retrain_", dir=str(project_root))
        merged_path = Path(temp_dir.name) / "merged_retraining_dataset.csv"
        merged_df.to_csv(merged_path, index=False)
        print(f"[INFO] Dataset combinado temporal: {merged_path}")

    cmd = [
        sys.executable,
        str(train_script),
        "--dataset",
        str(merged_path),
        "--encoder-name",
        args.encoder_name,
        "--c",
        str(args.c),
        "--decision-threshold",
        str(args.decision_threshold),
        "--random-state",
        str(args.random_state),
    ]

    if args.overwrite:
        cmd.append("--overwrite")

    print("[INFO] Ejecutando entrenamiento base...")
    print("[INFO] Comando:")
    print(" ".join(cmd))

    completed = subprocess.run(cmd, cwd=str(project_root))
    if temp_dir is not None:
        temp_dir.cleanup()

    if completed.returncode != 0:
        print(f"[ERROR] El entrenamiento terminó con código {completed.returncode}")
        return completed.returncode

    print("[OK] Reentrenamiento con correcciones completado correctamente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
