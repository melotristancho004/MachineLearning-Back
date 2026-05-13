import argparse
import json
import pickle
import re
import hashlib
from pathlib import Path

try:
    import pandas as pd
except Exception as e:
    raise ImportError(
        "pandas is required to run this script but could not be imported; "
        "install it with 'pip install pandas' (or 'conda install pandas')"
    ) from e
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def clean_text(text: str) -> str:
    """Basic text normalization for spanish short messages."""
    if not isinstance(text, str):
        return ""

    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train emotion classifier (TF-IDF + LogisticRegression) and export artifacts."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to CSV dataset.",
    )
    parser.add_argument(
        "--text-col",
        default="text",
        help="Name of text column in dataset (default: text).",
    )
    parser.add_argument(
        "--label-col",
        default="emotion",
        help="Name of label column in dataset (default: emotion).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=2,
        help="Minimum document frequency for TF-IDF terms (default: 2).",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=20000,
        help="Maximum number of TF-IDF features (default: 20000).",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=2,
        help="Maximum n-gram size for TF-IDF (default: 2).",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=2.0,
        help="Inverse regularization strength for LogisticRegression (default: 2.0).",
    )
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="balanced",
        help="Class weighting mode for LogisticRegression (default: balanced).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Stratified cross-validation folds on train set (default: 5, 0 disables).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force retrain and overwrite existing artifacts.",
    )
    parser.add_argument(
        "--model-out",
        default="app/modelColab/modelo.pkl",
        help="Output path for trained model.",
    )
    parser.add_argument(
        "--vectorizer-out",
        default="app/modelColab/vectorizer.pkl",
        help="Output path for trained vectorizer.",
    )
    parser.add_argument(
        "--metrics-out",
        default="app/modelColab/training_metrics.json",
        help="Output path for metrics report.",
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame, text_col: str, label_col: str) -> None:
    missing = [col for col in (text_col, label_col) if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in dataset: {missing}. Available columns: {list(df.columns)}"
        )


def resolve_path(path_str: str, base_dir: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (base_dir / p)


def main() -> None:
    args = parse_args()

    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must be a float in the interval (0, 1).")
    if args.ngram_max < 1:
        raise ValueError("--ngram-max must be >= 1.")
    if args.min_df < 1:
        raise ValueError("--min-df must be >= 1.")
    if args.max_features < 100:
        raise ValueError("--max-features should be >= 100.")
    if args.c <= 0:
        raise ValueError("--c must be > 0.")

    # scripts/.. => MachineLearning-Back
    project_root = Path(__file__).resolve().parents[1]

    dataset_path = resolve_path(args.dataset, project_root)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    validate_columns(df, args.text_col, args.label_col)

    dup_count_before = df.duplicated(subset=[args.text_col, args.label_col]).sum()
    df = df.drop_duplicates(subset=[args.text_col, args.label_col]).reset_index(drop=True)

    df = df[[args.text_col, args.label_col]].dropna().copy()
    df[args.text_col] = df[args.text_col].astype(str).map(clean_text)
    df = df[df[args.text_col].str.len() > 0]

    if df.empty:
        raise ValueError("Dataset is empty after cleaning.")

    X = df[args.text_col]
    y = df[args.label_col].astype(str)

    label_counts = y.value_counts().to_dict()
    if len(label_counts) < 2:
        raise ValueError("At least 2 different labels are required for classification.")

    min_label_count = min(label_counts.values())
    use_stratify = min_label_count >= 2
    if not use_stratify:
        print(
            "Warning: At least one class has <2 samples; split will run without stratification."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y if use_stratify else None,
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, args.ngram_max),
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=0.95,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    class_weight = None if args.class_weight == "none" else "balanced"
    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        C=args.c,
        class_weight=class_weight,
        random_state=args.random_state,
    )
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    overlap = set(X_train).intersection(set(X_test))

    cv_results = None
    if args.cv_folds and args.cv_folds >= 2:
        min_train_label_count = y_train.value_counts().min()
        if min_train_label_count >= args.cv_folds:
            cv_pipeline = Pipeline(
                steps=[
                    (
                        "vectorizer",
                        TfidfVectorizer(
                            ngram_range=(1, args.ngram_max),
                            max_features=args.max_features,
                            min_df=args.min_df,
                            max_df=0.95,
                        ),
                    ),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1000,
                            solver="lbfgs",
                            C=args.c,
                            class_weight=class_weight,
                            random_state=args.random_state,
                        ),
                    ),
                ]
            )
            cv = StratifiedKFold(
                n_splits=args.cv_folds,
                shuffle=True,
                random_state=args.random_state,
            )
            cv_scores = cross_val_score(
                cv_pipeline,
                X_train,
                y_train,
                scoring="f1_weighted",
                cv=cv,
                n_jobs=-1,
            )
            cv_results = {
                "folds": int(args.cv_folds),
                "scoring": "f1_weighted",
                "mean": round(float(cv_scores.mean()), 4),
                "std": round(float(cv_scores.std()), 4),
                "scores": [round(float(score), 4) for score in cv_scores.tolist()],
            }
        else:
            print(
                "Warning: Not enough samples per class in train split for requested CV folds; "
                "skipping cross-validation."
            )

    metrics = {
        "dataset_path": str(dataset_path),
        "rows_used": int(len(df)),
        "duplicates_removed": int(dup_count_before),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "tfidf": {
            "ngram_range": [1, int(args.ngram_max)],
            "min_df": int(args.min_df),
            "max_df": 0.95,
            "max_features": int(args.max_features),
        },
        "model": {
            "type": "LogisticRegression",
            "C": float(args.c),
            "solver": "lbfgs",
            "class_weight": args.class_weight,
            "max_iter": 1000,
        },
        "labels": labels,
        "label_distribution_total": y.value_counts().to_dict(),
        "label_distribution_train": y_train.value_counts().to_dict(),
        "label_distribution_test": y_test.value_counts().to_dict(),
        "train_test_text_overlap_count": int(len(overlap)),
        "accuracy": round(float(accuracy), 4),
        "precision_weighted": round(float(precision), 4),
        "recall_weighted": round(float(recall), 4),
        "f1_weighted": round(float(f1), 4),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": cm.tolist(),
    }
    if cv_results is not None:
        metrics["cross_validation"] = cv_results

    model_out = resolve_path(args.model_out, project_root)
    vectorizer_out = resolve_path(args.vectorizer_out, project_root)
    metrics_out = resolve_path(args.metrics_out, project_root)

    # compute dataset checksum
    def sha256_of_file(p: Path) -> str:
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    dataset_checksum = sha256_of_file(dataset_path)

    # If metrics exist, compare checksum and skip if same (unless --overwrite)
    if metrics_out.exists() and model_out.exists() and vectorizer_out.exists() and not args.overwrite:
        try:
            with metrics_out.open("r", encoding="utf-8") as f:
                prev = json.load(f)
            if prev.get("dataset_checksum") == dataset_checksum:
                print("Artefactos existentes y dataset sin cambios. Salvo por --overwrite, no se reentrena.")
                return
        except Exception:
            # si no se puede leer metrics, continuar y reentrenar
            pass

    # ...existing training code...

    # al guardar métricas, incluir checksum
    metrics["dataset_checksum"] = dataset_checksum
    with metrics_out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    vectorizer_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    with model_out.open("wb") as f:
        pickle.dump(model, f)

    with vectorizer_out.open("wb") as f:
        pickle.dump(vectorizer, f)

    with metrics_out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Training complete.")
    print(f"Model saved to: {model_out}")
    print(f"Vectorizer saved to: {vectorizer_out}")
    print(f"Metrics saved to: {metrics_out}")
    print(
        "Summary -> "
        f"accuracy={metrics['accuracy']}, "
        f"precision={metrics['precision_weighted']}, "
        f"recall={metrics['recall_weighted']}, "
        f"f1={metrics['f1_weighted']}"
    )

    print("Filas:", len(df))
    print(
        f"Duplicados exactos ({args.text_col}+{args.label_col}) antes de deduplicar:",
        int(dup_count_before),
    )
    print(f"Textos únicos ({args.text_col}):", df[args.text_col].nunique())
    print("Textos iguales en train y test:", len(overlap))
    if cv_results is not None:
        print(
            "CV (f1_weighted) -> "
            f"mean={cv_results['mean']}, std={cv_results['std']}, folds={cv_results['folds']}"
        )


if __name__ == "__main__":
    main()
