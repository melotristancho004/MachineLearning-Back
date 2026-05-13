import argparse
import json
import pickle
import re
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
from sklearn.model_selection import train_test_split
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

    # scripts/.. => MachineLearning-Back
    project_root = Path(__file__).resolve().parents[1]

    dataset_path = resolve_path(args.dataset, project_root)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    validate_columns(df, args.text_col, args.label_col)

    df = df[[args.text_col, args.label_col]].dropna().copy()
    df[args.text_col] = df[args.text_col].astype(str).map(clean_text)
    df = df[df[args.text_col].str.len() > 0]

    if df.empty:
        raise ValueError("Dataset is empty after cleaning.")

    X = df[args.text_col]
    y = df[args.label_col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        min_df=2,
        max_df=0.95,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=100, solver="lbfgs")
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

    metrics = {
        "dataset_path": str(dataset_path),
        "rows_used": int(len(df)),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "labels": labels,
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

    model_out = resolve_path(args.model_out, project_root)
    vectorizer_out = resolve_path(args.vectorizer_out, project_root)
    metrics_out = resolve_path(args.metrics_out, project_root)

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


if __name__ == "__main__":
    main()
