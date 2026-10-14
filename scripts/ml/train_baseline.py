"""Treina e avalia a baseline TF-IDF + regressão logística na validação."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data" / ".matplotlib"))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import FeatureUnion, Pipeline  # noqa: E402
from language_baseline_common import (  # noqa: E402
    calculate_metrics, file_sha256, length_band, load_split, subset_metrics,
)


SEED = 42
THRESHOLD = 0.5
WORD_PARAMS = {
    "analyzer": "word",
    "ngram_range": (1, 2),
    "min_df": 3,
    "max_df": 0.98,
    "max_features": 60_000,
    "lowercase": True,
    "sublinear_tf": True,
    "norm": "l2",
    "dtype": np.float32,
}
CHAR_PARAMS = {
    "analyzer": "char_wb",
    "ngram_range": (3, 5),
    "min_df": 3,
    "max_df": 0.98,
    "max_features": 60_000,
    "lowercase": True,
    "sublinear_tf": True,
    "norm": "l2",
    "dtype": np.float32,
}
MODEL_PARAMS = {
    "C": 1.0,
    "l1_ratio": 0.0,
    "solver": "liblinear",
    "max_iter": 1_000,
    "tol": 1e-4,
    "fit_intercept": True,
    "class_weight": None,
    "random_state": SEED,
}


def build_pipeline() -> Pipeline:
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(**WORD_PARAMS)),
            ("char", TfidfVectorizer(**CHAR_PARAMS)),
        ],
        n_jobs=-1,
    )
    classifier = LogisticRegression(**MODEL_PARAMS)
    return Pipeline([("features", features), ("classifier", classifier)])


def top_coefficients(pipeline: Pipeline, limit: int = 20) -> dict:
    names = pipeline.named_steps["features"].get_feature_names_out()
    coefficients = pipeline.named_steps["classifier"].coef_[0]
    positive = np.argsort(coefficients)[-limit:][::-1]
    negative = np.argsort(coefficients)[:limit]

    def items(indices):
        return [
            {"feature": str(names[index]), "coefficient": round(float(coefficients[index]), 6)}
            for index in indices
        ]

    return {"toward_fake": items(positive), "toward_true": items(negative)}


def plot_results(metrics: dict, labels, probabilities, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    figure.suptitle(
        "Baseline TF-IDF + Regressão Logística — conjunto de validação",
        fontsize=15,
        fontweight="bold",
    )

    metric_names = ["Acurácia", "Precisão macro", "Recall macro", "F1 macro", "ROC-AUC"]
    metric_values = [
        metrics["accuracy"],
        metrics["precision_macro"],
        metrics["recall_macro"],
        metrics["f1_macro"],
        metrics["roc_auc"],
    ]
    bars = axes[0].barh(metric_names, metric_values, color="#315c7d")
    axes[0].set_xlim(0, 1)
    axes[0].set_title("Métricas globais")
    axes[0].set_xlabel("Valor")
    axes[0].bar_label(bars, labels=[f"{value:.3f}" for value in metric_values], padding=3)

    matrix = np.asarray(metrics["confusion_matrix"])
    image = axes[1].imshow(matrix, cmap="Blues")
    axes[1].set_title("Matriz de confusão")
    axes[1].set_xlabel("Predição")
    axes[1].set_ylabel("Rótulo real")
    axes[1].set_xticks([0, 1], ["true", "fake"])
    axes[1].set_yticks([0, 1], ["true", "fake"])
    for row in range(2):
        for column in range(2):
            color = "white" if matrix[row, column] > matrix.max() / 2 else "black"
            axes[1].text(column, row, f"{matrix[row, column]:,}".replace(",", "."),
                         ha="center", va="center", color=color, fontsize=12, fontweight="bold")
    figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)

    labels_array = np.asarray(labels)
    probabilities_array = np.asarray(probabilities)
    bins = np.linspace(0, 1, 21)
    axes[2].hist(
        probabilities_array[labels_array == 0], bins=bins, alpha=0.65,
        label="Rótulo true", color="#3a86a8",
    )
    axes[2].hist(
        probabilities_array[labels_array == 1], bins=bins, alpha=0.65,
        label="Rótulo fake", color="#d66b3d",
    )
    axes[2].axvline(THRESHOLD, color="black", linestyle="--", label="Limiar 0,5")
    axes[2].set_title("Probabilidade prevista de risco")
    axes[2].set_xlabel("P(fake | texto)")
    axes[2].set_ylabel("Notícias")
    axes[2].legend()

    figure.tight_layout(rect=(0, 0, 1, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def serializable_params(params: dict) -> dict:
    return {
        key: str(value) if key == "dtype" else list(value) if isinstance(value, tuple) else value
        for key, value in params.items()
    }


def train(
    train_path: Path,
    validation_path: Path,
    model_path: Path,
    metrics_path: Path,
    figure_path: Path,
) -> dict:
    train_records = load_split(train_path)
    validation_records = load_split(validation_path)
    train_texts = [record["text"] for record in train_records]
    train_labels = np.asarray([record["label_id"] for record in train_records])
    validation_texts = [record["text"] for record in validation_records]
    validation_labels = np.asarray([record["label_id"] for record in validation_records])

    pipeline = build_pipeline()
    started = time.perf_counter()
    pipeline.fit(train_texts, train_labels)
    elapsed = time.perf_counter() - started

    fake_column = list(pipeline.named_steps["classifier"].classes_).index(1)
    probabilities = pipeline.predict_proba(validation_texts)[:, fake_column]
    predictions = (probabilities >= THRESHOLD).astype(int)
    overall = calculate_metrics(validation_labels, predictions, probabilities)

    by_dataset = {
        dataset: subset_metrics(
            validation_records,
            validation_labels,
            predictions,
            probabilities,
            lambda record, expected=dataset: record["dataset"] == expected,
        )
        for dataset in sorted({record["dataset"] for record in validation_records})
    }
    by_length = {
        band: subset_metrics(
            validation_records,
            validation_labels,
            predictions,
            probabilities,
            lambda record, expected=band: length_band(record) == expected,
        )
        for band in ("short_1_50", "medium_51_150", "long_151_plus")
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path, compress=3)
    plot_results(overall, validation_labels, probabilities, figure_path)

    features = pipeline.named_steps["features"]
    classifier = pipeline.named_steps["classifier"]
    report = {
        "experiment": "baseline_tfidf_logistic_regression",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_split_used": False,
        "data": {
            "train": {
                "path": str(train_path.relative_to(ROOT)),
                "sha256": file_sha256(train_path),
                "records": len(train_records),
                "labels": dict(sorted(Counter(map(int, train_labels)).items())),
            },
            "validation": {
                "path": str(validation_path.relative_to(ROOT)),
                "sha256": file_sha256(validation_path),
                "records": len(validation_records),
                "labels": dict(sorted(Counter(map(int, validation_labels)).items())),
            },
        },
        "parameters": {
            "threshold": THRESHOLD,
            "feature_union": {"n_jobs": -1},
            "word_tfidf": serializable_params(WORD_PARAMS),
            "character_tfidf": serializable_params(CHAR_PARAMS),
            "logistic_regression": MODEL_PARAMS,
            "length_bands_words": {
                "short_1_50": "1 to 50",
                "medium_51_150": "51 to 150",
                "long_151_plus": "151 or more",
            },
        },
        "training": {
            "elapsed_seconds": round(elapsed, 3),
            "word_features": len(features.transformer_list[0][1].vocabulary_),
            "character_features": len(features.transformer_list[1][1].vocabulary_),
            "total_features": int(classifier.coef_.shape[1]),
            "iterations": int(classifier.n_iter_[0]),
            "converged": bool(classifier.n_iter_[0] < MODEL_PARAMS["max_iter"]),
        },
        "metrics": {
            "overall": overall,
            "by_dataset": by_dataset,
            "by_length": by_length,
        },
        "top_coefficients": top_coefficients(pipeline),
        "artifacts": {
            "model": {
                "path": str(model_path.relative_to(ROOT)),
                "sha256": file_sha256(model_path),
                "bytes": model_path.stat().st_size,
            },
            "figure": str(figure_path.relative_to(ROOT)),
        },
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        type=Path,
        default=ROOT / "data/processed/language_risk/splits/train.jsonl",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=ROOT / "data/processed/language_risk/splits/validation.jsonl",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=ROOT / "models/language_risk/baseline_tfidf_logreg.joblib",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=ROOT / "data/processed/language_risk/baseline/metrics.json",
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=ROOT / "reports/figures/11_baseline_regressao_logistica.png",
    )
    args = parser.parse_args()

    report = train(
        args.train,
        args.validation,
        args.model_output,
        args.metrics_output,
        args.figure_output,
    )
    print(f"F1 macro (validação): {report['metrics']['overall']['f1_macro']:.4f}")
    print(f"Modelo local: {args.model_output}")
    print(f"Métricas locais: {args.metrics_output}")
    print(f"Gráfico: {args.figure_output}")
    print("Split de teste utilizado: não")


if __name__ == "__main__":
    main()
