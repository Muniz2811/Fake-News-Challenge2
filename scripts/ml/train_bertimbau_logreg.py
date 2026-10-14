"""Treina regressão logística nos embeddings congelados do BERTimbau."""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from language_baseline_common import (
    calculate_metrics,
    file_sha256,
    length_band,
    load_split,
    subset_metrics,
)


ROOT = Path(__file__).resolve().parents[2]
THRESHOLD = 0.5
MODEL_PARAMS = {
    "C": 1.0,
    "solver": "liblinear",
    "max_iter": 1000,
    "class_weight": None,
    "random_state": 42,
}


def load_embeddings(
    split_path: Path, embedding_dir: Path
) -> tuple[list[dict], np.ndarray, dict]:
    records = load_split(split_path)
    manifest_path = embedding_dir / f"{split_path.stem}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise ValueError(f"Embeddings incompletos para {split_path.stem}")
    if manifest.get("source_sha256") != file_sha256(split_path):
        raise ValueError(f"Split de {split_path.stem} mudou desde a vetorização")
    if manifest.get("records") != len(records) or manifest.get("completed") != len(
        records
    ):
        raise ValueError(f"Quantidade divergente em {split_path.stem}")
    vector_path = embedding_dir / f"{split_path.stem}.npy"
    if manifest.get("vectors_sha256") != file_sha256(vector_path):
        raise ValueError(f"Embeddings de {split_path.stem} foram alterados")
    vectors = np.load(vector_path, mmap_mode="r")
    if (
        vectors.shape != (len(records), manifest["dimension"])
        or vectors.dtype != np.float32
    ):
        raise ValueError(f"Matriz inválida em {split_path.stem}")
    if not np.isfinite(vectors).all():
        raise ValueError(f"Matriz contém valores inválidos em {split_path.stem}")
    return records, vectors, manifest


def train(
    splits_dir: Path,
    embedding_dir: Path,
    model_path: Path,
    metrics_path: Path,
    baseline_metrics_path: Path,
) -> dict:
    train_path = splits_dir / "train.jsonl"
    validation_path = splits_dir / "validation.jsonl"
    train_records, train_vectors, train_manifest = load_embeddings(
        train_path, embedding_dir
    )
    validation_records, validation_vectors, validation_manifest = load_embeddings(
        validation_path,
        embedding_dir,
    )
    match_fields = (
        "model_id",
        "model_revision",
        "checkpoint_sha256",
        "pooling",
        "max_tokens",
        "dimension",
    )
    if any(
        train_manifest[field] != validation_manifest[field] for field in match_fields
    ):
        raise ValueError(
            "Treino e validação usam configurações de embeddings diferentes"
        )
    train_labels = np.asarray([record["label_id"] for record in train_records])
    validation_labels = np.asarray(
        [record["label_id"] for record in validation_records]
    )
    if set(train_labels) != {0, 1} or set(validation_labels) != {0, 1}:
        raise ValueError("Ambos os splits devem conter as classes 0=true e 1=fake")

    baseline = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    for split, manifest in (
        ("train", train_manifest),
        ("validation", validation_manifest),
    ):
        if baseline["data"][split]["sha256"] != manifest["source_sha256"]:
            raise ValueError(f"TF-IDF usa um split diferente para {split}")

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(**MODEL_PARAMS)),
        ]
    )
    started = time.perf_counter()
    pipeline.fit(train_vectors, train_labels)
    elapsed = time.perf_counter() - started
    fake_column = list(pipeline.classes_).index(1)
    probabilities = pipeline.predict_proba(validation_vectors)[:, fake_column]
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

    comparison = {
        "baseline_experiment": baseline["experiment"],
        "tfidf_validation": baseline["metrics"]["overall"],
        "bertimbau_validation": overall,
        "delta_bertimbau_minus_tfidf": {
            key: round(overall[key] - baseline["metrics"]["overall"][key], 6)
            for key in (
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "f1_fake",
                "roc_auc",
                "brier_score",
            )
        },
        "f1_macro_by_dataset": {
            dataset: {
                "tfidf": baseline["metrics"]["by_dataset"][dataset]["f1_macro"],
                "bertimbau": metrics["f1_macro"],
                "delta": round(
                    metrics["f1_macro"]
                    - baseline["metrics"]["by_dataset"][dataset]["f1_macro"],
                    6,
                ),
            }
            for dataset, metrics in by_dataset.items()
        },
        "f1_macro_by_length": {
            band: {
                "tfidf": baseline["metrics"]["by_length"][band]["f1_macro"],
                "bertimbau": metrics["f1_macro"],
                "delta": round(
                    metrics["f1_macro"]
                    - baseline["metrics"]["by_length"][band]["f1_macro"],
                    6,
                ),
            }
            for band, metrics in by_length.items()
        },
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path, compress=3)
    report = {
        "experiment": "baseline_bertimbau_frozen_logistic_regression",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_split_used": False,
        "data": {
            split: {
                "path": str((splits_dir / f"{split}.jsonl").relative_to(ROOT)),
                "sha256": manifest["source_sha256"],
                "records": manifest["records"],
                "labels": dict(
                    sorted(Counter(int(item["label_id"]) for item in records).items())
                ),
            }
            for split, records, manifest in (
                ("train", train_records, train_manifest),
                ("validation", validation_records, validation_manifest),
            )
        },
        "encoder": {key: train_manifest[key] for key in match_fields},
        "parameters": {
            "threshold": THRESHOLD,
            "standard_scaler": True,
            "logistic_regression": MODEL_PARAMS,
        },
        "training": {
            "elapsed_seconds": round(elapsed, 3),
            "iterations": int(pipeline.named_steps["classifier"].n_iter_[0]),
            "converged": bool(
                pipeline.named_steps["classifier"].n_iter_[0] < MODEL_PARAMS["max_iter"]
            ),
        },
        "metrics": {
            "overall": overall,
            "by_dataset": by_dataset,
            "by_length": by_length,
        },
        "comparison": comparison,
        "artifacts": {
            "model": {
                "path": str(model_path.relative_to(ROOT)),
                "sha256": file_sha256(model_path),
                "bytes": model_path.stat().st_size,
            },
        },
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
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
        "--splits-dir", type=Path, default=ROOT / "data/processed/language_risk/splits"
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=ROOT / "data/processed/language_risk/bertimbau",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=ROOT / "models/language_risk/bertimbau_logreg.joblib",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=ROOT / "data/processed/language_risk/bertimbau/metrics.json",
    )
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=ROOT / "data/processed/language_risk/baseline/metrics.json",
    )
    args = parser.parse_args()
    report = train(
        args.splits_dir,
        args.embedding_dir,
        args.model_output,
        args.metrics_output,
        args.baseline_metrics,
    )
    metrics = report["metrics"]["overall"]
    delta = report["comparison"]["delta_bertimbau_minus_tfidf"]["f1_macro"]
    tfidf_f1 = report["comparison"]["tfidf_validation"]["f1_macro"]
    print(f"F1 macro BERTimbau + regressão logística: {metrics['f1_macro']:.4f}")
    print(f"F1 macro TF-IDF + regressão logística: {tfidf_f1:.4f}")
    print(f"Diferença BERTimbau - TF-IDF: {delta:+.4f}")
    print(f"Métricas locais: {args.metrics_output}")
    print("Split de teste utilizado: não")


if __name__ == "__main__":
    main()
