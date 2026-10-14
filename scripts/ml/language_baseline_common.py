"""Splits e métricas compartilhados pelas baselines linguísticas."""

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            for field in ("id", "dataset", "text", "label_id", "split", "audit"):
                if field not in record:
                    raise ValueError(f"{path}:{line_number}: campo ausente: {field}")
            records.append(record)
    if not records:
        raise ValueError(f"Split vazio: {path}")
    expected_split = path.stem
    if any(record["split"] != expected_split for record in records):
        raise ValueError(f"{path}: há registros identificados com outro split")
    if any(
        type(record["label_id"]) is not int or record["label_id"] not in {0, 1}
        for record in records
    ):
        raise ValueError(f"{path}: rótulos devem ser 0=true ou 1=fake")
    return records


def calculate_metrics(labels, predictions, probabilities) -> dict:
    return {
        "records": int(len(labels)),
        "accuracy": round(float(accuracy_score(labels, predictions)), 6),
        "precision_macro": round(
            float(
                precision_score(labels, predictions, average="macro", zero_division=0)
            ),
            6,
        ),
        "recall_macro": round(
            float(recall_score(labels, predictions, average="macro", zero_division=0)),
            6,
        ),
        "f1_macro": round(
            float(f1_score(labels, predictions, average="macro", zero_division=0)), 6
        ),
        "f1_fake": round(
            float(f1_score(labels, predictions, pos_label=1, zero_division=0)), 6
        ),
        "roc_auc": round(float(roc_auc_score(labels, probabilities)), 6),
        "average_precision_fake": round(
            float(average_precision_score(labels, probabilities)), 6
        ),
        "brier_score": round(float(brier_score_loss(labels, probabilities)), 6),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
    }


def subset_metrics(records, labels, predictions, probabilities, selector) -> dict:
    indices = [index for index, record in enumerate(records) if selector(record)]
    subset_labels = np.asarray(labels)[indices]
    if len(set(subset_labels.tolist())) < 2:
        raise ValueError("Recorte de avaliação sem as duas classes")
    return calculate_metrics(
        subset_labels,
        np.asarray(predictions)[indices],
        np.asarray(probabilities)[indices],
    )


def length_band(record: dict) -> str:
    words = int(record["audit"]["word_count"])
    if words <= 50:
        return "short_1_50"
    if words <= 150:
        return "medium_51_150"
    return "long_151_plus"
