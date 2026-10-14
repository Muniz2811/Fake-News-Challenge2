"""Avalia no teste o checkpoint MLP já escolhido pela validação, sem treino."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from language_baseline_common import (
    calculate_metrics,
    file_sha256,
    length_band,
    subset_metrics,
)
from train_bertimbau_logreg import load_embeddings
from train_bertimbau_mlp import ENCODER_FIELDS, THRESHOLD, build_model


ROOT = Path(__file__).resolve().parents[2]


def evaluate_test(
    test_split_path: Path,
    embedding_dir: Path,
    mlp_metrics_path: Path,
    output_path: Path,
    batch_size: int = 256,
) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size deve ser positivo")
    if output_path.exists():
        raise FileExistsError(
            f"Relatório de teste já existe: {output_path}. "
            "Preserve essa avaliação; não use o teste para ajustar o modelo."
        )
    import torch

    selection = json.loads(mlp_metrics_path.read_text(encoding="utf-8"))
    if selection["test_split_used"] or selection["experiment"] != "bertimbau_frozen_mlp":
        raise ValueError("Relatório de seleção inválido")
    chosen_seed = selection["best_validation_run"]["seed"]
    chosen_run = next(
        run for run in selection["runs"] if run["seed"] == chosen_seed
    )
    model_path = ROOT / chosen_run["artifact"]["path"]
    if file_sha256(model_path) != chosen_run["artifact"]["sha256"]:
        raise ValueError("Checkpoint MLP difere do selecionado na validação")
    records, vectors, manifest = load_embeddings(test_split_path, embedding_dir)
    if any(selection["encoder"][field] != manifest[field] for field in ENCODER_FIELDS):
        raise ValueError("Teste foi vetorizado com outro encoder")
    if manifest["source_sha256"] in {
        selection["data"]["train"]["sha256"],
        selection["data"]["validation"]["sha256"],
    }:
        raise ValueError("Teste coincide com um split usado no treino ou validação")
    labels = np.asarray([row["label_id"] for row in records])
    if set(labels) != {0, 1}:
        raise ValueError("O teste deve conter as duas classes")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if checkpoint["seed"] != chosen_seed or checkpoint["dimension"] != manifest["dimension"]:
        raise ValueError("Checkpoint MLP incompatível com o teste")
    if checkpoint["threshold"] != THRESHOLD:
        raise ValueError("Limiar do checkpoint difere do experimento")
    model = build_model(
        checkpoint["dimension"], checkpoint["hidden_size"], checkpoint["dropout"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    means = checkpoint["scaler_mean"].numpy()
    scales = checkpoint["scaler_scale"].numpy()
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(vectors), batch_size):
            batch = np.asarray(vectors[start : start + batch_size], dtype=np.float32)
            normalized = (batch - means) / scales
            logits = model(torch.from_numpy(normalized))
            probabilities.append(torch.sigmoid(logits).numpy().ravel())
    probabilities = np.concatenate(probabilities)
    predictions = (probabilities >= THRESHOLD).astype(int)
    overall = calculate_metrics(labels, predictions, probabilities)
    by_dataset = {
        dataset: subset_metrics(
            records,
            labels,
            predictions,
            probabilities,
            lambda row, expected=dataset: row["dataset"] == expected,
        )
        for dataset in sorted({row["dataset"] for row in records})
    }
    by_length = {
        band: subset_metrics(
            records,
            labels,
            predictions,
            probabilities,
            lambda row, expected=band: length_band(row) == expected,
        )
        for band in ("short_1_50", "medium_51_150", "long_151_plus")
    }
    report = {
        "experiment": "bertimbau_frozen_mlp_heldout_test",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_selection": {
            "source_report": str(mlp_metrics_path.relative_to(ROOT)),
            "source_report_sha256": file_sha256(mlp_metrics_path),
            "criterion": "best_validation_f1_macro_then_brier",
            "selected_seed": chosen_seed,
            "selected_validation_f1_macro": chosen_run["metrics"]["f1_macro"],
            "checkpoint": chosen_run["artifact"],
        },
        "test": {
            "path": str(test_split_path.relative_to(ROOT)),
            "source_sha256": manifest["source_sha256"],
            "vectors_sha256": manifest["vectors_sha256"],
            "records": len(records),
        },
        "encoder": {field: manifest[field] for field in ENCODER_FIELDS},
        "metrics": {
            "overall": overall,
            "by_dataset": by_dataset,
            "by_length": by_length,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "note": "Avaliação única do checkpoint congelado; nenhuma otimização no teste.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-split", type=Path,
        default=ROOT / "data/processed/language_risk/splits/test.jsonl",
    )
    parser.add_argument(
        "--embedding-dir", type=Path,
        default=ROOT / "data/processed/language_risk/bertimbau",
    )
    parser.add_argument(
        "--mlp-metrics", type=Path,
        default=ROOT / "data/processed/language_risk/bertimbau/metrics_mlp.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/processed/language_risk/bertimbau/metrics_mlp_test.json",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    try:
        report = evaluate_test(
            args.test_split, args.embedding_dir, args.mlp_metrics,
            args.output, args.batch_size,
        )
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    metrics = report["metrics"]["overall"]
    print(f"Checkpoint selecionado: semente {report['model_selection']['selected_seed']}")
    print(f"F1 macro no teste: {metrics['f1_macro']:.4f}")
    print(f"Acurácia no teste: {metrics['accuracy']:.4f}")
    print(f"Métricas locais: {args.output}")


if __name__ == "__main__":
    main()
