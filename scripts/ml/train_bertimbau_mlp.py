"""Compara uma MLP pequena com as baselines nos mesmos embeddings BERTimbau."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from language_baseline_common import (
    calculate_metrics,
    file_sha256,
    length_band,
    subset_metrics,
)
from train_bertimbau_logreg import load_embeddings


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "contraponto"))

from bertimbau_mlp import build_model  # noqa: E402

THRESHOLD = 0.5
ENCODER_FIELDS = (
    "model_id",
    "model_revision",
    "checkpoint_sha256",
    "pooling",
    "max_tokens",
    "dimension",
)


def select_device(torch, requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA indisponível neste ambiente")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS indisponível neste ambiente")
    return requested


def validate_inputs(
    splits_dir: Path,
    embedding_dir: Path,
    baseline_metrics_path: Path,
    logreg_metrics_path: Path,
):
    train_records, train_vectors, train_manifest = load_embeddings(
        splits_dir / "train.jsonl", embedding_dir
    )
    validation_records, validation_vectors, validation_manifest = load_embeddings(
        splits_dir / "validation.jsonl", embedding_dir
    )
    if any(
        train_manifest[field] != validation_manifest[field]
        for field in ENCODER_FIELDS
    ):
        raise ValueError("Treino e validação usam embeddings incompatíveis")

    baseline = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    logreg = json.loads(logreg_metrics_path.read_text(encoding="utf-8"))
    for split, manifest in (
        ("train", train_manifest),
        ("validation", validation_manifest),
    ):
        for name, report in (("TF-IDF", baseline), ("BERTimbau + regressão", logreg)):
            if report["data"][split]["sha256"] != manifest["source_sha256"]:
                raise ValueError(f"{name} usa outro split de {split}")
    if any(
        logreg["encoder"][field] != train_manifest[field]
        for field in ENCODER_FIELDS
    ):
        raise ValueError("Regressão e MLP usam embeddings diferentes")

    train_labels = np.asarray([row["label_id"] for row in train_records])
    validation_labels = np.asarray(
        [row["label_id"] for row in validation_records]
    )
    if set(train_labels) != {0, 1} or set(validation_labels) != {0, 1}:
        raise ValueError("Treino e validação devem conter as classes 0 e 1")
    return (
        train_records,
        train_vectors,
        train_manifest,
        train_labels,
        validation_records,
        validation_vectors,
        validation_manifest,
        validation_labels,
        baseline,
        logreg,
    )


def evaluate(model, vectors, labels, torch, device: str, batch_size: int):
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(vectors), batch_size):
            batch = vectors[start : start + batch_size].to(device)
            probabilities.append(torch.sigmoid(model(batch)).cpu().numpy().ravel())
    probabilities = np.concatenate(probabilities)
    predictions = (probabilities >= THRESHOLD).astype(int)
    return calculate_metrics(labels, predictions, probabilities), predictions, probabilities


def train_seed(
    train_vectors,
    train_labels,
    validation_vectors,
    validation_labels,
    scaler,
    model_path: Path,
    *,
    seed: int,
    hidden_size: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    device: str,
):
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = build_model(train_vectors.shape[1], hidden_size, dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_function = torch.nn.BCEWithLogitsLoss()
    train_tensor = torch.from_numpy(train_vectors)
    label_tensor = torch.from_numpy(train_labels.astype(np.float32))
    validation_tensor = torch.from_numpy(validation_vectors)
    generator = torch.Generator().manual_seed(seed)
    best = None
    stale_epochs = 0
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(train_tensor), generator=generator)
        for indices in order.split(batch_size):
            batch = train_tensor[indices].to(device)
            labels = label_tensor[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch).flatten(), labels)
            loss.backward()
            optimizer.step()

        metrics, predictions, probabilities = evaluate(
            model,
            validation_tensor,
            validation_labels,
            torch,
            device,
            batch_size,
        )
        # A validação seleciona a época; o teste continua reservado.
        rank = (metrics["f1_macro"], -metrics["brier_score"])
        if best is None or rank > best["rank"]:
            best = {
                "rank": rank,
                "epoch": epoch,
                "metrics": metrics,
                "predictions": predictions,
                "probabilities": probabilities,
                "state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best["state_dict"],
            "scaler_mean": torch.from_numpy(scaler.mean_.astype(np.float32)),
            "scaler_scale": torch.from_numpy(scaler.scale_.astype(np.float32)),
            "dimension": train_vectors.shape[1],
            "hidden_size": hidden_size,
            "dropout": dropout,
            "threshold": THRESHOLD,
            "seed": seed,
            "best_epoch": best["epoch"],
        },
        model_path,
    )
    return {
        "seed": seed,
        "best_epoch": best["epoch"],
        "epochs_run": epoch,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "metrics": best["metrics"],
        "predictions": best["predictions"],
        "probabilities": best["probabilities"],
        "artifact": {
            "path": str(model_path.relative_to(ROOT)),
            "sha256": file_sha256(model_path),
            "bytes": model_path.stat().st_size,
        },
    }


def train(
    splits_dir: Path,
    embedding_dir: Path,
    model_dir: Path,
    metrics_path: Path,
    baseline_metrics_path: Path,
    logreg_metrics_path: Path,
    *,
    seeds: tuple[int, ...] = (42, 43, 44),
    hidden_size: int = 128,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    batch_size: int = 256,
    max_epochs: int = 30,
    patience: int = 5,
    device: str = "auto",
) -> dict:
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or hidden_size < 1
        or not 0 <= dropout < 1
        or learning_rate <= 0
        or weight_decay < 0
        or batch_size < 1
        or max_epochs < 1
        or patience < 1
    ):
        raise ValueError("Hiperparâmetros ou sementes inválidos")
    import torch

    selected_device = select_device(torch, device)
    (
        train_records,
        train_vectors,
        train_manifest,
        train_labels,
        validation_records,
        validation_vectors,
        validation_manifest,
        validation_labels,
        baseline,
        logreg,
    ) = validate_inputs(
        splits_dir, embedding_dir, baseline_metrics_path, logreg_metrics_path
    )
    scaler = StandardScaler()
    train_scaled = np.asarray(scaler.fit_transform(train_vectors), dtype=np.float32)
    validation_scaled = np.asarray(
        scaler.transform(validation_vectors), dtype=np.float32
    )
    runs = []
    for seed in seeds:
        print(f"Treinando MLP: semente {seed}, dispositivo {selected_device}", flush=True)
        run = train_seed(
            train_scaled,
            train_labels,
            validation_scaled,
            validation_labels,
            scaler,
            model_dir / f"bertimbau_mlp_seed{seed}.pt",
            seed=seed,
            hidden_size=hidden_size,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            device=selected_device,
        )
        print(
            f"Semente {seed}: época {run['best_epoch']}, "
            f"F1 macro {run['metrics']['f1_macro']:.4f}",
            flush=True,
        )
        runs.append(run)

    datasets = sorted({row["dataset"] for row in validation_records})
    bands = ("short_1_50", "medium_51_150", "long_151_plus")
    for run in runs:
        run["by_dataset"] = {
            dataset: subset_metrics(
                validation_records,
                validation_labels,
                run["predictions"],
                run["probabilities"],
                lambda row, expected=dataset: row["dataset"] == expected,
            )
            for dataset in datasets
        }
        run["by_length"] = {
            band: subset_metrics(
                validation_records,
                validation_labels,
                run["predictions"],
                run["probabilities"],
                lambda row, expected=band: length_band(row) == expected,
            )
            for band in bands
        }

    # O melhor modelo é apontado apenas para análise; média e desvio consideram
    # todas as sementes, sem escolher uma execução favorável como resultado geral.
    best_run = max(runs, key=lambda run: (
        run["metrics"]["f1_macro"], -run["metrics"]["brier_score"]
    ))
    metric_names = (
        "accuracy", "precision_macro", "recall_macro", "f1_macro",
        "f1_fake", "roc_auc", "average_precision_fake", "brier_score",
    )
    summary = {
        key: {
            "mean": round(statistics.mean(run["metrics"][key] for run in runs), 6),
            "std_population": round(
                statistics.pstdev(run["metrics"][key] for run in runs), 6
            ),
        }
        for key in metric_names
    }
    subgroup_summary = {
        grouping: {
            subgroup: {
                "f1_macro_mean": round(
                    statistics.mean(
                        run[grouping][subgroup]["f1_macro"] for run in runs
                    ), 6
                ),
                "f1_macro_std_population": round(
                    statistics.pstdev(
                        run[grouping][subgroup]["f1_macro"] for run in runs
                    ), 6
                ),
            }
            for subgroup in subgroups
        }
        for grouping, subgroups in (("by_dataset", datasets), ("by_length", bands))
    }
    report = {
        "experiment": "bertimbau_frozen_mlp",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_split_used": False,
        "data": {
            split: {
                "path": str((splits_dir / f"{split}.jsonl").relative_to(ROOT)),
                "sha256": manifest["source_sha256"],
                "records": len(records),
            }
            for split, records, manifest in (
                ("train", train_records, train_manifest),
                ("validation", validation_records, validation_manifest),
            )
        },
        "encoder": {field: train_manifest[field] for field in ENCODER_FIELDS},
        "parameters": {
            "architecture": f"{train_vectors.shape[1]}-{hidden_size}-1",
            "activation": "ReLU",
            "dropout": dropout,
            "standard_scaler_fit_on_train_only": True,
            "loss": "BCEWithLogitsLoss",
            "optimizer": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "patience": patience,
            "early_stopping_metric": "validation_f1_macro_then_brier",
            "threshold": THRESHOLD,
            "seeds": list(seeds),
            "device": selected_device,
        },
        "runs": [
            {key: value for key, value in run.items()
             if key not in ("predictions", "probabilities")}
            for run in runs
        ],
        "validation_summary": summary,
        "validation_subgroup_summary": subgroup_summary,
        "best_validation_run": {
            "seed": best_run["seed"],
            "metrics": best_run["metrics"],
            "by_dataset": best_run["by_dataset"],
            "by_length": best_run["by_length"],
            "note": "Seleção e recortes na mesma validação; não são teste independente.",
        },
        "comparison": {
            "tfidf_logreg": baseline["metrics"]["overall"],
            "bertimbau_logreg": logreg["metrics"]["overall"],
            "bertimbau_mlp_mean": {
                key: values["mean"] for key, values in summary.items()
            },
            "f1_macro_delta_mean_mlp_minus_bertimbau_logreg": round(
                summary["f1_macro"]["mean"]
                - logreg["metrics"]["overall"]["f1_macro"], 6
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
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
        "--embedding-dir", type=Path,
        default=ROOT / "data/processed/language_risk/bertimbau",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=ROOT / "models/language_risk"
    )
    parser.add_argument(
        "--metrics-output", type=Path,
        default=ROOT / "data/processed/language_risk/bertimbau/metrics_mlp.json",
    )
    parser.add_argument(
        "--baseline-metrics", type=Path,
        default=ROOT / "data/processed/language_risk/baseline/metrics.json",
    )
    parser.add_argument(
        "--logreg-metrics", type=Path,
        default=ROOT / "data/processed/language_risk/bertimbau/metrics.json",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    try:
        report = train(
            args.splits_dir, args.embedding_dir, args.model_dir,
            args.metrics_output, args.baseline_metrics, args.logreg_metrics,
            seeds=tuple(args.seeds), hidden_size=args.hidden_size,
            dropout=args.dropout, learning_rate=args.learning_rate,
            weight_decay=args.weight_decay, batch_size=args.batch_size,
            max_epochs=args.max_epochs, patience=args.patience, device=args.device,
        )
    except ValueError as error:
        parser.error(str(error))
    mlp = report["validation_summary"]["f1_macro"]
    print(f"F1 macro MLP (média ± desvio): {mlp['mean']:.4f} ± {mlp['std_population']:.4f}")
    print(f"F1 macro BERTimbau + regressão: {report['comparison']['bertimbau_logreg']['f1_macro']:.4f}")
    print(f"F1 macro TF-IDF + regressão: {report['comparison']['tfidf_logreg']['f1_macro']:.4f}")
    print(f"Métricas locais: {args.metrics_output}")
    print("Split de teste utilizado: não")


if __name__ == "__main__":
    main()
