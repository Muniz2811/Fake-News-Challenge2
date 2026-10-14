"""Extrai embeddings BERTimbau por split com retomada por lote."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "contraponto"))

from bertimbau_embeddings import BertimbauEncoder  # noqa: E402
from language_baseline_common import file_sha256, load_split  # noqa: E402


MODEL_ID = "neuralmind/bert-base-portuguese-cased"
MODEL_REVISION = "94d69c95f98f7d5b2a8700c420230ae10def0baa"
MODEL_WEIGHTS_SHA256 = (
    "cb1693767adef60abf23d9fde3996f0c1e6310afad103a2db94ad44854568955"
)


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def extract_split(
    split_path: Path,
    output_dir: Path,
    encoder: BertimbauEncoder,
    document_batch_size: int,
    checkpoint_sha256: str,
    restart: bool = False,
) -> dict:
    records = load_split(split_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    split = split_path.stem
    vector_path = output_dir / f"{split}.npy"
    progress_path = output_dir / f"{split}.json"
    source_hash = file_sha256(split_path)
    identity = {
        "schema_version": "1.0",
        "split": split,
        "source_sha256": source_hash,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "checkpoint_sha256": checkpoint_sha256,
        "pooling": "mean of all content tokens across non-overlapping chunks",
        "max_tokens": encoder.max_tokens,
        "dimension": encoder.dimension,
        "records": len(records),
    }
    if progress_path.exists() and not restart:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if any(progress.get(key) != value for key, value in identity.items()):
            raise ValueError(f"Embeddings de {split} pertencem a outro experimento")
        vectors = np.load(vector_path, mmap_mode="r+")
        if (
            vectors.shape != (len(records), encoder.dimension)
            or vectors.dtype != np.float32
        ):
            raise ValueError(f"Matriz de {split} incompatível com o manifesto")
        completed = int(progress["completed"])
        if not 0 <= completed <= len(records):
            raise ValueError(f"Progresso inválido para {split}")
    else:
        if vector_path.exists() and not restart:
            raise ValueError(f"Matriz sem manifesto: {vector_path}")
        vectors = np.lib.format.open_memmap(
            vector_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(records), encoder.dimension),
        )
        completed = 0
        progress = {**identity, "completed": 0, "complete": False}
        write_json(progress_path, progress)

    started = time.monotonic()
    for start in range(completed, len(records), document_batch_size):
        end = min(start + document_batch_size, len(records))
        batch = encoder.encode([item["text"] for item in records[start:end]])
        if (
            batch.shape != (end - start, encoder.dimension)
            or not np.isfinite(batch).all()
        ):
            raise ValueError(
                f"Embeddings inválidos em {split}, registros {start}:{end}"
            )
        vectors[start:end] = batch
        vectors.flush()
        progress["completed"] = end
        progress["complete"] = end == len(records)
        progress["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(progress_path, progress)
        if end % (document_batch_size * 10) == 0 or end == len(records):
            elapsed = time.monotonic() - started
            rate = (end - completed) / elapsed if elapsed else 0.0
            print(
                f"{split}: {end}/{len(records)} notícias; {rate:.2f} notícias/s",
                flush=True,
            )
    if progress["complete"]:
        progress["vectors_sha256"] = file_sha256(vector_path)
        write_json(progress_path, progress)
    return progress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path, default=ROOT / "models/language_risk/bertimbau_base"
    )
    parser.add_argument(
        "--splits-dir", type=Path, default=ROOT / "data/processed/language_risk/splits"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/processed/language_risk/bertimbau",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=["train", "validation"],
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--batch-size", type=int, default=16, help="blocos por inferência"
    )
    parser.add_argument("--document-batch-size", type=int, default=64)
    parser.add_argument(
        "--restart",
        action="store_true",
        help="recalcula do zero os splits selecionados",
    )
    args = parser.parse_args()
    if args.document_batch_size < 1:
        parser.error("--document-batch-size deve ser positivo")
    checkpoint_file = args.model_dir / "pytorch_model.bin"
    if not checkpoint_file.is_file():
        parser.error(f"Checkpoint ausente: {checkpoint_file}")
    checkpoint_sha256 = file_sha256(checkpoint_file)
    if checkpoint_sha256 != MODEL_WEIGHTS_SHA256:
        parser.error("Os pesos locais não correspondem à revisão fixada do BERTimbau")
    encoder = BertimbauEncoder(
        args.model_dir, args.device, args.max_tokens, args.batch_size
    )
    print(f"Dispositivo: {encoder.device}; dimensão: {encoder.dimension}", flush=True)
    for split in args.splits:
        extract_split(
            args.splits_dir / f"{split}.jsonl",
            args.output_dir,
            encoder,
            args.document_batch_size,
            checkpoint_sha256,
            args.restart,
        )
    print(f"Embeddings locais: {args.output_dir}")
    print(f"Python: {platform.python_version()}")
    print(f"Split de teste vetorizado: {'sim' if 'test' in args.splits else 'não'}")


if __name__ == "__main__":
    main()
