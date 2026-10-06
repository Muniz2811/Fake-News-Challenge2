"""Prepara o corpus canônico do motor de risco linguístico.

Este passo somente integra e valida Fake.br e FakeRecogna 2.0. Ele não cria
splits e não treina modelos.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_RECORDS = {"fakebr": 7_200, "fakerecogna2": 52_800}
VALID_LABELS = {"fake", "true"}


def normalized_fingerprint(text: str) -> str:
    """Hash estável, ignorando caixa e diferenças triviais de espaços."""
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_stem(path: Path) -> int:
    if not path.stem.isdigit():
        raise ValueError(f"Nome de arquivo não numérico na Fake.br: {path}")
    return int(path.stem)


def domain_from_url(value: str | None) -> str | None:
    if not value or not value.strip():
        return None
    return urlsplit(value.strip()).netloc.casefold() or None


def url_fingerprint(value: str | None) -> str | None:
    """Preserva uma chave de agrupamento sem expor a URL ao modelo."""
    if not value or not value.strip():
        return None
    normalized = unicodedata.normalize("NFKC", value.strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def fakebr_metadata(root: Path, label: str, record_id: str) -> dict:
    path = root / "full_texts" / f"{label}-meta-information" / f"{record_id}-meta.txt"
    if not path.is_file():
        return {
            "source_domain": None,
            "source_url_sha256": None,
            "category": None,
            "published_at_raw": None,
            "pair_id": record_id,
            "metadata_missing": True,
        }
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if len(lines) != 25:
        raise ValueError(f"Esperadas 25 linhas de metadados em {path}; recebidas {len(lines)}")
    return {
        "source_domain": domain_from_url(lines[1]),
        "source_url_sha256": url_fingerprint(lines[1]),
        "category": lines[2].strip() or None,
        "published_at_raw": lines[3].strip() or None,
        "pair_id": record_id,
        "metadata_missing": False,
    }


def iter_fakebr(root: Path):
    for label in ("fake", "true"):
        directory = root / "size_normalized_texts" / label
        files = sorted(directory.glob("*.txt"), key=numeric_stem)
        if not files:
            raise FileNotFoundError(f"Nenhum texto encontrado em {directory}")

        for path in files:
            source_id = path.stem
            text = path.read_text(encoding="utf-8-sig").strip()
            audit = fakebr_metadata(root, label, source_id)
            audit.update({"word_count": len(text.split()), "character_count": len(text)})
            yield {
                "schema_version": "1.0",
                "id": f"fakebr:{label}:{source_id}",
                "dataset": "fakebr",
                "source_record_id": f"{label}/{source_id}",
                "language": "pt-BR",
                "text_variant": "size_normalized",
                "text": text,
                "label": label,
                "label_id": int(label == "fake"),
                "audit": audit,
            }


def iter_fakerecogna(path: Path):
    required = {"Unnamed: 0", "Categoria", "Data", "URL", "Noticia", "Label"}
    csv.field_size_limit(10_000_000)

    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Colunas ausentes em {path}: {sorted(missing)}")

        for row_number, row in enumerate(reader, 2):
            raw_label = (row.get("Label") or "").strip()
            if raw_label not in {"0", "1"}:
                raise ValueError(f"Label inválido em {path}:{row_number}: {raw_label!r}")

            label = "fake" if raw_label == "1" else "true"
            source_id = (row.get("Unnamed: 0") or "").strip() or str(row_number - 2)
            text = (row.get("Noticia") or "").strip()
            source_url = row.get("URL")
            yield {
                "schema_version": "1.0",
                "id": f"fakerecogna2:{source_id}",
                "dataset": "fakerecogna2",
                "source_record_id": source_id,
                "language": "pt-BR",
                "text_variant": "extractive_summary",
                "text": text,
                "label": label,
                "label_id": int(raw_label),
                "audit": {
                    "source_domain": domain_from_url(source_url),
                    "source_url_sha256": url_fingerprint(source_url),
                    "category": (row.get("Categoria") or "").strip() or None,
                    "published_at_raw": (row.get("Data") or "").strip() or None,
                    "word_count": len(text.split()),
                    "character_count": len(text),
                },
            }


def percentile(values: list[int], fraction: float) -> float:
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def summarize(values: list[int]) -> dict:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": round(percentile(ordered, 0.5), 2),
        "p95": round(percentile(ordered, 0.95), 2),
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 2),
    }


def prepare(fakebr_root: Path, fakerecogna_csv: Path, output_dir: Path) -> tuple[Path, Path]:
    if not fakebr_root.is_dir():
        raise FileNotFoundError(f"Diretório da Fake.br não encontrado: {fakebr_root}")
    if not fakerecogna_csv.is_file():
        raise FileNotFoundError(f"CSV da FakeRecogna não encontrado: {fakerecogna_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "corpus.jsonl"
    temporary_path = output_dir / "corpus.jsonl.tmp"
    report_path = output_dir / "validation_report.json"

    dataset_counts = Counter()
    label_counts = Counter()
    dataset_label_counts = Counter()
    ids = Counter()
    text_groups: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    word_lengths: dict[str, list[int]] = defaultdict(list)
    missing_metadata_ids: list[str] = []

    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as output:
            for record in chain(iter_fakebr(fakebr_root), iter_fakerecogna(fakerecogna_csv)):
                if not record["text"]:
                    raise ValueError(f"Texto vazio no registro {record['id']}")
                if record["label"] not in VALID_LABELS:
                    raise ValueError(f"Rótulo inválido no registro {record['id']}")

                fingerprint = normalized_fingerprint(record["text"])
                record["audit"]["normalized_text_sha256"] = fingerprint
                dataset = record["dataset"]
                label = record["label"]
                dataset_counts[dataset] += 1
                label_counts[label] += 1
                dataset_label_counts[(dataset, label)] += 1
                ids[record["id"]] += 1
                text_groups[fingerprint].append((record["id"], dataset, label))
                word_lengths[f"{dataset}:{label}"].append(record["audit"]["word_count"])
                if record["audit"].get("metadata_missing"):
                    missing_metadata_ids.append(record["id"])
                output.write(json.dumps(record, ensure_ascii=False) + "\n")

        if dict(dataset_counts) != EXPECTED_RECORDS:
            raise ValueError(
                f"Contagens inesperadas: {dict(dataset_counts)}; esperado: {EXPECTED_RECORDS}"
            )
        duplicate_ids = [record_id for record_id, count in ids.items() if count > 1]
        if duplicate_ids:
            raise ValueError(f"IDs duplicados: {duplicate_ids[:10]}")

        duplicate_groups = [records for records in text_groups.values() if len(records) > 1]
        conflicting_groups = [
            records for records in duplicate_groups if len({record[2] for record in records}) > 1
        ]
        cross_dataset_groups = [
            records for records in duplicate_groups if len({record[1] for record in records}) > 1
        ]

        temporary_path.replace(corpus_path)
        report = {
            "schema_version": "1.0",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "fakebr": {
                    "path": str(fakebr_root.relative_to(ROOT)),
                    "text_variant": "size_normalized_texts",
                },
                "fakerecogna2": {
                    "path": str(fakerecogna_csv.relative_to(ROOT)),
                    "sha256": file_sha256(fakerecogna_csv),
                    "text_variant": "extractive_summary",
                },
            },
            "records": {
                "total": sum(dataset_counts.values()),
                "by_dataset": dict(sorted(dataset_counts.items())),
                "by_label": dict(sorted(label_counts.items())),
                "by_dataset_and_label": {
                    f"{dataset}:{label}": dataset_label_counts[(dataset, label)]
                    for dataset in sorted(dataset_counts)
                    for label in sorted(VALID_LABELS)
                },
            },
            "quality": {
                "empty_texts": 0,
                "duplicate_ids": 0,
                "missing_fakebr_metadata": len(missing_metadata_ids),
                "missing_fakebr_metadata_ids": missing_metadata_ids,
                "normalized_duplicate_groups": len(duplicate_groups),
                "records_in_duplicate_groups": sum(len(group) for group in duplicate_groups),
                "conflicting_label_groups": len(conflicting_groups),
                "cross_dataset_duplicate_groups": len(cross_dataset_groups),
                "conflicting_examples": conflicting_groups[:5],
                "cross_dataset_examples": cross_dataset_groups[:5],
            },
            "word_count_by_dataset_and_label": {
                key: summarize(values) for key, values in sorted(word_lengths.items())
            },
            "model_input_contract": {
                "allowed_features": ["text"],
                "target": "label_id (1=fake, 0=true)",
                "excluded_from_features": [
                    "schema_version",
                    "id",
                    "dataset",
                    "source_record_id",
                    "language",
                    "text_variant",
                    "audit.*",
                ],
            },
            "output": str(corpus_path.relative_to(ROOT)),
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return corpus_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fakebr-root", type=Path, default=ROOT / "Fake.br-Corpus")
    parser.add_argument(
        "--fakerecogna-csv",
        type=Path,
        default=ROOT / "data/raw/fakerecogna2/fakerecogna_extrativo.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "data/processed/language_risk"
    )
    args = parser.parse_args()

    corpus, report = prepare(args.fakebr_root, args.fakerecogna_csv, args.output_dir)
    print(f"Corpus validado: {corpus}")
    print(f"Relatório de validação: {report}")


if __name__ == "__main__":
    main()
