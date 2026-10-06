"""Deduplica e divide o corpus linguístico sem vazamento entre splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPLITS = ("train", "validation", "test")
RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
SEED = 42


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(path: Path) -> list[dict]:
    records = []
    required = {"id", "dataset", "text", "label", "label_id", "audit"}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = required.difference(record)
            if missing:
                raise ValueError(f"{path}:{line_number}: campos ausentes: {sorted(missing)}")
            if not record["audit"].get("normalized_text_sha256"):
                raise ValueError(f"{path}:{line_number}: texto sem fingerprint")
            records.append(record)
    if not records:
        raise ValueError(f"Corpus vazio: {path}")
    return records


def numeric_suffix(value: str) -> int:
    match = re.search(r"(\d+)$", value)
    return int(match.group(1)) if match else math.inf


def retention_rank(record: dict) -> tuple:
    audit = record["audit"]
    completeness = sum(
        audit.get(field) is not None
        for field in ("source_domain", "source_url_sha256", "category", "published_at_raw")
    )
    return (-completeness, numeric_suffix(record["id"]), record["id"])


def deduplicate(records: list[dict]) -> tuple[list[dict], dict]:
    by_text = defaultdict(list)
    for record in records:
        by_text[record["audit"]["normalized_text_sha256"]].append(record)

    retained = []
    removals = []
    conflicts = []
    for fingerprint, group in sorted(by_text.items()):
        labels = {record["label"] for record in group}
        if len(labels) > 1:
            conflicts.append(
                {
                    "normalized_text_sha256": fingerprint,
                    "records": [
                        {
                            "id": record["id"],
                            "dataset": record["dataset"],
                            "label": record["label"],
                        }
                        for record in sorted(group, key=lambda item: item["id"])
                    ],
                }
            )
            continue

        ordered = sorted(group, key=retention_rank)
        kept = ordered[0]
        retained.append(kept)
        for removed in ordered[1:]:
            removals.append(
                {
                    "removed_id": removed["id"],
                    "kept_id": kept["id"],
                    "label": kept["label"],
                    "normalized_text_sha256": fingerprint,
                }
            )

    retained.sort(key=lambda record: record["id"])
    return retained, {
        "input_records": len(records),
        "retained_records": len(retained),
        "same_label_duplicate_records_removed": len(removals),
        "conflicting_groups_excluded": len(conflicts),
        "conflicting_records_excluded": sum(len(group["records"]) for group in conflicts),
        "same_label_removals": removals,
        "conflicts": conflicts,
    }


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def grouping_keys(record: dict):
    audit = record["audit"]
    yield "text", audit["normalized_text_sha256"]
    if record["dataset"] == "fakebr" and audit.get("pair_id"):
        yield "fakebr_pair", str(audit["pair_id"])
    if audit.get("source_url_sha256"):
        yield "source_url", audit["source_url_sha256"]


def build_groups(records: list[dict]) -> tuple[list[dict], dict]:
    union_find = UnionFind(len(records))
    owners = {}
    key_counts = Counter()

    for index, record in enumerate(records):
        for kind, value in grouping_keys(record):
            key = f"{kind}:{value}"
            key_counts[key] += 1
            if key in owners:
                union_find.union(index, owners[key])
            else:
                owners[key] = index

    components = defaultdict(list)
    for index, record in enumerate(records):
        components[union_find.find(index)].append(record)

    groups = []
    for component in components.values():
        ids = sorted(record["id"] for record in component)
        group_id = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
        strata = Counter((record["dataset"], record["label"]) for record in component)
        groups.append({"id": group_id, "records": component, "strata": strata})

    repeated_by_kind = Counter()
    for key, count in key_counts.items():
        if count > 1:
            repeated_by_kind[key.split(":", 1)[0]] += 1

    return groups, {
        "components": len(groups),
        "multi_record_components": sum(len(group["records"]) > 1 for group in groups),
        "largest_component": max(len(group["records"]) for group in groups),
        "repeated_grouping_keys": dict(sorted(repeated_by_kind.items())),
        "records_without_url_hash": sum(
            not record["audit"].get("source_url_sha256") for record in records
        ),
    }


def proportional_targets(total: int) -> dict[str, int]:
    raw = {split: total * RATIOS[split] for split in SPLITS}
    targets = {split: math.floor(raw[split]) for split in SPLITS}
    remainder = total - sum(targets.values())
    priority = sorted(SPLITS, key=lambda split: (raw[split] - targets[split], -SPLITS.index(split)), reverse=True)
    for split in priority[:remainder]:
        targets[split] += 1
    return targets


def assign_splits(groups: list[dict], seed: int) -> tuple[dict[str, list[dict]], dict]:
    strata_totals = Counter()
    record_total = 0
    for group in groups:
        strata_totals.update(group["strata"])
        record_total += len(group["records"])

    strata_targets = {
        split: {stratum: proportional_targets(total)[split] for stratum, total in strata_totals.items()}
        for split in SPLITS
    }
    total_targets_by_split = proportional_targets(record_total)
    current_strata = {split: Counter() for split in SPLITS}
    current_totals = Counter()
    assignments = {split: [] for split in SPLITS}

    rng = random.Random(seed)
    ordered_groups = list(groups)
    rng.shuffle(ordered_groups)
    ordered_groups.sort(
        key=lambda group: (
            len(group["records"]),
            max(group["strata"].values()),
            len(group["strata"]),
        ),
        reverse=True,
    )

    for group in ordered_groups:
        group_size = len(group["records"])
        candidates = []
        for split_index, split in enumerate(SPLITS):
            overflow = max(
                0, current_totals[split] + group_size - total_targets_by_split[split]
            )
            need = group_size * max(
                0, total_targets_by_split[split] - current_totals[split]
            ) / max(total_targets_by_split[split], 1)
            for stratum, count in group["strata"].items():
                target = strata_targets[split][stratum]
                after = current_strata[split][stratum] + count
                overflow += max(0, after - target)
                need += count * max(0, target - current_strata[split][stratum]) / max(target, 1)
            candidates.append((overflow, -need, split_index, split))

        split = min(candidates)[-1]
        assignments[split].append(group)
        current_totals[split] += group_size
        current_strata[split].update(group["strata"])

    output = {split: [] for split in SPLITS}
    for split in SPLITS:
        for group in assignments[split]:
            for record in group["records"]:
                record["split"] = split
                record["audit"]["split_group_sha256"] = group["id"]
                output[split].append(record)
        output[split].sort(key=lambda record: record["id"])

    allocation = {
        split: {
            "target_total": total_targets_by_split[split],
            "actual_total": current_totals[split],
            "actual_ratio": round(current_totals[split] / record_total, 6),
            "by_dataset_and_label": {
                f"{dataset}:{label}": {
                    "target": strata_targets[split][(dataset, label)],
                    "actual": current_strata[split][(dataset, label)],
                }
                for dataset, label in sorted(strata_totals)
            },
        }
        for split in SPLITS
    }
    return output, allocation


def leakage_check(splits: dict[str, list[dict]]) -> dict:
    key_splits = defaultdict(set)
    for split, records in splits.items():
        for record in records:
            for kind, value in grouping_keys(record):
                key_splits[(kind, value)].add(split)

    leaks = Counter()
    examples = defaultdict(list)
    for (kind, value), locations in key_splits.items():
        if len(locations) > 1:
            leaks[kind] += 1
            if len(examples[kind]) < 5:
                examples[kind].append({"value": value, "splits": sorted(locations)})
    return {
        "text_groups_across_splits": leaks["text"],
        "fakebr_pairs_across_splits": leaks["fakebr_pair"],
        "source_urls_across_splits": leaks["source_url"],
        "examples": dict(examples),
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def run(corpus_path: Path, output_dir: Path, seed: int) -> Path:
    records = load_records(corpus_path)
    retained, deduplication = deduplicate(records)
    groups, grouping = build_groups(retained)
    splits, allocation = assign_splits(groups, seed)
    leakage = leakage_check(splits)
    if any(value for key, value in leakage.items() if key.endswith("across_splits")):
        raise RuntimeError(f"Vazamento detectado após o split: {leakage}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = {}
    for split in SPLITS:
        path = output_dir / f"{split}.jsonl"
        write_jsonl(path, splits[split])
        output_files[split] = {
            "path": str(path.relative_to(ROOT)),
            "records": len(splits[split]),
            "sha256": file_sha256(path),
        }

    manifest = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "path": str(corpus_path.relative_to(ROOT)),
            "sha256": file_sha256(corpus_path),
        },
        "policy": {
            "ratios": RATIOS,
            "seed": seed,
            "stratification": "dataset + label",
            "grouping": ["normalized text", "Fake.br pair_id", "source URL SHA-256"],
            "same_label_duplicates": "keep the record with richer metadata, then lowest ID",
            "conflicting_label_duplicates": "exclude all records from train, validation and test",
        },
        "deduplication": deduplication,
        "grouping": grouping,
        "allocation": allocation,
        "leakage_checks": leakage,
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
                "split",
                "audit.*",
            ],
        },
        "outputs": output_files,
    }
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "data/processed/language_risk/corpus.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/processed/language_risk/splits",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    manifest = run(args.corpus, args.output_dir, args.seed)
    print(f"Splits validados: {manifest.parent}")
    print(f"Manifesto: {manifest}")


if __name__ == "__main__":
    main()
