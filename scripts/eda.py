"""Offline structural EDA of all acquired annotations; standard library only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

try:
    from .acquire import ROOT, is_feverous_header, sha256, write_json
except ImportError:
    from acquire import ROOT, is_feverous_header, sha256, write_json

TAXONOMIES = {
    "fakebr": ("fake", "true"),
    "fakerecogna2": ("fake", "true"),
    "averitec": ("Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"),
    "feverous": ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO"),
}
EVIDENCE_ID = re.compile(r"^(.+?)_(header_cell|table_caption|sentence|cell|item)_(\d+(?:_\d+)*)$")
FAKEBR_METADATA_METRICS = {
    18: "pausality",
    20: "avg_sentence_length",
    21: "avg_word_length",
    22: "spelling_error_percentage",
    23: "emotiveness",
    24: "diversity",
}


def report_path(path: Path) -> str:
    """Prefer portable project-relative paths in versioned reports."""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def date_year(value) -> str:
    if not value:
        return "<missing>"
    match = re.search(r"\b(?:19|20)\d{2}\b", str(value))
    return match.group() if match else "<unparsed>"


def iter_records(path: Path, skipped: Counter | None = None):
    """Read JSON arrays or stream JSONL; skip only explicitly identified headers."""
    with path.open(encoding="utf-8-sig") as stream:
        if path.suffix == ".jsonl":
            rows = ((number, json.loads(line)) for number, line in enumerate(stream, 1) if line.strip())
        else:
            data = json.load(stream)
            if not isinstance(data, list):
                raise ValueError(f"Expected JSON array: {path}")
            rows = enumerate(data, 1)
        for number, row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected object")
            if (path.suffix == ".jsonl" and is_feverous_header(row, number)) or ("claim" not in row and "header" in row):
                if skipped is not None:
                    skipped["header_records"] += 1
                continue
            if not isinstance(row.get("claim"), str):
                raise ValueError(f"{path}:{number}: missing/non-string claim")
            yield row


def summarize(values) -> dict:
    values = sorted(values)
    if not values:
        return {"n": 0}
    def percentile(q):
        location = (len(values) - 1) * q
        lo, hi = math.floor(location), math.ceil(location)
        return round(values[lo] + (values[hi] - values[lo]) * (location - lo), 4)
    return {"n": len(values), "min": values[0], "p50": percentile(.5), "p95": percentile(.95),
            "p99": percentile(.99), "max": values[-1], "mean": round(sum(values) / len(values), 4)}


def fingerprint(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class Schema:
    """Count actual Python types, nulls, empties and missing object properties.

    Dynamic FEVEROUS context keys become '*'; missingness for these keys is not
    meaningful. List members use '[]'. No content strings enter the schema.
    """
    def __init__(self):
        self.types = defaultdict(Counter)
        self.empty = Counter()
        self.objects = Counter()
        self.parent = {}

    def visit(self, value, path="$", parent=None):
        self.types[path][type(value).__name__] += 1
        if parent is not None:
            self.parent[path] = parent
        if value is None or (isinstance(value, str) and not value.strip()) or value == [] or value == {}:
            self.empty[path] += 1
        if isinstance(value, dict):
            self.objects[path] += 1
            for key, child in value.items():
                dynamic = path.endswith(".context")
                self.visit(child, path + "." + ("*" if dynamic else key), None if dynamic else path)
        elif isinstance(value, list):
            for child in value:
                self.visit(child, path + "[]")

    def report(self):
        result = {}
        for path, types in sorted(self.types.items()):
            count = sum(types.values())
            parent_count = self.objects[self.parent[path]] if path in self.parent else None
            result[path] = {"types": dict(types), "present": count, "null_or_empty": self.empty[path],
                            "parent_objects": parent_count,
                            "missing": parent_count - count if parent_count is not None else None}
        return result


class Profile:
    def __init__(self, dataset, split):
        self.dataset, self.split = dataset, split
        self.n = 0
        self.labels, self.quality = Counter(), Counter()
        self.metrics = defaultdict(list)
        self.categories = defaultdict(Counter)
        self.schema = Schema()
        self.hashes = defaultdict(list)
        self.ids = Counter()
        self.sample = None

    def add(self, row, index):
        self.n += 1
        self.schema.visit(row)
        if self.sample is None:
            self.sample = row
        label = row.get("label")
        if isinstance(label, str) and label.strip():
            self.labels[label] += 1
            if label not in TAXONOMIES[self.dataset]:
                self.quality["unexpected_label"] += 1
        else:
            self.quality["unlabeled"] += 1
        text = row.get("claim", row.get("text", ""))
        self.metrics["text_words_whitespace"].append(len(text.split()))
        self.metrics["text_characters"].append(len(text))
        if not text.strip():
            self.quality["empty_text"] += 1
        identifier = row.get("id", str(index))
        self.ids[str(identifier)] += 1
        self.hashes[fingerprint(text)].append({"id": identifier, "label": label, "split": self.split})

    def report(self):
        counts = {label: self.labels[label] for label in TAXONOMIES[self.dataset]}
        counts.update(self.labels)
        labeled = sum(counts.values())
        nonzero = [n for n in counts.values() if n]
        all_classes = all(counts.values())
        ratio = max(nonzero) / min(nonzero) if nonzero and all_classes else None
        if not labeled:
            decision = "Sem rótulos: balanceamento não pode ser medido."
        elif not all_classes:
            decision = "Há classes ausentes; oversampling não cria exemplos novos de uma classe ausente."
        elif ratio == 1:
            decision = "Classes balanceadas: não aplicar oversampling por frequência."
        else:
            decision = "Há desbalanceamento: avaliar pesos/amostragem apenas no treino; não fazer oversampling automaticamente."
        duplicates = [items for items in self.hashes.values() if len(items) > 1]
        conflicts = [items for items in duplicates if len({i['label'] for i in items}) > 1]
        return {"records": self.n, "labeled_records": labeled, "labels": counts,
                "percent_of_labeled": {k: round(100 * v / labeled, 3) if labeled else None for k, v in counts.items()},
                "majority_minority_ratio": round(ratio, 3) if ratio is not None else None,
                "balance_assessment": decision, "quality": dict(self.quality),
                "duplicate_id_excess": sum(n - 1 for n in self.ids.values() if n > 1),
                "normalized_duplicate_groups": len(duplicates),
                "normalized_duplicate_excess": sum(len(items) - 1 for items in duplicates),
                "conflicting_duplicate_groups": len(conflicts),
                "duplicate_examples": duplicates[:10], "conflicting_duplicate_examples": conflicts[:10],
                "metrics": {k: summarize(v) for k, v in sorted(self.metrics.items())},
                "categories": {k: dict(v.most_common()) for k, v in sorted(self.categories.items())},
                "schema": self.schema.report()}


def analyze_fakebr(path: Path, processed: Path, samples: Path) -> dict:
    result = {}
    for variant in ("full_texts", "size_normalized_texts"):
        profile = Profile("fakebr", variant)
        pairs = defaultdict(set)
        output = processed / "fakebr" / (variant + ".jsonl")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="\n") as stream:
            for label in ("fake", "true"):
                files = sorted((path / variant / label).glob("*.txt"), key=lambda p: int(p.stem))
                if not files:
                    raise ValueError(f"Missing Fake.br variant/class: {variant}/{label}")
                for file in files:
                    text = file.read_text(encoding="utf-8-sig")
                    row = {"id": f"{label}/{file.stem}", "pair_id": file.stem, "label": label,
                           "label_id": int(label == "fake"), "text": text, "language": "pt-BR",
                           "source_path": file.relative_to(path).as_posix()}
                    profile.add(row, profile.n)
                    profile.metrics[f"words_{label}"].append(len(text.split()))
                    pairs[file.stem].add(label)
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if variant == "full_texts":
                        meta = path / variant / f"{label}-meta-information" / f"{file.stem}-meta.txt"
                        if not meta.exists():
                            profile.quality["missing_metadata_file"] += 1
                            continue
                        lines = meta.read_text(encoding="utf-8-sig").splitlines()
                        if len(lines) != 25:
                            profile.quality["metadata_line_count_not_25"] += 1
                        if len(lines) >= 4:
                            profile.categories[f"category_{label}"][lines[2]] += 1
                            domain = urlsplit(lines[1]).netloc.lower()
                            profile.categories[f"source_domain_{label}"][domain or "<missing>"] += 1
                            profile.categories[f"publication_year_{label}"][date_year(lines[3])] += 1
                        for line_index, metric in FAKEBR_METADATA_METRICS.items():
                            if line_index >= len(lines):
                                continue
                            try:
                                value = float(lines[line_index].replace(",", "."))
                            except ValueError:
                                profile.quality[f"invalid_metadata_{metric}"] += 1
                            else:
                                profile.metrics[f"metadata_{metric}_{label}"].append(value)
        result[variant] = profile.report()
        result[variant]["pairs"] = {"unique_ids": len(pairs), "complete": sum(len(v) == 2 for v in pairs.values()),
                                    "incomplete_ids": [k for k, v in pairs.items() if len(v) != 2]}
        result[variant]["processed_file"] = report_path(output)
        write_json(samples / f"fakebr_{variant}.json", profile.sample)
        print(f"Fake.br/{variant}: {profile.n}", flush=True)
    csv_files = sorted((path / "preprocessed").glob("*.csv"))
    result["preprocessed_csv"] = []
    csv.field_size_limit(10_000_000)
    for file in csv_files:
        labels, count = Counter(), 0
        with file.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames
            for row in reader:
                count += 1
                labels[row.get("label", "<missing>")] += 1
        result["preprocessed_csv"].append({"file": file.name, "rows": count, "columns": columns,
                                           "types_from_csv": "str (before explicit conversion)", "labels": dict(labels)})
    return result


def analyze_fakerecogna(path: Path, samples: Path) -> dict:
    profile = Profile("fakerecogna2", "extractive")
    csv.field_size_limit(10_000_000)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for index, raw in enumerate(reader):
            raw_label = str(raw.get("Label", "")).strip()
            if raw_label not in {"0", "1"}:
                profile.quality["invalid_label"] += 1
                continue
            label = "fake" if raw_label == "1" else "true"
            text = str(raw.get("Noticia") or "")
            source_url = str(raw.get("URL") or "").strip()
            row = {
                "id": str(raw.get("Unnamed: 0") or index),
                "title": raw.get("Titulo"),
                "subtitle": raw.get("Subtitulo"),
                "category": raw.get("Categoria"),
                "author": raw.get("Autor"),
                "date": raw.get("Data"),
                "url": source_url or None,
                "text": text,
                "label": label,
                "label_id": int(raw_label),
                "language": "pt-BR",
                "text_variant": "extractive_summary",
            }
            profile.add(row, index)
            profile.metrics[f"words_{label}"].append(len(text.split()))
            profile.metrics[f"title_words_{label}"].append(len(str(raw.get("Titulo") or "").split()))
            category = str(raw.get("Categoria") or "<missing>").strip().casefold() or "<missing>"
            profile.categories[f"category_{label}"][category] += 1
            profile.categories["category_all"][category] += 1
            source = urlsplit(source_url).netloc.casefold() if source_url else "<missing>"
            profile.categories[f"source_domain_{label}"][source or "<missing>"] += 1
            profile.categories[f"publication_year_{label}"][date_year(raw.get("Data"))] += 1
    result = profile.report()
    result["input"] = {"path": report_path(path), "sha256": sha256(path), "bytes": path.stat().st_size}
    result["text_variant"] = "extractive_summary"
    write_json(samples / "fakerecogna2_extractive.json", profile.sample)
    print(f"fakerecogna2/extractive: {profile.n}; labels={dict(profile.labels)}", flush=True)
    return {"extractive": result}


def add_averitec(profile: Profile, row: dict):
    questions = row.get("questions")
    if not isinstance(questions, list):
        profile.quality["questions_missing_or_not_list"] += 1
        questions = []
    justification = row.get("justification")
    if not isinstance(justification, str) or not justification.strip():
        profile.quality["justification_missing_or_empty"] += 1
    profile.metrics["justification_words"].append(len(justification.split()) if isinstance(justification, str) else 0)
    profile.metrics["questions_per_claim"].append(len(questions))
    qa_words, answers_count = 0, 0
    for question in questions:
        if not isinstance(question, dict):
            profile.quality["question_not_object"] += 1
            continue
        qa_words += len(str(question.get("question") or "").split())
        answers = question.get("answers")
        if not isinstance(answers, list):
            profile.quality["answers_missing_or_not_list"] += 1
            answers = []
        profile.metrics["answers_per_question"].append(len(answers))
        for answer in answers:
            if not isinstance(answer, dict):
                profile.quality["answer_not_object"] += 1
                continue
            answers_count += 1
            qa_words += len(str(answer.get("answer") or "").split())
            qa_words += len(str(answer.get("boolean_explanation") or "").split())
            for key in ("answer_type", "source_medium"):
                profile.categories[key][str(answer.get(key, "<missing>"))] += 1
            url = answer.get("source_url")
            if not isinstance(url, str) or not url.strip():
                profile.quality["answer_without_source_url"] += 1
                profile.categories["missing_source_by_answer_type"][str(answer.get("answer_type"))] += 1
            else:
                profile.categories["answer_source_domains"][urlsplit(url).netloc.lower()] += 1
    profile.metrics["answers_per_claim"].append(answers_count)
    profile.metrics["claim_qa_justification_words"].append(
        len(row["claim"].split()) + qa_words + profile.metrics["justification_words"][-1])
    profile.categories["claim_year"][date_year(row.get("claim_date"))] += 1
    for key in ("location_ISO_code", "reporting_source", "speaker"):
        value = row.get(key)
        profile.categories[key][str(value).strip() if value not in (None, "") else "<missing>"] += 1
    for key in ("claim_types", "fact_checking_strategies"):
        values = row.get(key)
        if not isinstance(values, list) or not values:
            profile.categories[key]["<missing>"] += 1
            continue
        for value in values:
            profile.categories[key][str(value).strip() if value not in (None, "") else "<missing>"] += 1


def add_feverous(profile: Profile, row: dict):
    evidence = row.get("evidence")
    if not isinstance(evidence, list):
        profile.quality["evidence_missing_or_not_list"] += 1
        return
    profile.metrics["evidence_sets_per_claim"].append(len(evidence))
    if not evidence:
        profile.quality["empty_evidence"] += 1
    pages, types, ids = set(), set(), set()
    contexts = set()
    for item in evidence:
        if not isinstance(item, dict):
            profile.quality["evidence_set_not_object"] += 1
            continue
        content = item.get("content", [])
        if not isinstance(content, list):
            profile.quality["evidence_content_not_list"] += 1
            continue
        profile.metrics["elements_per_evidence_set"].append(len(content))
        for element in content:
            if not isinstance(element, str):
                profile.quality["evidence_id_not_string"] += 1
                continue
            ids.add(element)
            match = EVIDENCE_ID.fullmatch(element)
            if match:
                page, kind, _ = match.groups()
                pages.add(page)
                types.add(kind)
                profile.categories["evidence_element_type"][kind] += 1
            else:
                profile.quality["unparsed_evidence_id"] += 1
        context = item.get("context", {})
        if isinstance(context, dict):
            for linked in context.values():
                if isinstance(linked, list):
                    contexts.update(x for x in linked if isinstance(x, str))
        else:
            profile.quality["context_not_object"] += 1
    profile.metrics["unique_evidence_ids_per_claim"].append(len(ids))
    profile.metrics["unique_context_ids_per_claim"].append(len(contexts))
    profile.metrics["unique_pages_per_claim"].append(len(pages))
    profile.categories["evidence_type_combination"]["+".join(sorted(types)) or "<none>"] += 1
    profile.categories["challenge"][str(row.get("challenge", "<missing>"))] += 1


def analyze_json_dataset(name: str, directory: Path, samples: Path):
    result, split_hashes = {}, {}
    splits = ("train", "dev") if name == "averitec" else ("train", "dev", "test")
    for split in splits:
        path = directory / (split + (".json" if name == "averitec" else ".jsonl"))
        profile, skipped = Profile(name, split), Counter()
        for index, row in enumerate(iter_records(path, skipped)):
            profile.add(row, index)
            if name == "averitec":
                add_averitec(profile, row)
            else:
                add_feverous(profile, row)
        profile.quality.update(skipped)
        result[split] = profile.report()
        result[split]["input"] = {"path": report_path(path), "sha256": sha256(path), "bytes": path.stat().st_size}
        split_hashes[split] = profile.hashes
        write_json(samples / f"{name}_{split}.json", profile.sample)
        print(f"{name}/{split}: {profile.n}; labels={dict(profile.labels)}", flush=True)
    overlaps = []
    for i, left in enumerate(splits):
        for right in splits[i + 1:]:
            common = split_hashes[left].keys() & split_hashes[right].keys()
            overlaps.append({"splits": [left, right], "normalized_common_texts": len(common),
                             "examples": [{"left": split_hashes[left][h], "right": split_hashes[right][h]}
                                          for h in sorted(common)[:10]]})
    result["cross_split_overlap"] = overlaps
    return result


def render_markdown(report: dict) -> str:
    lines = ["# EDA executada — anotações textuais", "", f"Execução UTC: {report['created_at_utc']}", "",
             "Contagens calculadas sobre todos os registros adquiridos. Amostras em `samples/`.", "",
             "Palavras são unidades separadas por espaços; **não são tokens de um modelo**.", "",
             "## Rótulos por variante/split", "", "| Dataset | Split | N | Rótulo | Contagem | % dos rotulados |",
             "|---|---|---:|---|---:|---:|"]
    profiles = []
    for dataset in ("fakebr", "fakerecogna2", "averitec", "feverous"):
        for split, data in report[dataset].items():
            if not isinstance(data, dict) or "records" not in data:
                continue
            profiles.append((dataset, split, data))
            for label, count in data["labels"].items():
                percentage = data["percent_of_labeled"][label]
                lines.append(f"| {dataset} | {split} | {data['records']} | {label} | {count} | {percentage if percentage is not None else 'n/a'} |")
            if data["quality"].get("unlabeled"):
                lines.append(f"| {dataset} | {split} | {data['records']} | SEM RÓTULO | {data['quality']['unlabeled']} | n/a |")
    for dataset, split, data in profiles:
        lines += ["", f"## {dataset} / {split}", "", data["balance_assessment"], "",
                  f"Razão majoritária/minoritária: {data['majority_minority_ratio']}. "
                  f"Grupos de textos duplicados normalizados: {data['normalized_duplicate_groups']}; "
                  f"com rótulos conflitantes: {data['conflicting_duplicate_groups']}.", "",
                  f"Ocorrências de qualidade: `{json.dumps(data['quality'], ensure_ascii=False)}`.", "",
                  "### Comprimentos e cardinalidades", "", "| Métrica | N | Min | P50 | P95 | P99 | Max | Média |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for metric, stats in data["metrics"].items():
            lines.append("| " + metric + " | " + " | ".join(str(stats.get(k, "n/a")) for k in ("n", "min", "p50", "p95", "p99", "max", "mean")) + " |")
        lines += ["", "### Esquema observado", "", "Ausentes: denominador = objetos pai do campo. `*` indica chaves dinâmicas de contexto.", "",
                  "| Caminho | Tipos (ocorrências) | Ausentes | Nulos/vazios |", "|---|---|---:|---:|"]
        for path, field in data["schema"].items():
            types = ", ".join(f"{t}: {n}" for t, n in field["types"].items())
            lines.append(f"| `{path}` | {types} | {field['missing'] if field['missing'] is not None else 'n/a'} | {field['null_or_empty']} |")
    lines += ["", "## Sobreposição entre splits", ""]
    for name in ("averitec", "feverous"):
        for item in report[name]["cross_split_overlap"]:
            lines.append(f"- {name}: {' / '.join(item['splits'])}: {item['normalized_common_texts']} textos normalizados comuns.")
    lines += ["", "Método de duplicação: SHA-256 de NFKC + casefold + espaços normalizados. "
              "Isso não detecta paráfrases nem notícias sobre o mesmo evento.", "",
              "Taxonomias originais foram preservadas. Nenhum split foi reamostrado ou alterado.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--fakebr-dir", type=Path, default=ROOT / "Fake.br-Corpus")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    manifest_path = args.data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Run scripts/acquire.py before EDA to record provenance")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, item in manifest["files"].items():
        if sha256(args.data_dir / relative) != item["sha256"]:
            raise ValueError(f"Annotation checksum mismatch: {relative}")
    for item in manifest["fakebr"]["files"]:
        if sha256(args.fakebr_dir / item["path"]) != item["sha256"]:
            raise ValueError(f"Fake.br checksum mismatch: {item['path']}")
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
              "platform": platform.platform(), "manifest_sha256": sha256(manifest_path),
              "provenance": {"fakebr_revision": manifest["fakebr"]["revision"],
                             "fakebr_working_tree_status": manifest["fakebr"]["working_tree_status"],
                             "averitec_revision": manifest["averitec_revision"],
                             "fakerecogna_revision": manifest["fakerecogna_revision"]},
              "method": "All locally acquired records; JSONL streamed; CSV parsed with the standard library; no tokenizer/model loaded."}
    samples = args.output_dir / "samples"
    report["fakebr"] = analyze_fakebr(args.fakebr_dir, args.processed_dir, samples)
    report["fakerecogna2"] = analyze_fakerecogna(
        args.data_dir / "fakerecogna2" / "fakerecogna_extrativo.csv", samples
    )
    report["averitec"] = analyze_json_dataset("averitec", args.data_dir / "averitec", samples)
    report["feverous"] = analyze_json_dataset("feverous", args.data_dir / "feverous", samples)
    write_json(args.output_dir / "eda.json", report)
    (args.output_dir / "EDA.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"Reports saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
