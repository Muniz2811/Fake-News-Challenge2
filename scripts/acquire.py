"""Acquire the original textual annotations. Python >=3.10; standard library only.

No models, search APIs, baseline dependencies, or Wikipedia dump are loaded.
An existing Fake.br checkout is read without pulling its files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAKEBR_REV = "780f5516c4ae070761632d98ac3368f3ded09d35"
AVERITEC_REV = "7c62d1ec8df3fb560d6efe2b85fa191135636f81"
FAKERECOGNA_REV = "6626f2b2d3c1f88dc36a3cb108ec2c58f6a53e77"
SOURCES = {
    "averitec/train.json": f"https://raw.githubusercontent.com/MichSchli/AVeriTeC/{AVERITEC_REV}/data/train.json",
    "averitec/dev.json": f"https://raw.githubusercontent.com/MichSchli/AVeriTeC/{AVERITEC_REV}/data/dev.json",
    "feverous/train.jsonl": "https://fever.ai/download/feverous/feverous_train_challenges.jsonl",
    "feverous/dev.jsonl": "https://fever.ai/download/feverous/feverous_dev_challenges.jsonl",
    "feverous/test.jsonl": "https://fever.ai/download/feverous/feverous_test_unlabeled.jsonl",
    "fakerecogna2/fakerecogna_extrativo.csv": (
        "https://huggingface.co/datasets/recogna-nlp/fakerecogna2-extrativa/"
        f"resolve/{FAKERECOGNA_REV}/fakerecogna_extrativo.csv"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".tmp")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)


def is_feverous_header(row: dict, line_number: int) -> bool:
    """Official releases start with an object whose column values are empty."""
    return line_number == 1 and "claim" in row and all(value == "" for value in row.values())


def validate_annotation(path: Path, jsonl: bool) -> None:
    with path.open(encoding="utf-8-sig") as stream:
        if jsonl:
            claims = 0
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number}: expected object")
                if is_feverous_header(row, number):
                    continue
                if isinstance(row.get("claim"), str):
                    claims += 1
                elif "header" not in row:
                    raise ValueError(f"{path}:{number}: object has neither claim nor header")
            if claims == 0:
                raise ValueError(f"{path}: no claims")
        else:
            rows = json.load(stream)
            if not isinstance(rows, list) or not rows:
                raise ValueError(f"{path}: expected nonempty JSON array")
            if any(not isinstance(r, dict) or not isinstance(r.get("claim"), str) for r in rows):
                raise ValueError(f"{path}: invalid claim record")


def validate_fakerecogna(path: Path) -> None:
    required = {"Unnamed: 0", "Titulo", "Categoria", "Autor", "Data", "URL", "Noticia", "Label", "Subtitulo"}
    counts = {"0": 0, "1": 0}
    rows = 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: unexpected FakeRecogna columns: {reader.fieldnames}")
        for number, row in enumerate(reader, 2):
            rows += 1
            label = (row.get("Label") or "").strip()
            if label not in counts:
                raise ValueError(f"{path}:{number}: invalid label {label!r}")
            if not (row.get("Noticia") or "").strip():
                raise ValueError(f"{path}:{number}: empty news text")
            counts[label] += 1
    if rows < 50_000 or not all(counts.values()):
        raise ValueError(f"{path}: incomplete FakeRecogna dataset: rows={rows}, labels={counts}")


def validate_source(path: Path, source_path: Path | None = None) -> None:
    kind = source_path or path
    if kind.name == "fakerecogna_extrativo.csv":
        validate_fakerecogna(path)
    elif kind.suffix in {".json", ".jsonl"}:
        validate_annotation(path, jsonl=kind.suffix == ".jsonl")
    else:
        raise ValueError(f"No validator registered for {path}")


def download(url: str, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "textual-dataset-eda/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                resumed = offset > 0 and getattr(response, "status", None) == 206
                mode = "ab" if resumed else "wb"
                if not resumed:
                    offset = 0
                metadata = {"resolved_url": response.url, "etag": response.headers.get("ETag"),
                            "last_modified": response.headers.get("Last-Modified")}
                expected = response.headers.get("Content-Length")
                copied = offset
                with partial.open(mode) as output:
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        output.write(block)
                        copied += len(block)
                if expected is not None and copied != offset + int(expected):
                    raise OSError(f"Incomplete download: {copied}/{offset + int(expected)} bytes")
            validate_source(partial, source_path=path)
            partial.replace(path)
            return metadata
        except (urllib.error.URLError, OSError, ValueError) as error:
            last_error = error
            if attempt == 2:
                break
            time.sleep(2 ** attempt)
    curl = shutil.which("curl")
    if curl:
        command = [
            curl,
            "--fail",
            "--location",
            "--retry",
            "3",
            "--retry-all-errors",
            "--connect-timeout",
            "30",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            url,
        ]
        try:
            subprocess.run(command, check=True)
            validate_source(partial, source_path=path)
            partial.replace(path)
            return {"resolved_url": url, "downloader": "curl", "urllib_error": str(last_error)}
        except (subprocess.CalledProcessError, ValueError) as error:
            last_error = error
    raise RuntimeError(f"Download failed: {url}: {last_error}") from last_error


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={path.resolve().as_posix()}", "-C", str(path), *args],
        text=True, encoding="utf-8", errors="replace",
    ).strip()


def fakebr_manifest(path: Path) -> dict:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/roneysco/Fake.br-Corpus.git", str(path)], check=True)
        if git(path, "rev-parse", "HEAD") != FAKEBR_REV:
            git(path, "fetch", "--depth", "1", "origin", FAKEBR_REV)
            git(path, "checkout", "--detach", FAKEBR_REV)
    files = []
    for variant in ("full_texts", "size_normalized_texts"):
        for label in ("fake", "true"):
            found = sorted((path / variant / label).glob("*.txt"))
            if not found:
                raise ValueError(f"Missing Fake.br texts: {path / variant / label}")
            files.extend(found)
    files.extend(sorted((path / "full_texts").glob("*-meta-information/*.txt")))
    files.extend(sorted((path / "preprocessed").glob("*.csv")))
    if (path / "README.md").exists():
        files.append(path / "README.md")
    entries = [{"path": p.relative_to(path).as_posix(), "bytes": p.stat().st_size,
                "sha256": sha256(p)} for p in files]
    revision, status = None, "not a git checkout"
    if (path / ".git").exists():
        revision, status = git(path, "rev-parse", "HEAD"), git(path, "status", "--short")
    return {"path": str(path.resolve()), "source": "https://github.com/roneysco/Fake.br-Corpus",
            "revision": revision, "working_tree_status": status, "files": entries}


def acquire_file(relative: str, url: str, data_dir: Path, manifest: dict) -> None:
    path = data_dir / relative
    old = manifest["files"].get(relative)
    metadata = {}
    if path.exists():
        if old and (old["sha256"] != sha256(path) or old["source_url"] != url):
            raise ValueError(f"Source/checksum mismatch for {path}; use another --data-dir for a new acquisition")
        validate_source(path)
        print(f"Verified/reused: {relative}", flush=True)
    else:
        print(f"Download: {relative}", flush=True)
        metadata = download(url, path)
        if old and old["sha256"] != sha256(path):
            raise ValueError(f"Upstream changed: {relative}; preserved old checksum in manifest")
    manifest["files"][relative] = old or {
        "source_url": url,
        **metadata,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        "origin": "download" if metadata else "preexisting_file",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--fakebr-dir", type=Path, default=ROOT / "Fake.br-Corpus")
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.data_dir / "manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                "platform": platform.platform(), "averitec_revision": AVERITEC_REV,
                "fakerecogna_revision": FAKERECOGNA_REV,
                "files": {key: value for key, value in previous.get("files", {}).items() if key in SOURCES}}
    print(f"Inventory Fake.br: {args.fakebr_dir}", flush=True)
    manifest["fakebr"] = fakebr_manifest(args.fakebr_dir)
    write_json(manifest_path, manifest)
    for relative, url in SOURCES.items():
        acquire_file(relative, url, args.data_dir, manifest)
        write_json(manifest_path, manifest)
    print(f"Acquisition complete: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
