"""Acquire the original textual annotations. Python >=3.10; standard library only.

No models, search APIs, baseline dependencies, or Wikipedia dump are loaded.
Existing Fake.br checkouts are read without pulling or modifying their files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
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
SOURCES = {
    "averitec/train.json": f"https://raw.githubusercontent.com/MichSchli/AVeriTeC/{AVERITEC_REV}/data/train.json",
    "averitec/dev.json": f"https://raw.githubusercontent.com/MichSchli/AVeriTeC/{AVERITEC_REV}/data/dev.json",
    "feverous/train.jsonl": "https://fever.ai/download/feverous/feverous_train_challenges.jsonl",
    "feverous/dev.jsonl": "https://fever.ai/download/feverous/feverous_dev_challenges.jsonl",
    "feverous/test.jsonl": "https://fever.ai/download/feverous/feverous_test_unlabeled.jsonl",
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


def download(url: str, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "textual-dataset-eda/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
                metadata = {"resolved_url": response.url, "etag": response.headers.get("ETag"),
                            "last_modified": response.headers.get("Last-Modified")}
                expected = response.headers.get("Content-Length")
                copied = 0
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(block)
                    copied += len(block)
                if expected is not None and copied != int(expected):
                    raise OSError(f"Incomplete download: {copied}/{expected} bytes")
            validate_annotation(partial, jsonl=path.suffix == ".jsonl")
            partial.replace(path)
            return metadata
        except (urllib.error.URLError, OSError) as error:
            if attempt == 2:
                raise RuntimeError(f"Download failed: {url}: {error}") from error
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


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
                "files": dict(previous.get("files", {}))}
    print(f"Inventory Fake.br: {args.fakebr_dir}", flush=True)
    manifest["fakebr"] = fakebr_manifest(args.fakebr_dir)
    write_json(manifest_path, manifest)
    for relative, url in SOURCES.items():
        path = args.data_dir / relative
        old = manifest["files"].get(relative)
        metadata = {}
        if path.exists():
            if old and (old["sha256"] != sha256(path) or old["source_url"] != url):
                raise ValueError(f"Source/checksum mismatch for {path}; use another --data-dir for a new acquisition")
            validate_annotation(path, path.suffix == ".jsonl")
            print(f"Verified/reused: {relative}", flush=True)
        else:
            print(f"Download: {relative}", flush=True)
            metadata = download(url, path)
            if old and old["sha256"] != sha256(path):
                raise ValueError(f"Upstream changed: {relative}; preserved old checksum in manifest")
        manifest["files"][relative] = old or {
            "source_url": url, **metadata, "sha256": sha256(path), "bytes": path.stat().st_size,
            "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
            "origin": "download" if metadata else "preexisting_file",
        }
        write_json(manifest_path, manifest)
    print(f"Acquisition complete: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
