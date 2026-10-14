"""Regressões da nova baseline sem baixar ou executar o BERT no CI."""

import json
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "ml"))

from extract_bertimbau_embeddings import extract_split  # noqa: E402
from evaluate_bertimbau_mlp import evaluate_test  # noqa: E402
from train_bertimbau_logreg import load_embeddings, train  # noqa: E402
from train_bertimbau_mlp import build_model, train as train_mlp  # noqa: E402


class FakeEncoder:
    dimension = 3
    max_tokens = 256

    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.asarray(
            [
                [2.0 if text.startswith("fake") else -2.0, float(len(text)) / 100, 0.5]
                for text in texts
            ],
            dtype=np.float32,
        )


class TestBertimbauBaseline(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(
            prefix="test-bertimbau-", dir=ROOT / "data"
        )
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)
        self.splits = self.path / "splits"
        self.splits.mkdir()
        self.embeddings = self.path / "embeddings"
        self.encoder = FakeEncoder()
        for split in ("train", "validation"):
            rows = []
            for band, word_count in enumerate((30, 100, 200)):
                for label in (0, 1):
                    rows.append(
                        {
                            "id": f"{split}:{band}:{label}",
                            "dataset": "fakebr",
                            "text": f"{'fake' if label else 'true'} notícia {band}",
                            "label_id": label,
                            "split": split,
                            "audit": {"word_count": word_count},
                        }
                    )
            (self.splits / f"{split}.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

    def test_resume_keeps_order_and_checks_source_hash(self):
        split = self.splits / "train.jsonl"
        extract_split(split, self.embeddings, self.encoder, 2, "fixture")
        self.assertEqual(len(self.encoder.calls), 3)
        progress_path = self.embeddings / "train.json"
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        progress["completed"] = 4
        progress["complete"] = False
        progress_path.write_text(json.dumps(progress), encoding="utf-8")
        self.encoder.calls.clear()
        extract_split(split, self.embeddings, self.encoder, 2, "fixture")
        self.assertEqual(len(self.encoder.calls), 1)
        self.assertEqual(len(self.encoder.calls[0]), 2)
        records, vectors, _ = load_embeddings(split, self.embeddings)
        self.assertEqual(len(records), len(vectors))
        del vectors
        altered = np.load(self.embeddings / "train.npy", mmap_mode="r+")
        altered[0, 0] += 1
        altered.flush()
        del altered
        with self.assertRaisesRegex(ValueError, "alterados"):
            load_embeddings(split, self.embeddings)
        split.write_text(split.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "mudou"):
            load_embeddings(split, self.embeddings)

    def test_train_compares_same_splits_without_test_data(self):
        manifests = {}
        for split in ("train", "validation"):
            manifests[split] = extract_split(
                self.splits / f"{split}.jsonl",
                self.embeddings,
                self.encoder,
                2,
                "fixture",
            )
        baseline_path = self.path / "baseline.json"
        baseline_path.write_text(
            json.dumps(
                {
                    "experiment": "baseline_tfidf_logistic_regression",
                    "data": {
                        split: {"sha256": manifest["source_sha256"]}
                        for split, manifest in manifests.items()
                    },
                    "metrics": {
                        "overall": {
                            key: 0.5
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
                        "by_dataset": {"fakebr": {"f1_macro": 0.5}},
                        "by_length": {
                            band: {"f1_macro": 0.5}
                            for band in (
                                "short_1_50", "medium_51_150", "long_151_plus",
                            )
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        model_path = self.path / "classifier.joblib"
        metrics_path = self.path / "metrics.json"
        report = train(
            self.splits, self.embeddings, model_path, metrics_path, baseline_path
        )
        self.assertTrue(model_path.is_file())
        self.assertTrue(metrics_path.is_file())
        self.assertFalse(report["test_split_used"])
        self.assertEqual(report["metrics"]["overall"]["records"], 6)
        self.assertEqual(report["metrics"]["overall"]["f1_macro"], 1.0)
        self.assertEqual(
            report["comparison"]["delta_bertimbau_minus_tfidf"]["f1_macro"], 0.5
        )

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch não instalado")
    def test_mlp_reuses_validated_embeddings_and_reports_all_seeds(self):
        manifests = {
            split: extract_split(
                self.splits / f"{split}.jsonl",
                self.embeddings,
                self.encoder,
                2,
                "fixture",
            )
            for split in ("train", "validation")
        }
        baseline_path = self.path / "baseline.json"
        baseline_path.write_text(
            json.dumps(
                {
                    "experiment": "baseline_tfidf_logistic_regression",
                    "data": {
                        split: {"sha256": manifest["source_sha256"]}
                        for split, manifest in manifests.items()
                    },
                    "metrics": {
                        "overall": {"f1_macro": 0.5},
                        "by_dataset": {"fakebr": {"f1_macro": 0.5}},
                        "by_length": {
                            band: {"f1_macro": 0.5}
                            for band in (
                                "short_1_50", "medium_51_150", "long_151_plus"
                            )
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        logreg_path = self.path / "logreg.json"
        logreg_path.write_text(
            json.dumps(
                {
                    "data": {
                        split: {"sha256": manifest["source_sha256"]}
                        for split, manifest in manifests.items()
                    },
                    "encoder": {
                        field: manifests["train"][field]
                        for field in (
                            "model_id", "model_revision", "checkpoint_sha256",
                            "pooling", "max_tokens", "dimension",
                        )
                    },
                    "metrics": {"overall": {"f1_macro": 0.6}},
                }
            ),
            encoding="utf-8",
        )
        metrics_path = self.path / "mlp.json"
        report = train_mlp(
            self.splits, self.embeddings, self.path / "models", metrics_path,
            baseline_path, logreg_path,
            seeds=(42, 43), hidden_size=4, batch_size=4,
            max_epochs=3, patience=2, device="cpu",
        )
        self.assertFalse(report["test_split_used"])
        self.assertEqual([run["seed"] for run in report["runs"]], [42, 43])
        self.assertEqual(report["best_validation_run"]["metrics"]["records"], 6)
        self.assertEqual(report["parameters"]["architecture"], "3-4-1")
        self.assertTrue(metrics_path.is_file())
        self.assertTrue((self.path / "models/bertimbau_mlp_seed42.pt").is_file())
        self.assertTrue((self.path / "models/bertimbau_mlp_seed43.pt").is_file())
        import torch

        checkpoint = torch.load(
            self.path / "models/bertimbau_mlp_seed42.pt", weights_only=True
        )
        model = build_model(
            checkpoint["dimension"], checkpoint["hidden_size"],
            checkpoint["dropout"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        vectors = np.load(self.embeddings / "validation.npy")
        scaled = (vectors - checkpoint["scaler_mean"].numpy()) / checkpoint[
            "scaler_scale"
        ].numpy()
        with torch.inference_mode():
            probabilities = torch.sigmoid(model(torch.from_numpy(scaled))).numpy()
        predictions = (probabilities.ravel() >= checkpoint["threshold"]).astype(int)
        labels = [
            json.loads(line)["label_id"]
            for line in (self.splits / "validation.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(
            int(np.sum(predictions == labels)),
            round(report["runs"][0]["metrics"]["accuracy"] * len(labels)),
        )
        test_rows = [
            json.loads(line)
            for line in (self.splits / "validation.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        for row in test_rows:
            row["split"] = "test"
            row["id"] = row["id"].replace("validation", "test")
            row["text"] += " extra"
        test_path = self.splits / "test.jsonl"
        test_path.write_text(
            "\n".join(json.dumps(row) for row in test_rows) + "\n",
            encoding="utf-8",
        )
        extract_split(test_path, self.embeddings, self.encoder, 2, "fixture")
        test_report_path = self.path / "mlp_test.json"
        test_report = evaluate_test(
            test_path, self.embeddings, metrics_path, test_report_path
        )
        self.assertEqual(test_report["metrics"]["overall"]["records"], 6)
        self.assertEqual(
            test_report["model_selection"]["selected_seed"],
            report["best_validation_run"]["seed"],
        )
        with self.assertRaises(FileExistsError):
            evaluate_test(test_path, self.embeddings, metrics_path, test_report_path)
        changed = json.loads(logreg_path.read_text(encoding="utf-8"))
        changed["data"]["train"]["sha256"] = "incorrect"
        logreg_path.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "outro split de train"):
            train_mlp(
                self.splits, self.embeddings, self.path / "models",
                metrics_path, baseline_path, logreg_path,
                seeds=(42,), max_epochs=1, device="cpu",
            )
